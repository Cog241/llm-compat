'''Adapter for APIs that implement the OpenAI HTTP surface.'''

from __future__ import annotations

import http.client
import json
import time
from typing import Any, Iterator, Mapping, Optional, Sequence

from llm_compat.config import ProviderConfig
from llm_compat.errors import LLMCompatError, NetworkError, ProtocolError, build_http_error
from llm_compat.http import HttpClient, HttpRequest, HttpResponse, UrllibHttpClient
from llm_compat.providers.base import (
    ChatResponse,
    Message,
    ModelInfo,
    Provider,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    Usage,
)
from llm_compat.utils import redact_sensitive, safe_url, truncate
from llm_compat.version import __version__


def _content_text(content: Any) -> str:
    '''Normalize string content and OpenAI-style content part arrays.'''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise ProtocolError('A response content part had an invalid type.')
            text = part.get('text')
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(text, dict):
                value = text.get('value')
                if not isinstance(value, str):
                    raise ProtocolError(
                        'A response content part text value had an invalid type.'
                    )
                parts.append(value)
            elif text is not None:
                raise ProtocolError('A response content part text had an invalid type.')
        return ''.join(parts)
    if content is None:
        return ''
    raise ProtocolError('The chat response content had an invalid type.')


def _token_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_usage(payload: Any) -> Optional[Usage]:
    if not isinstance(payload, dict):
        return None
    prompt = _token_count(payload.get('prompt_tokens'))
    completion = _token_count(payload.get('completion_tokens'))
    total = _token_count(payload.get('total_tokens'))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if prompt is None and completion is None and total is None:
        return None
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _tool_call_arguments(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError(
                'A chat response tool call arguments value could not be serialized.'
            ) from exc
    raise ProtocolError('A chat response tool call arguments value had an invalid type.')


def _parse_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    raw_calls = message.get('tool_calls')
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise ProtocolError('The chat response tool_calls field was not an array.')
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ProtocolError('A chat response tool call was not a JSON object.')
            function = raw_call.get('function')
            if not isinstance(function, dict):
                raise ProtocolError(
                    'A chat response tool call did not contain a function object.'
                )
            name = function.get('name')
            if name is not None and not isinstance(name, str):
                raise ProtocolError('A chat response tool call name was not a string.')
            call_id = raw_call.get('id')
            if call_id is not None and not isinstance(call_id, str):
                raise ProtocolError('A chat response tool call id was not a string.')
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name or '',
                    arguments=_tool_call_arguments(function.get('arguments')),
                )
            )
    if not calls:
        legacy = message.get('function_call')
        if legacy is not None and not isinstance(legacy, dict):
            raise ProtocolError('The chat response function_call field was not an object.')
        if isinstance(legacy, dict):
            name = legacy.get('name')
            if name is not None and not isinstance(name, str):
                raise ProtocolError('The legacy function_call name was not a string.')
            if isinstance(name, str):
                calls.append(
                    ToolCall(
                        id=None,
                        name=name,
                        arguments=_tool_call_arguments(legacy.get('arguments')),
                    )
                )
    return tuple(calls)


def _tool_calls_from_buffers(buffers: Mapping[int, dict[str, str]]) -> tuple[ToolCall, ...]:
    '''Turn accumulated streaming fragments into complete tool calls.'''
    calls = []
    for index in sorted(buffers):
        buffer = buffers[index]
        if not buffer.get('id') and not buffer.get('name') and not buffer.get('arguments'):
            continue
        calls.append(
            ToolCall(
                id=buffer.get('id') or None,
                name=buffer.get('name', ''),
                arguments=buffer.get('arguments', ''),
            )
        )
    return tuple(calls)


class OpenAICompatibleProvider(Provider):
    '''HTTP client for endpoints that follow the OpenAI chat API shape.'''

    def __init__(self, config: ProviderConfig, http: Optional[HttpClient] = None) -> None:
        super().__init__(config)
        self.http = http or UrllibHttpClient()
        self._models_cache: Optional[tuple[ModelInfo, ...]] = None
        self._models_error: Optional[LLMCompatError] = None

    def _headers(self, *, accept: str = 'application/json') -> dict[str, str]:
        headers = {
            'Accept': accept,
            'Content-Type': 'application/json',
            'User-Agent': f'llm-compat/{__version__}',
        }
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        headers.update(self.config.extra_headers)
        return headers

    def _endpoint(self, path: str) -> str:
        return f'{self.config.base_url}{path}'

    def _send(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        accept: str = 'application/json',
    ) -> HttpResponse:
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = self.http.send(
            HttpRequest(
                method=method,
                url=self._endpoint(path),
                headers=self._headers(accept=accept),
                body=body,
                timeout=self.config.timeout,
            )
        )
        if isinstance(response, HttpResponse):
            response.secrets = self.config.sensitive_values
        return response

    def _raise_for_status(self, response: HttpResponse) -> None:
        if 200 <= response.status_code < 300:
            return
        raise build_http_error(
            response.status_code,
            response.url,
            response.text,
            secrets=self.config.sensitive_values,
            retry_after=response.headers.get('retry-after'),
        )

    def _decode_json(self, response: HttpResponse) -> Any:
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ProtocolError(
                f'Expected JSON from {safe_url(response.url)} but received non-JSON content.'
            ) from exc

    def _error_from_payload(self, payload: Any) -> Optional[str]:
        if not isinstance(payload, dict) or 'error' not in payload:
            return None
        error = payload.get('error')
        if isinstance(error, dict):
            message = error.get('message') or error.get('code')
        else:
            message = error
        if not message:
            message = 'unspecified error'
        return truncate(redact_sensitive(str(message), self.config.sensitive_values))

    def list_models(self) -> list[ModelInfo]:
        if self._models_cache is not None:
            return list(self._models_cache)
        if self._models_error is not None:
            raise self._models_error
        try:
            with self._send('GET', '/models') as response:
                self._raise_for_status(response)
                payload = self._decode_json(response)
            models = self._parse_models(payload)
        except LLMCompatError as exc:
            self._models_error = exc
            raise
        self._models_cache = tuple(models)
        return list(models)

    def _parse_models(self, payload: Any) -> list[ModelInfo]:
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = payload.get('data')
        else:
            raise ProtocolError('The models response was not a JSON object or array.')
        if not isinstance(entries, list):
            raise ProtocolError('The models response did not contain a data array.')
        models: list[ModelInfo] = []
        seen: set[str] = set()
        for entry in entries:
            if isinstance(entry, str):
                identifier = entry
                owned_by = None
                created = None
            elif isinstance(entry, dict):
                identifier = entry.get('id')
                owned_by = entry.get('owned_by') if isinstance(entry.get('owned_by'), str) else None
                raw_created = entry.get('created')
                created = (
                    raw_created
                    if isinstance(raw_created, int) and not isinstance(raw_created, bool)
                    else None
                )
            else:
                continue
            if not isinstance(identifier, str) or not identifier.strip():
                continue
            identifier = identifier.strip()
            if identifier in seen:
                continue
            seen.add(identifier)
            models.append(ModelInfo(id=identifier, owned_by=owned_by, created=created))
        if not models and entries:
            raise ProtocolError('The models response did not contain any usable model ids.')
        return models

    def _chat_payload(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        stream: bool = False,
        reasoning: bool = False,
        include_usage: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': model,
            'messages': [dict(message) for message in messages],
        }
        if reasoning:
            payload['max_completion_tokens'] = max_tokens
        else:
            payload['max_tokens'] = max_tokens
            payload['temperature'] = temperature
        if tools is not None:
            payload['tools'] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice
        if response_format is not None:
            payload['response_format'] = dict(response_format)
        if stream:
            payload['stream'] = True
            if include_usage:
                payload['stream_options'] = {'include_usage': True}
        return payload

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float = 0.0,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        reasoning: bool = False,
    ) -> ChatResponse:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning=reasoning,
        )
        started = time.perf_counter()
        with self._send('POST', '/chat/completions', payload=payload) as response:
            self._raise_for_status(response)
            data = self._decode_json(response)
            latency_ms = (time.perf_counter() - started) * 1000.0
        return self._parse_chat(data, latency_ms)

    def _parse_chat(self, payload: Any, latency_ms: float) -> ChatResponse:
        provider_error = self._error_from_payload(payload)
        if provider_error is not None:
            raise ProtocolError(f'Provider returned an error: {provider_error}')
        if not isinstance(payload, dict):
            raise ProtocolError('The chat response was not a JSON object.')
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise ProtocolError('The chat response did not contain a non-empty choices array.')
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProtocolError('The chat response choice was not a JSON object.')
        message = choice.get('message')
        if not isinstance(message, dict):
            raise ProtocolError('The chat response choice did not contain a message object.')
        raw_content = message.get('content')
        if raw_content is not None and not isinstance(raw_content, (str, list)):
            raise ProtocolError('The chat response message content had an invalid type.')
        content = _content_text(raw_content)
        tool_calls = _parse_tool_calls(message)
        finish_reason = choice.get('finish_reason')
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = None
        response_model = payload.get('model')
        if not isinstance(response_model, str):
            response_model = ''
        return ChatResponse(
            content=content,
            model=response_model,
            finish_reason=finish_reason,
            usage=_parse_usage(payload.get('usage')) or Usage(),
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )

    def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float = 0.0,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        include_usage: bool = False,
    ) -> Iterator[StreamEvent]:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            include_usage=include_usage,
        )
        with self._send(
            'POST',
            '/chat/completions',
            payload=payload,
            accept='text/event-stream',
        ) as response:
            self._raise_for_status(response)
            yield from self._read_stream(response)

    def _read_stream(self, response: HttpResponse) -> Iterator[StreamEvent]:
        saw_event = False
        tool_buffers: dict[int, dict[str, str]] = {}
        calls_emitted = False

        def accumulate(delta: Mapping[str, Any]) -> None:
            raw_calls = delta.get('tool_calls')
            if raw_calls is not None:
                if not isinstance(raw_calls, list):
                    raise ProtocolError(
                        'A streaming tool_calls delta was not an array.'
                    )
                for position, raw_call in enumerate(raw_calls):
                    if not isinstance(raw_call, dict):
                        raise ProtocolError(
                            'A streaming tool call delta was not a JSON object.'
                        )
                    index = raw_call.get('index')
                    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                        index = position
                    buffer = tool_buffers.setdefault(
                        index, {'id': '', 'name': '', 'arguments': ''}
                    )
                    call_id = raw_call.get('id')
                    if call_id is not None and not isinstance(call_id, str):
                        raise ProtocolError(
                            'A streaming tool call id was not a string.'
                        )
                    if call_id and not buffer['id']:
                        buffer['id'] = call_id
                    function = raw_call.get('function')
                    if function is not None and not isinstance(function, dict):
                        raise ProtocolError(
                            'A streaming tool call did not contain a function object.'
                        )
                    if isinstance(function, dict):
                        name = function.get('name')
                        if name is not None and not isinstance(name, str):
                            raise ProtocolError(
                                'A streaming tool call name was not a string.'
                            )
                        if name:
                            buffer['name'] += name
                        buffer['arguments'] += _tool_call_arguments(
                            function.get('arguments')
                        )
            legacy = delta.get('function_call')
            if legacy is not None and not isinstance(legacy, dict):
                raise ProtocolError(
                    'A streaming function_call delta was not a JSON object.'
                )
            if isinstance(legacy, dict):
                buffer = tool_buffers.setdefault(0, {'id': '', 'name': '', 'arguments': ''})
                name = legacy.get('name')
                if name is not None and not isinstance(name, str):
                    raise ProtocolError(
                        'A streaming function_call name was not a string.'
                    )
                if name:
                    buffer['name'] += name
                buffer['arguments'] += _tool_call_arguments(legacy.get('arguments'))

        try:
            for line in response.iter_lines():
                stripped = line.strip()
                if not stripped or stripped.startswith(':'):
                    continue
                if stripped.startswith(('event:', 'id:', 'retry:')):
                    continue
                if not stripped.startswith('data:'):
                    raise ProtocolError(
                        'The streaming response did not use Server-Sent Events. '
                        'Expected lines beginning with data:.'
                    )
                data = stripped[5:].strip()
                if not data:
                    continue
                saw_event = True
                if data == '[DONE]':
                    if tool_buffers and not calls_emitted:
                        calls = _tool_calls_from_buffers(tool_buffers)
                        if calls:
                            yield StreamEvent(tool_calls=calls)
                    yield StreamEvent(done=True)
                    return
                try:
                    chunk = json.loads(data)
                except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                    raise ProtocolError('A streaming chunk was not valid JSON.') from exc
                if not isinstance(chunk, dict):
                    raise ProtocolError('A streaming chunk was not a JSON object.')
                provider_error = self._error_from_payload(chunk)
                if provider_error is not None:
                    raise ProtocolError(f'Provider sent an error in the stream: {provider_error}')
                usage = _parse_usage(chunk.get('usage'))
                choices = chunk.get('choices')
                if not isinstance(choices, list):
                    if usage is not None:
                        yield StreamEvent(usage=usage)
                        continue
                    raise ProtocolError('A streaming chunk did not contain a choices array.')
                if not choices:
                    if usage is not None:
                        yield StreamEvent(usage=usage)
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ProtocolError('A streaming choice was not a JSON object.')
                delta = choice.get('delta')
                if delta is not None and not isinstance(delta, dict):
                    raise ProtocolError('A streaming delta was not a JSON object.')
                if isinstance(delta, dict):
                    accumulate(delta)
                    raw_content = delta.get('content')
                    if raw_content is not None and not isinstance(raw_content, (str, list)):
                        raise ProtocolError(
                            'A streaming delta content had an invalid type.'
                        )
                    content = _content_text(raw_content)
                else:
                    content = ''
                finish_reason = choice.get('finish_reason')
                if not isinstance(finish_reason, str) or not finish_reason:
                    finish_reason = None
                calls: tuple[ToolCall, ...] = ()
                if finish_reason in {'tool_calls', 'function_call'} and tool_buffers:
                    calls = _tool_calls_from_buffers(tool_buffers)
                    calls_emitted = bool(calls)
                if content or finish_reason or usage is not None or calls:
                    yield StreamEvent(
                        content=content,
                        finish_reason=finish_reason,
                        usage=usage,
                        tool_calls=calls,
                    )
            if not saw_event:
                raise ProtocolError('The stream ended without any data events.')
            if tool_buffers and not calls_emitted:
                calls = _tool_calls_from_buffers(tool_buffers)
                if calls:
                    yield StreamEvent(tool_calls=calls)
        except (OSError, http.client.HTTPException) as exc:
            detail = redact_sensitive(str(exc), self.config.sensitive_values)
            raise NetworkError(f'The streaming response was interrupted: {detail}') from exc
