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
from llm_compat.utils import redact_secret, safe_url, truncate
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
            elif isinstance(part, dict):
                text = part.get('text')
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict) and isinstance(text.get('value'), str):
                    parts.append(text['value'])
        return ''.join(parts)
    return ''


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


def _parse_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    raw_calls = message.get('tool_calls')
    if isinstance(raw_calls, list):
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get('function')
            if not isinstance(function, dict):
                continue
            name = function.get('name')
            arguments = function.get('arguments')
            if not isinstance(name, str):
                name = ''
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            if not isinstance(arguments, str):
                arguments = ''
            call_id = raw_call.get('id')
            calls.append(
                ToolCall(
                    id=call_id if isinstance(call_id, str) else None,
                    name=name,
                    arguments=arguments,
                )
            )
    if not calls:
        legacy = message.get('function_call')
        if isinstance(legacy, dict) and isinstance(legacy.get('name'), str):
            arguments = legacy.get('arguments', '')
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                ToolCall(
                    id=None,
                    name=legacy['name'],
                    arguments=arguments if isinstance(arguments, str) else '',
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
        return self.http.send(
            HttpRequest(
                method=method,
                url=self._endpoint(path),
                headers=self._headers(accept=accept),
                body=body,
                timeout=self.config.timeout,
            )
        )

    def _raise_for_status(self, response: HttpResponse) -> None:
        if 200 <= response.status_code < 300:
            return
        raise build_http_error(
            response.status_code,
            response.url,
            response.text,
            secret=self.config.api_key,
            retry_after=response.headers.get('retry-after'),
        )

    def _decode_json(self, response: HttpResponse) -> Any:
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, ValueError) as exc:
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
        return redact_secret(truncate(str(message)), self.config.api_key)

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
                created = entry.get('created') if isinstance(entry.get('created'), int) else None
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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': model,
            'messages': [dict(message) for message in messages],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        if tools is not None:
            payload['tools'] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice
        if response_format is not None:
            payload['response_format'] = dict(response_format)
        if stream:
            payload['stream'] = True
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
    ) -> ChatResponse:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
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
        content = _content_text(message.get('content'))
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
    ) -> Iterator[StreamEvent]:
        payload = self._chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
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
                    yield StreamEvent(done=True)
                    return
                try:
                    chunk = json.loads(data)
                except (json.JSONDecodeError, ValueError) as exc:
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
                content = _content_text(delta.get('content')) if isinstance(delta, dict) else ''
                finish_reason = choice.get('finish_reason')
                if not isinstance(finish_reason, str) or not finish_reason:
                    finish_reason = None
                if content or finish_reason or usage is not None:
                    yield StreamEvent(
                        content=content,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
            if not saw_event:
                raise ProtocolError('The stream ended without any data events.')
        except (OSError, http.client.HTTPException) as exc:
            raise NetworkError(f'The streaming response was interrupted: {exc}') from exc
