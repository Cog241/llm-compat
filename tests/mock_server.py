'''A stateful local mock of an OpenAI-compatible HTTP API for tests.'''

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlsplit


class MockAPIServer:
    '''A local HTTP server with configurable success and failure responses.'''

    def __init__(self) -> None:
        self.api_key: Optional[str] = None
        self.models: list[str] = ['mock-chat', 'mock-embedding']
        self.fail_status: Optional[int] = None
        self.fail_message = 'Injected failure'
        self.invalid_models_json = False
        self.invalid_models_shape = False
        self.models_entries_invalid = False
        self.invalid_chat_json = False
        self.support_tools = True
        self.tool_call_id: Optional[str] = 'call_mock_1'
        self.tool_call_invalid_arguments = False
        self.duplicate_parallel_ids = False
        self.parallel_missing_city = False
        self.parallel_extra_call = False
        self.forced_extra_call = False
        self.follow_up_ignores_tool_result = False
        self.follow_up_new_tool_call = False
        self.ignore_system_message = False
        self.reject_reasoning = False
        self.reject_stream_options = False
        self.support_json = True
        self.strict_schema_extra_field = False
        self.strict_schema_missing_tags = False
        self.empty_choices = False
        self.error_payload = False
        self.missing_message_object = False
        self.invalid_content_type = False
        self.stream_done = True
        self.stream_finish_reason = True
        self.stream_content = 'one two three four five'
        self.stream_invalid_json = False
        self.stream_chunk_not_object = False
        self.stream_invalid_content_type = False
        self.stream_no_events = False
        self.stream_tool_finish_reason: Optional[str] = 'tool_calls'
        self.stream_tool_arguments = '{"city":"Paris"}'
        self.include_usage = True
        self.partial_usage = False
        self.zero_completion_usage = False
        self.redirect_to: Optional[str] = None
        self.redirect_once = False
        self.delay = 0.0
        self._requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(('127.0.0.1', 0), _MockHandler)
        self._httpd.daemon_threads = True
        self._httpd.mock = self  # type: ignore[attr-defined]
        self._httpd_url = f'http://127.0.0.1:{self._httpd.server_port}/v1'
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={'poll_interval': 0.05},
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return self._httpd_url

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    def record(self, request: dict[str, Any]) -> None:
        with self._lock:
            self._requests.append(request)

    def __enter__(self) -> 'MockAPIServer':
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)
        return False


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle('GET')

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._handle('POST')

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def mock(self) -> MockAPIServer:
        return self.server.mock  # type: ignore[attr-defined, no-any-return]

    def _handle(self, method: str) -> None:
        length = int(self.headers.get('Content-Length', '0') or '0')
        raw_body = self.rfile.read(length) if length else b''
        try:
            body = json.loads(raw_body) if raw_body else None
        except (json.JSONDecodeError, ValueError):
            body = None
        path = urlsplit(self.path).path
        self.mock.record(
            {
                'method': method,
                'path': path,
                'headers': {key.lower(): value for key, value in self.headers.items()},
                'body': body,
                'raw_body': raw_body.decode('utf-8', errors='replace'),
            }
        )
        if self.mock.fail_status is not None:
            self._send_json(
                self.mock.fail_status,
                {'error': {'message': self.mock.fail_message, 'type': 'mock_error'}},
            )
            return
        if self.mock.redirect_to is not None:
            location = self.mock.redirect_to
            if self.mock.redirect_once:
                self.mock.redirect_to = None
            self.send_response(302)
            self.send_header('Location', location)
            self.send_header('Content-Length', '0')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            return
        if not self._authorized():
            self._send_json(
                401,
                {'error': {'message': 'Invalid API key provided', 'code': 'invalid_api_key'}},
            )
            return
        if method == 'GET' and path.rstrip('/').endswith('/models'):
            self._handle_models()
            return
        if method == 'POST' and path.rstrip('/').endswith('/chat/completions'):
            self._handle_chat(body)
            return
        self._send_json(404, {'error': {'message': f'No mock route for {method} {path}'}})

    def _authorized(self) -> bool:
        if self.mock.api_key is None:
            return True
        return self.headers.get('Authorization') == f'Bearer {self.mock.api_key}'

    def _handle_models(self) -> None:
        if self.mock.invalid_models_json:
            self._send_raw(200, b'not-json', 'application/json')
            return
        if self.mock.invalid_models_shape:
            self._send_json(200, {'object': 'list', 'data': 'not-a-list'})
            return
        if self.mock.models_entries_invalid:
            self._send_json(200, {'object': 'list', 'data': [1, None]})
            return
        data = [
            {'id': model, 'object': 'model', 'created': 1700000000, 'owned_by': 'mock'}
            for model in self.mock.models
        ]
        self._send_json(200, {'object': 'list', 'data': data})

    def _handle_chat(self, body: Any) -> None:
        if not isinstance(body, dict):
            self._send_json(400, {'error': {'message': 'Request body must be a JSON object'}})
            return
        if self.mock.invalid_chat_json:
            self._send_raw(200, b'not-json', 'application/json')
            return
        if self.mock.error_payload:
            self._send_json(
                200,
                {'error': {'message': 'Provider-side error in a 200 response'}},
            )
            return
        if body.get('stream') and body.get('stream_options') and self.mock.reject_stream_options:
            self._send_json(
                400,
                {'error': {'message': 'stream_options is not supported', 'type': 'invalid_request'}},
            )
            return
        if body.get('stream'):
            self._handle_stream(body)
            return
        messages = body.get('messages')
        messages = messages if isinstance(messages, list) else []
        tool_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get('role') == 'tool'
        ]
        if tool_messages:
            if self.mock.follow_up_new_tool_call and self.mock.support_tools:
                message = {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [self._weather_call('call_mock_follow_up')],
                }
                self._send_completion(body, message, 'tool_calls')
                return
            content = (
                'The tool result was ignored.'
                if self.mock.follow_up_ignores_tool_result
                else 'The weather result is TOOL_RESULT_18C.'
            )
            self._send_completion(body, {'role': 'assistant', 'content': content})
            return
        system_text = ' '.join(
            str(message.get('content') or '')
            for message in messages
            if isinstance(message, dict) and message.get('role') == 'system'
        )
        if 'azure' in system_text.casefold():
            content = 'pong' if self.mock.ignore_system_message else 'azure'
            self._send_completion(body, {'role': 'assistant', 'content': content})
            return
        if body.get('max_completion_tokens') is not None and 'temperature' not in body:
            if self.mock.reject_reasoning:
                self._send_json(
                    400,
                    {'error': {'message': 'max_completion_tokens is not supported'}},
                )
                return
            self._send_completion(body, {'role': 'assistant', 'content': '4'})
            return
        content = 'pong'
        finish_reason = 'stop'
        message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if body.get('tools') and self.mock.support_tools:
            tool_names = {
                tool.get('function', {}).get('name')
                for tool in body.get('tools', [])
                if isinstance(tool, dict)
            }
            prompt = ' '.join(
                str(item.get('content') or '')
                for item in messages
                if isinstance(item, dict) and item.get('role') == 'user'
            )
            if 'get_time' in tool_names and 'get_time' in prompt.casefold():
                second_id = (
                    'call_mock_1' if self.mock.duplicate_parallel_ids else 'call_mock_2'
                )
                calls = [
                    self._weather_call('call_mock_1'),
                    {
                        'id': second_id,
                        'type': 'function',
                        'function': {
                            'name': 'get_time',
                            'arguments': json.dumps(
                                {} if self.mock.parallel_missing_city else {'city': 'Paris'}
                            ),
                        },
                    },
                ]
                if self.mock.parallel_extra_call:
                    calls.append(self._weather_call('call_mock_3'))
                message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
            else:
                calls = [self._weather_call(self.mock.tool_call_id)]
                if self.mock.forced_extra_call:
                    calls.append(self._weather_call('call_mock_2'))
                message = {'role': 'assistant', 'content': None, 'tool_calls': calls}
            finish_reason = 'tool_calls'
        elif body.get('response_format'):
            response_format = body.get('response_format') or {}
            if not self.mock.support_json:
                content = 'This is not JSON.'
            elif response_format.get('type') == 'json_schema':
                payload: dict[str, Any] = {'name': 'compat', 'version': 2}
                if not self.mock.strict_schema_missing_tags:
                    payload['tags'] = ['strict']
                if self.mock.strict_schema_extra_field:
                    payload['extra'] = True
                content = json.dumps(payload)
            else:
                content = json.dumps({'name': 'llm-compat', 'version': 1})
            message = {'role': 'assistant', 'content': content}
        self._send_completion(body, message, finish_reason)

    def _weather_call(self, call_id: Optional[str]) -> dict[str, Any]:
        arguments = (
            '{"city":'
            if self.mock.tool_call_invalid_arguments
            else json.dumps({'city': 'Paris'})
        )
        call: dict[str, Any] = {
            'type': 'function',
            'function': {'name': 'get_weather', 'arguments': arguments},
        }
        if call_id is not None:
            call['id'] = call_id
        return call

    def _send_completion(
        self,
        body: dict[str, Any],
        message: dict[str, Any],
        finish_reason: str = 'stop',
    ) -> None:
        if self.mock.invalid_content_type:
            message = {**message, 'content': 123}
        choice_message: Any = None if self.mock.missing_message_object else message
        response: dict[str, Any] = {
            'id': 'chatcmpl-mock',
            'object': 'chat.completion',
            'created': 1700000000,
            'model': body.get('model', 'mock-chat'),
            'choices': (
                []
                if self.mock.empty_choices
                else [
                    {'index': 0, 'message': choice_message, 'finish_reason': finish_reason}
                ]
            ),
        }
        if self.mock.include_usage:
            response['usage'] = (
                {'prompt_tokens': 5, 'total_tokens': 12}
                if self.mock.partial_usage
                else {
                    'prompt_tokens': 5,
                    'completion_tokens': 0 if self.mock.zero_completion_usage else 7,
                    'total_tokens': 12,
                }
            )
        self._send_json(200, response)
    def _handle_stream(self, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        base = {
            'id': 'chatcmpl-mock',
            'object': 'chat.completion.chunk',
            'created': 1700000000,
            'model': body.get('model', 'mock-chat'),
        }
        try:
            if self.mock.delay:
                time.sleep(self.mock.delay)
            if self.mock.stream_no_events:
                return
            if self.mock.stream_invalid_json:
                self.wfile.write(b'data: not-json\n\n')
                self.wfile.flush()
                return
            if self.mock.stream_chunk_not_object:
                self.wfile.write(b'data: []\n\n')
                self.wfile.flush()
                return
            if self.mock.stream_invalid_content_type:
                self._write_sse(
                    {
                        **base,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'content': 123},
                                'finish_reason': None,
                            }
                        ],
                    }
                )
                return
            self._write_sse(
                {
                    **base,
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'role': 'assistant', 'content': ''},
                            'finish_reason': None,
                        }
                    ],
                }
            )
            tools = body.get('tools') if self.mock.support_tools else None
            if tools:
                arguments = self.mock.stream_tool_arguments
                split = max(1, len(arguments) // 2)
                pieces = [arguments[:split], arguments[split:]]
                call: dict[str, Any] = {
                    'index': 0,
                    'type': 'function',
                    'function': {'name': 'get_weather', 'arguments': ''},
                }
                if self.mock.tool_call_id is not None:
                    call['id'] = self.mock.tool_call_id
                self._write_sse(
                    {
                        **base,
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'tool_calls': [call]},
                                'finish_reason': None,
                            }
                        ],
                    }
                )
                for piece in pieces:
                    if not piece:
                        continue
                    self._write_sse(
                        {
                            **base,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {
                                        'tool_calls': [
                                            {
                                                'index': 0,
                                                'function': {'arguments': piece},
                                            }
                                        ]
                                    },
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                finish_reason = self.mock.stream_tool_finish_reason
            else:
                tokens = self.mock.stream_content.split(' ') if self.mock.stream_content else []
                for index, token in enumerate(tokens):
                    content = token if index == len(tokens) - 1 else f'{token} '
                    self._write_sse(
                        {
                            **base,
                            'choices': [
                                {
                                    'index': 0,
                                    'delta': {'content': content},
                                    'finish_reason': None,
                                }
                            ],
                        }
                    )
                finish_reason = 'stop' if self.mock.stream_finish_reason else None
            final: dict[str, Any] = {
                **base,
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': finish_reason,
                    }
                ],
            }
            if self.mock.include_usage:
                final['usage'] = (
                    {'prompt_tokens': 5, 'total_tokens': 12}
                    if self.mock.partial_usage
                    else {
                        'prompt_tokens': 5,
                        'completion_tokens': (
                            0 if self.mock.zero_completion_usage else 7
                        ),
                        'total_tokens': 12,
                    }
                )
            self._write_sse(final)
            if self.mock.stream_done:
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
    def _write_sse(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        self.wfile.write(b'data: ' + encoded + b'\n\n')
        self.wfile.flush()

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode('utf-8')
        self._send_raw(status, body, 'application/json')

    def _send_raw(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True
