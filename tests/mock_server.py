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
        self.invalid_chat_json = False
        self.support_tools = True
        self.tool_call_id: Optional[str] = 'call_mock_1'
        self.support_json = True
        self.stream_done = True
        self.stream_finish_reason = True
        self.stream_content = 'one two three four five'
        self.include_usage = True
        self.redirect_to: Optional[str] = None
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
            self.send_response(302)
            self.send_header('Location', self.mock.redirect_to)
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
        if body.get('stream'):
            self._handle_stream(body)
            return
        content = 'pong'
        finish_reason = 'stop'
        message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if body.get('tools') and self.mock.support_tools:
            tool_call: dict[str, Any] = {
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'arguments': json.dumps({'city': 'Paris'}),
                },
            }
            if self.mock.tool_call_id is not None:
                tool_call['id'] = self.mock.tool_call_id
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [tool_call],
            }
            finish_reason = 'tool_calls'
        elif body.get('response_format'):
            if self.mock.support_json:
                content = json.dumps({'name': 'llm-compat', 'version': 1})
            else:
                content = 'This is not JSON.'
            message = {'role': 'assistant', 'content': content}
        response: dict[str, Any] = {
            'id': 'chatcmpl-mock',
            'object': 'chat.completion',
            'created': 1700000000,
            'model': body.get('model', 'mock-chat'),
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': finish_reason}
            ],
        }
        if self.mock.include_usage:
            response['usage'] = {
                'prompt_tokens': 5,
                'completion_tokens': 7,
                'total_tokens': 12,
            }
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
            tokens = self.mock.stream_content.split(' ') if self.mock.stream_content else []
            for index, token in enumerate(tokens):
                content = token if index == len(tokens) - 1 else f'{token} '
                self._write_sse(
                    {
                        **base,
                        'choices': [
                            {'index': 0, 'delta': {'content': content}, 'finish_reason': None}
                        ],
                    }
                )
            final: dict[str, Any] = {
                **base,
                'choices': [
                    {
                        'index': 0,
                        'delta': {},
                        'finish_reason': 'stop' if self.mock.stream_finish_reason else None,
                    }
                ],
            }
            if self.mock.include_usage:
                final['usage'] = {
                    'prompt_tokens': 5,
                    'completion_tokens': 7,
                    'total_tokens': 12,
                }
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
