'''Tests for the standard-library transport and OpenAI-compatible adapter.'''

from __future__ import annotations

import socket
from typing import Any

import pytest

from llm_compat.config import ProviderConfig
from llm_compat.errors import HTTPStatusError, NetworkError, ProtocolError
from llm_compat.http import HttpClient, HttpRequest, HttpResponse, UrllibHttpClient
from llm_compat.providers.base import ModelInfo, select_chat_model
from llm_compat.providers.openai import (
    OpenAICompatibleProvider,
    _parse_tool_calls,
    _parse_usage,
)
from tests.mock_server import MockAPIServer


def test_http_client_returns_json_and_headers(mock_server: MockAPIServer) -> None:
    client: HttpClient = UrllibHttpClient()
    response = client.send(
        HttpRequest(
            method='GET',
            url=f'{mock_server.base_url}/models',
            headers={'Accept': 'application/json'},
            timeout=5.0,
        )
    )
    with response:
        assert response.status_code == 200
        payload = response.json()
    assert payload['object'] == 'list'
    assert len(payload['data']) == 2


def test_invalid_request_headers_are_reported_as_network_errors() -> None:
    with pytest.raises(NetworkError) as captured:
        UrllibHttpClient().send(
            HttpRequest(
                method='GET',
                url='https://example.com/v1/models',
                headers={
                    'Authorization': 'Bearer secret-value',
                    'X-Test': 'bad\nsecret-value',
                },
            )
        )
    assert 'secret-value' not in str(captured.value)


def test_http_client_returns_error_status_without_raising(mock_server: MockAPIServer) -> None:
    mock_server.api_key = 'secret'
    response = UrllibHttpClient().send(
        HttpRequest(method='GET', url=f'{mock_server.base_url}/models', timeout=5.0)
    )
    with response:
        assert response.status_code == 401
        assert 'Invalid API key' in response.text


def test_http_client_network_error() -> None:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with pytest.raises(NetworkError):
        UrllibHttpClient().send(
            HttpRequest(
                method='GET',
                url=f'http://127.0.0.1:{port}/v1/models',
                timeout=0.5,
            )
        )


def test_http_response_line_iteration() -> None:
    response = HttpResponse(200, {}, body=b'data: one\r\ndata: two\r\n')
    assert list(response.iter_lines()) == ['data: one', 'data: two']


def test_cross_origin_redirect_is_not_followed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    with MockAPIServer() as target:
        mock_server.api_key = 'secret'
        target.api_key = 'secret'
        mock_server.redirect_to = f'{target.base_url}/models'
        provider = OpenAICompatibleProvider(config_factory(api_key='secret'))
        with pytest.raises(HTTPStatusError) as captured:
            provider.list_models()
        assert captured.value.status_code == 302
        assert 'scheme, host, and port' in (captured.value.hint or '')
        assert target.requests == []


def test_model_discovery_is_cached(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    provider = OpenAICompatibleProvider(config_factory())
    first = provider.list_models()
    second = provider.list_models()
    assert [model.id for model in first] == ['mock-chat', 'mock-embedding']
    assert [model.id for model in second] == ['mock-chat', 'mock-embedding']
    assert len(mock_server.requests) == 1


def test_api_key_is_sent_as_bearer_header(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.api_key = 'top-secret'
    provider = OpenAICompatibleProvider(config_factory(api_key='top-secret'))
    provider.list_models()
    request = mock_server.requests[0]
    assert request['headers']['authorization'] == 'Bearer top-secret'


def test_missing_api_key_surfaces_http_error(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.api_key = 'top-secret'
    provider = OpenAICompatibleProvider(config_factory())
    with pytest.raises(HTTPStatusError) as captured:
        provider.list_models()
    assert captured.value.status_code == 401
    assert captured.value.retryable is False
    assert captured.value.hint is not None


def test_chat_completion_is_normalized(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    provider = OpenAICompatibleProvider(config_factory())
    response = provider.chat(
        model='mock-chat',
        messages=[{'role': 'user', 'content': 'ping'}],
        max_tokens=16,
    )
    assert response.content == 'pong'
    assert response.model == 'mock-chat'
    assert response.finish_reason == 'stop'
    assert response.usage.completion_tokens == 7
    assert response.latency_ms > 0
    assert response.tool_calls == ()


def test_invalid_chat_json_raises_protocol_error(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_chat_json = True
    provider = OpenAICompatibleProvider(config_factory())
    with pytest.raises(ProtocolError):
        provider.chat(
            model='mock-chat',
            messages=[{'role': 'user', 'content': 'ping'}],
            max_tokens=16,
        )


def test_streaming_events_include_content_usage_and_done(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    provider = OpenAICompatibleProvider(config_factory())
    events = list(
        provider.stream_chat(
            model='mock-chat',
            messages=[{'role': 'user', 'content': 'count'}],
            max_tokens=16,
        )
    )
    assert ''.join(event.content for event in events) == 'one two three four five'
    assert events[-1].done is True
    assert any(event.usage is not None and event.usage.completion_tokens == 7 for event in events)


def test_stream_without_done_marker_is_reported(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_done = False
    provider = OpenAICompatibleProvider(config_factory())
    events = list(
        provider.stream_chat(
            model='mock-chat',
            messages=[{'role': 'user', 'content': 'count'}],
            max_tokens=16,
        )
    )
    assert events
    assert all(not event.done for event in events)


def test_non_sse_stream_raises_protocol_error(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_chat_json = True
    provider = OpenAICompatibleProvider(config_factory())
    with pytest.raises(ProtocolError):
        list(
            provider.stream_chat(
                model='mock-chat',
                messages=[{'role': 'user', 'content': 'count'}],
                max_tokens=16,
            )
        )


def _parser_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        ProviderConfig(base_url='https://example.com/v1', timeout=1.0)
    )


def test_chat_parser_handles_content_parts_and_derives_usage() -> None:
    provider = _parser_provider()
    response = provider._parse_chat(
        {
            'model': 'mock-chat',
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': 'Hello '},
                            {'type': 'text', 'text': {'value': 'world'}},
                        ],
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 3},
        },
        latency_ms=12.5,
    )
    assert response.content == 'Hello world'
    assert response.usage.total_tokens == 5
    assert response.latency_ms == 12.5


def test_chat_parser_supports_legacy_function_call() -> None:
    response = _parser_provider()._parse_chat(
        {
            'choices': [
                {
                    'message': {
                        'content': None,
                        'function_call': {
                            'name': 'get_weather',
                            'arguments': '{"city": "Paris"}',
                        },
                    },
                    'finish_reason': 'function_call',
                }
            ]
        },
        latency_ms=1.0,
    )
    assert response.tool_calls[0].name == 'get_weather'
    assert response.tool_calls[0].id is None
    assert response.usage.prompt_tokens is None


@pytest.mark.parametrize(
    'payload',
    [
        None,
        {'choices': []},
        {'choices': ['not-an-object']},
        {'choices': [{}]},
        {'error': 'provider failed'},
    ],
)
def test_chat_parser_rejects_bad_shapes(payload: Any) -> None:
    with pytest.raises(ProtocolError):
        _parser_provider()._parse_chat(payload, latency_ms=1.0)


def test_tool_call_parser_serializes_object_arguments() -> None:
    calls = _parse_tool_calls(
        {
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'lookup', 'arguments': {'query': 'x'}},
                }
            ]
        }
    )
    assert calls[0].id == 'call-1'
    assert calls[0].arguments == '{"query": "x"}'


def test_usage_parser_ignores_booleans() -> None:
    assert _parse_usage({'prompt_tokens': True}) is None


def test_model_parser_deduplicates_and_skips_invalid_entries() -> None:
    models = _parser_provider()._parse_models(
        [{'id': 'one'}, 'two', {'id': 'one'}, {'id': '  '}, 3]
    )
    assert [model.id for model in models] == ['one', 'two']


@pytest.mark.parametrize('payload', [None, {'data': {}}, {'data': [1]}])
def test_model_parser_rejects_unusable_payloads(payload: Any) -> None:
    with pytest.raises(ProtocolError):
        _parser_provider()._parse_models(payload)


def test_stream_parser_ignores_sse_metadata() -> None:
    response = HttpResponse(
        200,
        {},
        body=(
            b': keepalive\n'
            b'event: message\n'
            b'id: 1\n'
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n'
            b'data: [DONE]\n'
        ),
    )
    events = list(_parser_provider()._read_stream(response))
    assert ''.join(event.content for event in events) == 'hi'
    assert events[-1].done is True


def test_select_chat_model_prefers_chat_models() -> None:
    models = [
        ModelInfo(id='text-embedding-3-small'),
        ModelInfo(id='mock-chat-model'),
    ]
    assert select_chat_model(models) == 'mock-chat-model'


def test_select_chat_model_returns_none_when_only_non_chat_models() -> None:
    models = [ModelInfo(id='text-embedding-3-small'), ModelInfo(id='whisper-1')]
    assert select_chat_model(models) is None
