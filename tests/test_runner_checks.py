'''End-to-end checks and runner behavior against the local mock API.'''

from __future__ import annotations

import socket
from typing import Any

import pytest

from llm_compat.config import ProviderConfig
from llm_compat.errors import AuthorizationRequiredError, ConfigurationError
from llm_compat.providers.registry import build_provider
from llm_compat.results import Status
from llm_compat.runner import RunOptions, run_compatibility
from tests.mock_server import MockAPIServer


def _run(
    config: ProviderConfig,
    **options: Any,
) -> Any:
    return run_compatibility(
        config,
        RunOptions(authorize=True, **options),
        provider=build_provider(config),
    )


def test_all_checks_pass_against_compatible_endpoint(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    report = _run(config_factory(), samples=2)
    statuses = {result.id: result.status for result in report.results}
    assert statuses == {
        'connectivity': Status.PASS,
        'auth': Status.PASS,
        'models': Status.PASS,
        'chat': Status.PASS,
        'streaming': Status.PASS,
        'latency': Status.PASS,
        'tools': Status.PASS,
        'json': Status.PASS,
    }
    assert report.summary.verdict == 'compatible'
    assert report.summary.score == 100.0
    assert report.model == 'mock-chat'
    assert report.model_source == 'explicit'
    assert report.auth_configured is False


def test_authorization_is_required_before_any_request(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    with pytest.raises(AuthorizationRequiredError):
        run_compatibility(config_factory(), RunOptions(authorize=False))
    assert mock_server.requests == []


def test_model_is_auto_selected_from_discovery(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.models = ['mock-embedding', 'mock-chat']
    report = _run(config_factory(model=None), samples=1)
    assert report.model == 'mock-chat'
    assert report.model_source == 'auto'
    chat_requests = [
        request
        for request in mock_server.requests
        if request['path'].endswith('/chat/completions')
    ]
    assert chat_requests
    assert all(request['body']['model'] == 'mock-chat' for request in chat_requests)


def test_tool_calling_failure_is_a_compatibility_failure(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.support_tools = False
    report = _run(config_factory(), only=('tools',), samples=1)
    result = report.results[0]
    assert result.status == Status.FAIL
    assert 'did not return a tool call' in result.message
    assert report.summary.verdict == 'incompatible'


def test_tool_call_without_id_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.tool_call_id = None
    report = _run(config_factory(), only=('tools',), samples=1)
    result = report.results[0]
    assert result.status == Status.FAIL
    assert 'usable id' in result.message


def test_structured_json_failure(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.support_json = False
    report = _run(config_factory(), only=('json',), samples=1)
    result = report.results[0]
    assert result.status == Status.FAIL
    assert 'not valid JSON' in result.message


def test_stream_without_done_marker_warns(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_done = False
    report = _run(config_factory(), only=('streaming',), samples=1)
    result = report.results[0]
    assert result.status == Status.WARN
    assert result.details['done'] is False
    assert report.summary.verdict == 'partial'


def test_stream_with_single_content_chunk_warns(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_content = 'one'
    report = _run(config_factory(), only=('streaming',), samples=1)
    result = report.results[0]
    assert result.status == Status.WARN
    assert result.details['content_chunks'] == 1
    assert 'incremental streaming' in result.message


def test_stream_without_finish_reason_warns(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_finish_reason = False
    report = _run(config_factory(), only=('streaming',), samples=1)
    result = report.results[0]
    assert result.status == Status.WARN
    assert result.details['done'] is True
    assert result.details['finish_reason'] is None
    assert 'finish reason' in result.message


def test_missing_usage_warns_but_measures_latency(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.include_usage = False
    report = _run(config_factory(), only=('latency',), samples=1)
    result = report.results[0]
    assert result.status == Status.WARN
    assert result.details['throughput_tokens_per_sec'] is None
    assert result.details['latency_ms']['avg'] > 0


def test_empty_model_list_skips_model_checks(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.models = []
    report = _run(
        config_factory(model=None),
        only=('connectivity', 'auth', 'models', 'chat'),
        samples=1,
    )
    statuses = {result.id: result.status for result in report.results}
    assert statuses['connectivity'] == Status.PASS
    assert statuses['auth'] == Status.PASS
    assert statuses['models'] == Status.FAIL
    assert statuses['chat'] == Status.SKIP
    assert report.model is None


def test_rejected_api_key_is_reported_as_auth_failure(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.api_key = 'secret'
    report = _run(
        config_factory(model='mock-chat'),
        only=('connectivity', 'auth', 'models'),
        samples=1,
    )
    statuses = {result.id: result.status for result in report.results}
    assert statuses['connectivity'] == Status.PASS
    assert statuses['auth'] == Status.FAIL
    assert statuses['models'] == Status.FAIL
    auth = next(result for result in report.results if result.id == 'auth')
    assert auth.error is not None
    assert auth.error.status_code == 401
    assert 'Invalid API key' in auth.message


def test_network_failure_is_an_error(
    config_factory: Any,
) -> None:
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    config = config_factory(
        base_url=f'http://127.0.0.1:{port}/v1',
        timeout=0.5,
    )
    report = _run(config, only=('connectivity',), samples=1)
    result = report.results[0]
    assert result.status == Status.ERROR
    assert result.error is not None
    assert result.error.kind == 'network'


def test_server_error_is_an_error_with_status(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.fail_status = 500
    report = _run(config_factory(), only=('chat',), samples=1)
    result = report.results[0]
    assert result.status == Status.ERROR
    assert result.error is not None
    assert result.error.status_code == 500
    assert result.error.retryable is True


def test_unparsable_models_payload_warns_auth(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_models_json = True
    report = _run(config_factory(), only=('auth',), samples=1)
    result = report.results[0]
    assert result.status == Status.WARN
    assert 'could not be parsed' in result.message


def test_unknown_check_ids_are_rejected(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    with pytest.raises(ConfigurationError):
        _run(config_factory(), only=('does-not-exist',), samples=1)
    assert mock_server.requests == []


def test_skipping_everything_is_rejected(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    all_ids = (
        'connectivity',
        'auth',
        'models',
        'chat',
        'streaming',
        'latency',
        'tools',
        'json',
    )
    with pytest.raises(ConfigurationError):
        _run(config_factory(), skip=all_ids, samples=1)
    assert mock_server.requests == []
