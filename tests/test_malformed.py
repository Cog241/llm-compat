'''Regression tests for malformed and inconsistent provider responses.'''

from __future__ import annotations

from typing import Any

import pytest

from llm_compat.config import ProviderConfig
from llm_compat.providers.registry import build_provider
from llm_compat.results import FailureKind, Status
from llm_compat.runner import RunOptions, run_compatibility
from tests.mock_server import MockAPIServer


def _run(config: ProviderConfig, **options: Any) -> Any:
    return run_compatibility(
        config,
        RunOptions(authorize=True, samples=1, **options),
        provider=build_provider(config),
    )


def _one(config: ProviderConfig, check_id: str) -> Any:
    return _run(config, only=(check_id,)).results[0]


@pytest.mark.parametrize(
    'flag',
    ['invalid_models_json', 'invalid_models_shape', 'models_entries_invalid'],
)
def test_malformed_model_responses_are_classified(
    flag: str,
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    setattr(mock_server, flag, True)
    report = _run(config_factory(), only=('connectivity', 'auth', 'models'))
    by_id = {result.id: result for result in report.results}
    assert by_id['connectivity'].status == Status.WARN
    assert by_id['auth'].status == Status.WARN
    assert by_id['models'].status == Status.FAIL
    assert all(
        by_id[check].failure_kind == FailureKind.MALFORMED
        for check in ('connectivity', 'auth', 'models')
    )


def test_empty_model_list_is_unsupported(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.models = []
    result = _one(config_factory(), 'models')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.UNSUPPORTED
    assert 'empty' in result.message


def test_missing_models_endpoint_is_unsupported(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.fail_status = 404
    config = config_factory(model=None)
    report = _run(config, only=('connectivity', 'auth', 'models'))
    by_id = {result.id: result for result in report.results}
    assert by_id['models'].status == Status.FAIL
    assert by_id['models'].failure_kind == FailureKind.UNSUPPORTED
    assert by_id['auth'].status == Status.WARN
    assert by_id['auth'].failure_kind == FailureKind.UNSUPPORTED


def test_invalid_content_type_is_malformed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_content_type = True
    result = _one(config_factory(), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'invalid type' in result.message


def test_error_payload_with_http_200_is_malformed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.error_payload = True
    result = _one(config_factory(), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'Provider returned an error' in result.message


def test_missing_message_object_is_malformed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.missing_message_object = True
    result = _one(config_factory(), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'message object' in result.message


@pytest.mark.parametrize(
    ('flag', 'needle'),
    [
        ('stream_invalid_content_type', 'invalid type'),
        ('stream_chunk_not_object', 'not a JSON object'),
        ('stream_no_events', 'without any data events'),
    ],
)
def test_malformed_stream_shapes_are_classified(
    flag: str,
    needle: str,
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    setattr(mock_server, flag, True)
    result = _one(config_factory(), 'streaming')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert needle in result.message


def test_stream_without_finish_reason_warns(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_finish_reason = False
    result = _one(config_factory(), 'streaming')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'finish reason' in result.message


def test_single_chunk_stream_is_unsupported(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_content = 'one'
    result = _one(config_factory(), 'streaming')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.UNSUPPORTED
    assert 'incremental' in result.message


@pytest.mark.parametrize(
    ('status_code', 'expected_status', 'failure_kind'),
    [
        (401, Status.FAIL, FailureKind.AUTHENTICATION),
        (403, Status.FAIL, FailureKind.AUTHENTICATION),
        (400, Status.FAIL, FailureKind.UNSUPPORTED),
        (404, Status.FAIL, FailureKind.UNSUPPORTED),
        (422, Status.FAIL, FailureKind.UNSUPPORTED),
        (408, Status.ERROR, FailureKind.TEMPORARY),
        (429, Status.ERROR, FailureKind.TEMPORARY),
        (500, Status.ERROR, FailureKind.TEMPORARY),
        (503, Status.ERROR, FailureKind.TEMPORARY),
    ],
)
def test_http_failure_classification(
    status_code: int,
    expected_status: Status,
    failure_kind: FailureKind,
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.fail_status = status_code
    mock_server.fail_message = 'Injected provider failure'
    result = _one(config_factory(model='mock-chat'), 'chat')
    assert result.status == expected_status
    assert result.failure_kind == failure_kind
    assert result.error is not None
    assert result.error.status_code == status_code
    if status_code in {408, 429, 500, 503}:
        assert result.error.retryable is True
    else:
        assert result.error.retryable is False