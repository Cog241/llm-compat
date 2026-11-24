'''Regression tests for v0.2 checks and reporting behavior.'''

from __future__ import annotations

from typing import Any

from llm_compat.checks.performance import PerformanceCheck
from llm_compat.config import ProviderConfig
from llm_compat.errors import HTTPStatusError
from llm_compat.providers.registry import build_provider
from llm_compat.results import FailureKind, Status
from llm_compat.runner import RunOptions, run_compatibility
from tests.mock_server import MockAPIServer


def _run(config: ProviderConfig, **options: Any) -> Any:
    return run_compatibility(
        config,
        RunOptions(authorize=True, **options),
        provider=build_provider(config),
    )


def _one(config: ProviderConfig, check_id: str, **options: Any) -> Any:
    options.setdefault('samples', 1)
    report = _run(config, only=(check_id,), **options)
    return report.results[0]


def test_optional_checks_are_excluded_by_default(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    report = _run(config_factory(), samples=1)
    assert len(report.results) == 8
    assert all(result.id not in {
        'system',
        'tools_forced',
        'tools_parallel',
        'tools_stream',
        'json_schema',
        'reasoning',
    } for result in report.results)


def test_all_checks_pass_against_extended_mock(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    report = _run(config_factory(), samples=1, all_checks=True)
    statuses = {result.id: result.status for result in report.results}
    assert len(report.results) == 14
    assert set(statuses) == {
        'connectivity',
        'auth',
        'models',
        'chat',
        'streaming',
        'latency',
        'tools',
        'json',
        'system',
        'tools_forced',
        'tools_parallel',
        'tools_stream',
        'json_schema',
        'reasoning',
    }
    assert all(status == Status.PASS for status in statuses.values()), statuses


def test_tool_result_round_trip_is_sent_back(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'tools')
    assert result.status == Status.PASS
    chat_requests = [
        request
        for request in mock_server.requests
        if request['path'].endswith('/chat/completions')
    ]
    assert len(chat_requests) == 2
    follow_up = chat_requests[1]['body']
    roles = [message['role'] for message in follow_up['messages']]
    assert roles.count('tool') == 1
    tool_message = next(
        message for message in follow_up['messages'] if message['role'] == 'tool'
    )
    assert tool_message['tool_call_id'] == 'call_mock_1'
    assert 'TOOL_RESULT_18C' in tool_message['content']
    assert result.details['follow_up']['content_preview']


def test_tool_follow_up_that_ignores_result_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.follow_up_ignores_tool_result = True
    result = _one(config_factory(), 'tools')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'did not use the returned tool result' in result.message


def test_tool_follow_up_that_calls_again_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.follow_up_new_tool_call = True
    result = _one(config_factory(), 'tools')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'another tool' in result.message


def test_forced_tool_selection_passes(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'tools_forced')
    assert result.status == Status.PASS
    assert result.details['tool_choice'] == 'get_weather'
    request = mock_server.requests[-1]['body']
    assert request['tool_choice'] == {'type': 'function', 'function': {'name': 'get_weather'}}


def test_forced_tool_selection_with_extra_call_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.forced_extra_call = True
    result = _one(config_factory(), 'tools_forced')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'exactly' in result.message


def test_parallel_tool_calls_pass(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'tools_parallel')
    assert result.status == Status.PASS
    assert len(result.details['tool_calls']) == 2
    assert {call['name'] for call in result.details['tool_calls']} == {
        'get_weather',
        'get_time',
    }


def test_parallel_tool_calls_with_extra_call_fail(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.parallel_extra_call = True
    result = _one(config_factory(), 'tools_parallel')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'exactly two' in result.message


def test_parallel_tool_calls_with_duplicate_ids_fail(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.duplicate_parallel_ids = True
    result = _one(config_factory(), 'tools_parallel')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'unique' in result.message


def test_parallel_tool_calls_with_missing_arguments_fail(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.parallel_missing_city = True
    result = _one(config_factory(), 'tools_parallel')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'city argument' in result.message


def test_streamed_tool_calls_pass(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'tools_stream')
    assert result.status == Status.PASS
    assert result.details['done'] is True
    assert result.details['finish_reason'] == 'tool_calls'
    assert result.details['tool_calls'][0]['name'] == 'get_weather'


def test_streamed_tool_calls_without_tool_finish_warn(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_tool_finish_reason = 'stop'
    result = _one(config_factory(), 'tools_stream')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'finish reason' in result.message


def test_streamed_tool_calls_without_done_warn(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_done = False
    result = _one(config_factory(), 'tools_stream')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'DONE' in result.message


def test_strict_json_schema_passes(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'json_schema')
    assert result.status == Status.PASS
    assert result.details['strict'] is True
    request = mock_server.requests[-1]['body']
    assert request['response_format']['json_schema']['strict'] is True


def test_strict_json_schema_extra_field_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.strict_schema_extra_field = True
    result = _one(config_factory(), 'json_schema')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'exactly' in result.message


def test_strict_json_schema_missing_field_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.strict_schema_missing_tags = True
    result = _one(config_factory(), 'json_schema')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'exactly' in result.message


def test_system_message_is_honored(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'system')
    assert result.status == Status.PASS
    request = mock_server.requests[-1]['body']
    assert request['messages'][0]['role'] == 'system'


def test_ignored_system_message_fails(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.ignore_system_message = True
    result = _one(config_factory(), 'system')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert 'system-role instruction' in result.message


def test_reasoning_request_shape_passes(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'reasoning')
    assert result.status == Status.PASS
    request = mock_server.requests[-1]['body']
    assert request['max_completion_tokens'] == 64
    assert 'temperature' not in request
    assert request['messages'][0]['role'] == 'user'


def test_reasoning_request_rejection_is_unsupported(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.reject_reasoning = True
    result = _one(config_factory(), 'reasoning')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.UNSUPPORTED
    assert result.error is not None
    assert result.error.status_code == 400


def test_authentication_failure_classification(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.api_key = 'secret'
    result = _one(config_factory(), 'auth')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.AUTHENTICATION


def test_temporary_failure_classification(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.fail_status = 503
    result = _one(config_factory(model='mock-chat'), 'chat')
    assert result.status == Status.ERROR
    assert result.failure_kind == FailureKind.TEMPORARY


def test_malformed_response_classification(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_chat_json = True
    result = _one(config_factory(), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert result.error is not None
    assert result.error.kind == 'protocol'


def test_unsupported_response_classification(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.fail_status = 400
    result = _one(config_factory(model='mock-chat'), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.UNSUPPORTED


def test_empty_choices_is_malformed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.empty_choices = True
    result = _one(config_factory(), 'chat')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'choices' in result.message


def test_malformed_stream_chunk_is_malformed(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_invalid_json = True
    result = _one(config_factory(), 'streaming')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MALFORMED
    assert 'valid JSON' in result.message


def test_performance_reports_ttft_median_and_requires_p95_sample_size(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'latency', samples=2)
    assert result.status == Status.PASS
    assert result.details['latency_ms']['median'] > 0
    assert result.details['p95_latency_ms'] is None
    assert result.details['ttft_ms']['median'] > 0
    assert result.details['generation_tokens_per_sec']['avg'] > 0
    assert result.details['end_to_end_tokens_per_sec']['avg'] > 0


def test_performance_p95_with_twenty_samples(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    result = _one(config_factory(), 'latency', samples=20)
    assert result.status == Status.PASS
    assert result.details['p95_latency_ms'] is not None
    assert result.details['latency_ms']['p95'] is not None
    assert result.details['latency_ms']['samples'] == 20


def test_performance_falls_back_when_stream_options_are_rejected(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.reject_stream_options = True
    result = _one(config_factory(), 'latency')
    assert result.status == Status.PASS
    assert result.details['stream_options_supported'] is False
    assert result.details['generation_tokens_per_sec']['avg'] > 0


def test_generic_400_is_not_treated_as_stream_options_rejection() -> None:
    error = HTTPStatusError(400, 'bad request', body='invalid model')
    assert PerformanceCheck._is_stream_options_rejection(error) is False


def test_empty_performance_stream_is_model_behavior(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.stream_content = ''
    result = _one(config_factory(), 'latency')
    assert result.status == Status.FAIL
    assert result.failure_kind == FailureKind.MODEL_BEHAVIOR
    assert result.details['failures'][0]['error']['type'] == 'EmptyStream'


def test_missing_usage_is_a_malformed_warning(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.include_usage = False
    result = _one(config_factory(), 'latency')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED


def test_partial_usage_is_a_malformed_warning(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.partial_usage = True
    result = _one(config_factory(), 'latency')
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED
    assert result.details['usage_missing_samples'] == 1


def test_zero_completion_usage_counts_once_per_sample(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.zero_completion_usage = True
    result = _one(config_factory(), 'latency', samples=3)
    assert result.status == Status.WARN
    assert result.failure_kind == FailureKind.MALFORMED
    assert result.details['usage_missing_samples'] == 3


def test_summary_counts_failure_classes(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    mock_server.invalid_chat_json = True
    report = _run(config_factory(), only=('chat',), samples=1)
    assert report.summary.failure_kinds == {'malformed': 1}


def test_optional_checks_can_be_skipped(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    report = _run(
        config_factory(),
        samples=1,
        all_checks=True,
        skip=('reasoning', 'json_schema'),
    )
    ids = {result.id for result in report.results}
    assert 'reasoning' not in ids
    assert 'json_schema' not in ids
    assert 'tools_forced' in ids