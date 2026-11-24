'''Tests for summaries, JSON export, and terminal rendering.'''

from __future__ import annotations

import json

from llm_compat.report import CompatibilityReport, render_json, render_text
from llm_compat.results import CheckResult, ErrorInfo, Status, summarize


def _result(
    status: Status,
    *,
    check_id: str = 'check',
    name: str = 'Check',
    message: str = 'message',
    hint: str | None = None,
) -> CheckResult:
    error = None
    if status in {Status.FAIL, Status.ERROR}:
        error = ErrorInfo(
            type='HTTPStatusError',
            kind='http',
            message=message,
            status_code=400,
            hint=hint,
        )
    return CheckResult(
        id=check_id,
        name=name,
        category='test',
        status=status,
        message=message,
        duration_ms=12.5,
        details={'sample': True},
        error=error,
    )


def _report(results: tuple[CheckResult, ...]) -> CompatibilityReport:
    return CompatibilityReport(
        provider_name='custom',
        base_url='https://user:secret@example.com/v1?api_key=hidden',
        model='mock-chat',
        model_source='auto',
        auth_configured=True,
        started_at='2026-01-01T00:00:00.000Z',
        finished_at='2026-01-01T00:00:01.000Z',
        duration_ms=1000.0,
        results=results,
        summary=summarize(results),
    )


def test_summary_compatible() -> None:
    summary = summarize((_result(Status.PASS), _result(Status.PASS, check_id='two')))
    assert summary.verdict == 'compatible'
    assert summary.score == 100.0
    assert summary.counts['pass'] == 2


def test_summary_partial() -> None:
    summary = summarize((_result(Status.PASS), _result(Status.FAIL, check_id='two')))
    assert summary.verdict == 'partial'
    assert summary.score == 50.0


def test_summary_incompatible_when_nothing_passes() -> None:
    summary = summarize((_result(Status.FAIL), _result(Status.ERROR, check_id='two')))
    assert summary.verdict == 'incompatible'
    assert summary.score == 0.0


def test_summary_warnings_make_partial_and_skips_are_not_executed() -> None:
    summary = summarize(
        (
            _result(Status.PASS),
            _result(Status.WARN, check_id='two'),
            _result(Status.SKIP, check_id='three'),
        )
    )
    assert summary.verdict == 'partial'
    assert summary.total == 3
    assert summary.executed == 2
    assert summary.score == 50.0


def test_summary_skips_make_partial() -> None:
    summary = summarize((_result(Status.PASS), _result(Status.SKIP, check_id='two')))
    assert summary.verdict == 'partial'
    assert summary.executed == 1
    assert summary.score == 100.0


def test_summary_inconclusive_when_all_skipped() -> None:
    summary = summarize((_result(Status.SKIP),))
    assert summary.verdict == 'inconclusive'
    assert summary.executed == 0


def test_json_export_is_safe_and_complete() -> None:
    report = _report((_result(Status.PASS),))
    payload = report.to_dict()
    encoded = json.dumps(payload)
    assert payload['schema_version'] == '1.0'
    assert payload['tool']['name'] == 'llm-compat'
    assert payload['provider']['base_url'] == 'https://example.com/v1'
    assert payload['summary']['verdict'] == 'compatible'
    assert payload['results'][0]['details'] == {'sample': True}
    assert 'secret' not in encoded
    assert 'hidden' not in encoded
    assert 'api_key' not in encoded


def test_terminal_report_is_human_readable() -> None:
    result = _result(
        Status.FAIL,
        name='Tool calling',
        message='The model did not return a tool call.',
        hint='Use a model with tool support.',
    )
    text = render_text(_report((result,)))
    assert 'LLM Compat' in text
    assert 'Provider: custom' in text
    assert 'Model: mock-chat (auto-selected)' in text
    assert 'FAIL' in text
    assert 'Tool calling' in text
    assert 'The model did not return a tool call.' in text
    assert 'Hint: Use a model with tool support.' in text
    assert 'Verdict: INCOMPATIBLE' in text
    assert 'Checks:' in text


def test_terminal_report_supports_color() -> None:
    text = render_text(_report((_result(Status.PASS),)), color=True)
    assert '\x1b[32m' in text
    assert '\x1b[0m' in text


def test_terminal_report_pluralizes_warning_count() -> None:
    text = render_text(_report((_result(Status.PASS), _result(Status.WARN, check_id='two'))))
    assert '1 warning, 0 errors' in text


def test_render_json_round_trips() -> None:
    text = render_json(_report((_result(Status.PASS),)))
    payload = json.loads(text)
    assert payload['provider']['model'] == 'mock-chat'
