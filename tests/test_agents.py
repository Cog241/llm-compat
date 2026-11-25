'''Agent profile registry, capability mapping, verification, and CLI behavior.'''

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_compat.agents import evaluate_agent, get_profile, list_profiles
from llm_compat.agents.base import AgentExecution, render_agent_text
from llm_compat.cli import main
from llm_compat.config import ProviderConfig
from llm_compat.report import CompatibilityReport
from llm_compat.results import CheckResult, FailureKind, Status, summarize
from llm_compat.utils import utc_now
from tests.mock_server import MockAPIServer


def _report(results: list[CheckResult]) -> CompatibilityReport:
    now = utc_now()
    return CompatibilityReport(
        provider_name='mock',
        base_url='https://example.com/v1',
        model='mock-chat',
        model_source='explicit',
        auth_configured=False,
        started_at=now,
        finished_at=now,
        duration_ms=1.0,
        results=tuple(results),
        summary=summarize(results),
    )


def _result(check_id: str, status: Status, kind: FailureKind | None = None) -> CheckResult:
    return CheckResult(
        id=check_id,
        name=check_id.title(),
        category='test',
        status=status,
        message=f'{check_id} message',
        failure_kind=kind,
    )


def test_registry_has_five_documented_profiles() -> None:
    profiles = list_profiles()
    assert [profile.key for profile in profiles] == [
        'aider',
        'cline',
        'codex',
        'opencode',
        'roo-code',
    ]
    for profile in profiles:
        assert profile.docs_url.startswith('https://')
        assert profile.capabilities
        assert profile.protocol in {'chat_completions', 'responses'}
        assert profile.verified_by
    assert get_profile('roo') is get_profile('roo-code')


def test_required_and_optional_capabilities_reflect_documented_needs() -> None:
    aider = get_profile('aider')
    assert aider is not None
    assert {'chat', 'system'} <= {
        item.check_id for item in aider.required_capabilities
    }
    assert all(
        not item.required for item in aider.capabilities if item.check_id == 'streaming'
    )
    assert all(
        item.check_id is None or not item.required
        for item in aider.capabilities
        if item.check_id in {'tools', 'json_schema'}
    )
    roo = get_profile('roo-code')
    assert roo is not None
    assert {'chat', 'streaming', 'system', 'tools', 'tools_stream'} <= {
        item.check_id for item in roo.required_capabilities
    }
    codex = get_profile('codex')
    assert codex is not None
    assert codex.check_ids == ()


def test_partial_when_required_capability_passes_but_another_fails() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    results = [
        _result('chat', Status.PASS),
        _result('streaming', Status.FAIL, FailureKind.UNSUPPORTED),
        _result('system', Status.PASS),
        _result('tools', Status.PASS),
        _result('tools_stream', Status.PASS),
    ]
    report = evaluate_agent(profile, _report(results))
    assert report.verdict == 'partial'
    failed = next(item for item in report.capabilities if item.capability.check_id == 'streaming')
    assert failed.status == Status.FAIL
    assert failed.suggestion is not None
    assert 'gateway' in failed.suggestion


def test_required_warning_makes_the_agent_verdict_partial() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    results = [
        _result(
            check_id,
            Status.WARN if check_id == 'streaming' else Status.PASS,
        )
        for check_id in ('chat', 'streaming', 'system', 'tools', 'tools_stream')
    ]
    report = evaluate_agent(profile, _report(results))
    assert report.verdict == 'partial'


def test_warn_only_required_capabilities_are_partial_not_inconclusive() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    report = evaluate_agent(profile, _report([_result('streaming', Status.WARN)]))
    assert report.verdict == 'partial'


def test_missing_check_is_skipped_and_makes_required_capability_unverified() -> None:
    profile = get_profile('cline')
    assert profile is not None
    report = evaluate_agent(profile, _report([_result('chat', Status.PASS)]))
    assert report.verdict == 'partial'
    skipped = [item for item in report.capabilities if item.status == Status.SKIP]
    assert skipped
    assert all('not run' in item.message for item in skipped)


def test_codex_is_inconclusive_without_responses_checks() -> None:
    profile = get_profile('codex')
    assert profile is not None
    report = evaluate_agent(profile, _report([]))
    assert report.verdict == 'inconclusive'
    assert report.verification_method == 'not_run'
    assert report.to_dict()['verification']['method'] == 'not_run'
    assert all(item.status == Status.SKIP for item in report.capabilities)
    assert 'Responses API' in report.capabilities[0].message
    text = render_agent_text(report)
    assert 'no protocol checks were run' in text
    assert 'inferred from protocol tests' not in text


def test_agent_execution_changes_verification_and_can_fail_the_verdict() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    results = [
        _result(check_id, Status.PASS)
        for check_id in ('chat', 'streaming', 'system', 'tools', 'tools_stream')
    ]
    execution = AgentExecution(
        command='agent --check',
        exit_code=0,
        timed_out=False,
        duration_ms=12.0,
        output_preview='ok',
    )
    report = evaluate_agent(profile, _report(results), execution=execution)
    assert report.verdict == 'compatible'
    assert report.verification_method == 'agent_execution'
    assert report.agent_execution_verified is True
    failed = evaluate_agent(
        profile,
        _report(results),
        execution=AgentExecution('agent --check', 1, False, 12.0, 'failed'),
    )
    assert failed.verdict == 'incompatible'
    assert failed.agent_execution_verified is False


def test_agent_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['agent', 'list', '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload['agents']) == 5
    assert {agent['key'] for agent in payload['agents']} == {
        'aider',
        'cline',
        'codex',
        'opencode',
        'roo-code',
    }
    assert all(
        capability['suggestion']
        for agent in payload['agents']
        for capability in agent['capabilities']
    )


def test_agent_alias_resolves_in_the_cli(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'roo',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--model',
            'mock-chat',
            '--dry-run',
            '--format',
            'json',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['agent']['key'] == 'roo-code'
    assert mock_server.requests == []


def test_agent_dry_run_makes_no_request(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'opencode',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--model',
            'mock-chat',
            '--dry-run',
            '--format',
            'json',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['agent']['key'] == 'opencode'
    assert payload['estimated_requests'] > 0
    assert mock_server.requests == []


def test_codex_dry_run_explains_that_no_checks_are_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'codex',
            '--provider',
            'ollama',
            '--dry-run',
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert 'no protocol checks are implemented' in output
    assert 'inconclusive' in output


def test_codex_test_without_execution_reports_not_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'codex',
            '--provider',
            'ollama',
            '--format',
            'json',
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'inconclusive'
    assert payload['verification']['method'] == 'not_run'
    assert payload['protocol_report']['results'] == []


def test_agent_test_requires_authorization_before_requests(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'opencode',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--model',
            'mock-chat',
        ]
    )
    assert code == 2
    assert mock_server.requests == []
    assert 'explicit authorization' in capsys.readouterr().err


def test_agent_test_against_mock_reports_inferred_compatibility(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'opencode',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--model',
            'mock-chat',
            '--yes',
            '--format',
            'json',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['verdict'] == 'compatible'
    assert payload['verification']['method'] == 'inferred_protocol'
    assert payload['verification']['agent_execution'] is None


def test_agent_verify_command_is_explicitly_authorized_and_reported(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        'agent',
        'test',
        '--agent',
        'codex',
        '--provider',
        'custom',
        '--base-url',
        mock_server.base_url,
        '--model',
        'mock-chat',
        '--verify-command',
        "printf 'verified'",
        '--format',
        'json',
    ]
    assert main(args) == 2
    assert mock_server.requests == []
    args.append('--yes')
    assert main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['verification']['method'] == 'agent_execution'
    assert payload['verification']['agent_execution']['status'] == 'passed'
    assert payload['verification']['agent_execution_verified'] is True
    assert payload['verdict'] == 'inconclusive'


def test_agent_rejects_empty_verify_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            'agent',
            'test',
            '--agent',
            'codex',
            '--provider',
            'ollama',
            '--dry-run',
            '--verify-command',
            '   ',
        ]
    )
    assert code == 2
    assert 'non-empty' in capsys.readouterr().err


def test_unknown_agent_profile_is_rejected_by_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(['agent', 'test', '--agent', 'missing'])
    assert excinfo.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_agent_report_json_has_no_unredacted_secret() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    secret = 'sk-test-secret-value-123456'
    config = ProviderConfig(
        name='mock',
        base_url='https://example.com/v1',
        api_key=secret,
        model='mock-chat',
    )
    now = utc_now()
    report = CompatibilityReport(
        provider_name=config.name,
        base_url=config.base_url,
        model=config.model,
        model_source='explicit',
        auth_configured=True,
        started_at=now,
        finished_at=now,
        duration_ms=1.0,
        results=(_result('chat', Status.PASS),),
        summary=summarize([_result('chat', Status.PASS)]),
        secrets=config.sensitive_values,
    )
    agent = evaluate_agent(profile, report)
    assert secret not in json.dumps(agent.to_dict())


@pytest.mark.parametrize(
    ('check_id', 'kind'),
    [
        ('chat', FailureKind.AUTHENTICATION),
        ('streaming', FailureKind.TEMPORARY),
        ('tools', FailureKind.MODEL_BEHAVIOR),
    ],
)
def test_failure_kind_produces_actionable_suggestion(
    check_id: str,
    kind: FailureKind,
) -> None:
    profile = get_profile('opencode')
    assert profile is not None
    result = _result(check_id, Status.FAIL, kind)
    report = evaluate_agent(profile, _report([result]))
    item = next(item for item in report.capabilities if item.capability.check_id == check_id)
    assert item.suggestion is not None
    assert len(item.suggestion) > 20


def test_agent_text_is_explicit_about_inferred_verification_and_redacts() -> None:
    profile = get_profile('opencode')
    assert profile is not None
    secret = 'sk-text-secret-value-123456'
    config = ProviderConfig(
        name='mock',
        base_url='https://example.com/v1',
        api_key=secret,
        model=f'model-{secret}',
    )
    now = utc_now()
    result = _result('chat', Status.FAIL, FailureKind.AUTHENTICATION)
    result = CheckResult(
        id=result.id,
        name=result.name,
        category=result.category,
        status=result.status,
        message=f'credential {secret} rejected',
        failure_kind=result.failure_kind,
    )
    report = CompatibilityReport(
        provider_name=config.name,
        base_url=config.base_url,
        model=config.model,
        model_source='explicit',
        auth_configured=True,
        started_at=now,
        finished_at=now,
        duration_ms=1.0,
        results=(result,),
        summary=summarize([result]),
        secrets=config.sensitive_values,
    )
    rendered = render_agent_text(evaluate_agent(profile, report), color=True)
    assert 'actual agent has not been run' in rendered
    assert secret not in rendered
    assert 'Fix:' in rendered

def test_agent_profile_to_dict_is_json_safe() -> None:
    payload: dict[str, Any] = get_profile('opencode').to_dict()  # type: ignore[union-attr]
    json.dumps(payload)
    assert payload['key'] == 'opencode'
