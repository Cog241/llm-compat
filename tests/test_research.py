'''Research configuration, runner, exports, resume, and CLI behavior.'''

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from llm_compat.cli import main
from llm_compat.errors import ConfigurationError
from llm_compat.research import ExperimentConfig, ResearchTarget, load_experiment
from llm_compat.research.config import experiment_from_dict
from llm_compat.research.export import (
    build_summary,
    csv_safe,
    write_requests_csv,
    write_results_csv,
)
from llm_compat.research.models import TrialResult
from llm_compat.research.runner import run_experiment
from tests.mock_server import MockAPIServer


def _target(server: MockAPIServer, name: str = 'mock', **overrides: Any) -> ResearchTarget:
    values: dict[str, Any] = {
        'name': name,
        'base_url': server.base_url,
        'model': 'mock-chat',
    }
    values.update(overrides)
    return ResearchTarget(**values)


def _config(
    server: MockAPIServer,
    output_dir: Path,
    *,
    trials: int = 2,
    budget: int = 20,
    checks: tuple[str, ...] = ('chat',),
    names: tuple[str, ...] = ('mock',),
) -> ExperimentConfig:
    return ExperimentConfig(
        name='Test experiment',
        targets=tuple(_target(server, name) for name in names),
        checks=checks,
        trials=trials,
        samples=1,
        max_tokens=64,
        randomize_order=False,
        request_budget=budget,
        output_dir=str(output_dir),
    )


def test_config_defaults_and_validation() -> None:
    config = experiment_from_dict(
        {
            'name': 'defaults',
            'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
        }
    )
    assert config.trials == 3
    assert config.samples == 3
    assert len(config.checks) == 8
    assert config.resume is True
    assert config.request_budget == 100


def test_all_checks_config_round_trips_without_combining_fields() -> None:
    config = experiment_from_dict(
        {
            'name': 'all checks',
            'all_checks': True,
            'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
        }
    )
    assert len(config.checks) == 14
    payload = config.to_dict()
    assert payload['all_checks'] is True
    assert payload['checks'] == []
    assert experiment_from_dict(payload).checks == config.checks


def test_all_checks_config_can_be_rebased_to_another_output_dir() -> None:
    config = experiment_from_dict(
        {
            'name': 'all checks',
            'all_checks': True,
            'output_dir': 'original-output',
            'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
        }
    )
    rebased = config.with_output_dir('other-output')
    assert rebased.output_dir == 'other-output'
    assert rebased.all_checks is True
    assert rebased.checks == config.checks


def test_config_rejects_literal_credentials() -> None:
    with pytest.raises(ConfigurationError, match='api_key is not accepted'):
        experiment_from_dict(
            {
                'name': 'bad',
                'targets': [
                    {
                        'name': 'target',
                        'base_url': 'https://example.com/v1',
                        'api_key': 'sk-never-store-me',
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('trials', 0, 'trials'),
        ('trials', 1001, 'trials'),
        ('request_budget', 0, 'request_budget'),
        ('rate_limit_per_second', -1, 'rate_limit_per_second'),
        ('rate_limit_per_second', True, 'rate_limit_per_second'),
        ('randomize_order', 'yes', 'randomize_order'),
        ('all_checks', 'yes', 'all_checks'),
    ],
)
def test_config_rejects_invalid_values(field: str, value: Any, message: str) -> None:
    payload: dict[str, Any] = {
        'name': 'bad',
        'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
        field: value,
    }
    with pytest.raises(ConfigurationError, match=message):
        experiment_from_dict(payload)


def test_config_rejects_non_target_values() -> None:
    with pytest.raises(ConfigurationError, match='ResearchTarget'):
        ExperimentConfig(name='bad', targets=('not-a-target',))


def test_config_rejects_duplicate_targets_and_unknown_checks() -> None:
    with pytest.raises(ConfigurationError, match='unique'):
        experiment_from_dict(
            {
                'name': 'dupes',
                'targets': [
                    {'name': 'same', 'base_url': 'https://a.example/v1'},
                    {'name': 'same', 'base_url': 'https://b.example/v1'},
                ],
            }
        )
    with pytest.raises(ConfigurationError, match='[Uu]nknown check id'):
        experiment_from_dict(
            {
                'name': 'checks',
                'checks': ['not-a-check'],
                'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
            }
        )


@pytest.mark.parametrize('env_name', ['NOT A NAME', '1BAD', 'A=B'])
def test_target_rejects_invalid_environment_names(env_name: str) -> None:
    with pytest.raises(ConfigurationError, match='environment variable name'):
        experiment_from_dict(
            {
                'name': 'bad-env',
                'targets': [
                    {
                        'name': 'local',
                        'base_url': 'https://example.com/v1',
                        'api_key_env': env_name,
                    }
                ],
            }
        )


def test_target_env_and_model_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('RESEARCH_MODEL', 'env-model')
    monkeypatch.setenv('RESEARCH_KEY', 'sk-env-value-123456')
    config = experiment_from_dict(
        {
            'name': 'env',
            'targets': [
                {
                    'name': 'local',
                    'base_url': 'https://example.com/v1',
                    'model_env': 'RESEARCH_MODEL',
                    'api_key_env': 'RESEARCH_KEY',
                }
            ],
        }
    )
    resolved = config.resolve_targets()
    assert resolved[0].requested_model == 'env-model'
    assert resolved[0].config.api_key == 'sk-env-value-123456'
    safe = resolved[0].to_safe_dict()
    assert 'sk-env-value-123456' not in json.dumps(safe)


def test_target_normalizes_blank_optional_model() -> None:
    target = ResearchTarget(
        name='local', base_url='https://example.com/v1', model='   '
    )
    assert target.model is None


def test_target_rejects_model_and_model_env_together() -> None:
    with pytest.raises(ConfigurationError, match='either model or model_env'):
        ResearchTarget(
            name='local',
            base_url='https://example.com/v1',
            model='literal-model',
            model_env='RESEARCH_MODEL',
        )


def test_fingerprint_is_stable_when_credentials_appear(monkeypatch: pytest.MonkeyPatch) -> None:
    config = experiment_from_dict(
        {
            'name': 'fingerprint',
            'targets': [
                {
                    'name': 'local',
                    'base_url': 'https://example.com/v1',
                    'api_key_env': 'LATE_KEY',
                }
            ],
        }
    )
    without = config.fingerprint(config.resolve_targets(require_credentials=False))
    monkeypatch.setenv('LATE_KEY', 'sk-late-key-123456')
    with_key = config.fingerprint(config.resolve_targets())
    assert without == with_key


def test_experiment_run_writes_sanitized_exports(
    mock_server: MockAPIServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = 'sk-research-secret-123456789'
    monkeypatch.setenv('RESEARCH_KEY', secret)
    output = tmp_path / 'out'
    target = _target(
        mock_server,
        name=f'local target {secret}',
        api_key_env='RESEARCH_KEY',
    )
    config = ExperimentConfig(
        name=f'Sanitized experiment {secret}',
        targets=(target,),
        checks=('chat', 'streaming'),
        trials=2,
        samples=1,
        randomize_order=False,
        request_budget=20,
        output_dir=str(output),
    )
    report = run_experiment(config, authorize=True)
    assert len(report.trials) == 2
    assert report.summary['overall']['requests_used'] > 0
    expected = {
        'raw_results.json',
        'results.csv',
        'requests.csv',
        'summary.json',
        'methodology.md',
        'environment.json',
        'experiment.json',
    }
    assert {path.name for path in output.iterdir()} == expected
    for path in output.iterdir():
        if path.suffix in {'.json', '.csv', '.md'}:
            assert secret not in path.read_text(encoding='utf-8')
    raw = json.loads((output / 'raw_results.json').read_text(encoding='utf-8'))
    assert raw['tool']['name'] == 'llm-compat'
    assert raw['authorization']['explicit'] is True
    assert raw['resolved_targets'][0]['api_key_env'] == 'RESEARCH_KEY'
    assert raw['resolved_targets'][0]['auth_configured'] is True
    assert len(raw['trials']) == 2
    assert secret not in json.dumps(report.to_dict())


def test_request_records_capture_authentication_failures(
    mock_server: MockAPIServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_server.api_key = 'right-key'
    monkeypatch.setenv('WRONG_KEY', 'wrong-key-123456')
    output = tmp_path / 'auth-failure'
    config = ExperimentConfig(
        name='Auth failure',
        targets=(_target(mock_server, api_key_env='WRONG_KEY'),),
        checks=('chat',),
        trials=1,
        samples=1,
        randomize_order=False,
        request_budget=5,
        output_dir=str(output),
    )
    report = run_experiment(config, authorize=True)
    assert report.trials[0].status == 'completed'
    request = report.trials[0].requests[0]
    assert request['success'] is False
    assert request['failure_kind'] == 'authentication'
    assert request['status_code'] == 401
    assert request['response_model'] is None
    assert request['total_tokens'] is None
    assert request['started_at'] and request['finished_at']
    assert report.summary['overall']['failure_kinds']['authentication'] >= 1


def test_resume_does_not_repeat_trials_or_requests(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'resume'
    config = _config(mock_server, output, trials=2, checks=('chat',))
    first = run_experiment(config, authorize=True)
    request_count = len(mock_server.requests)
    second = run_experiment(config, authorize=True)
    assert len(first.trials) == 2
    assert len(second.trials) == 2
    assert len(mock_server.requests) == request_count
    assert second.summary['overall']['remaining_trials'] == 0


def test_resume_rejects_duplicate_trial_records(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'duplicates'
    config = _config(mock_server, output, trials=1)
    run_experiment(config, authorize=True)
    raw_path = output / 'raw_results.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    raw['trials'].append(dict(raw['trials'][0]))
    raw_path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ConfigurationError, match='duplicate trial id'):
        run_experiment(config, authorize=True)

def test_resume_rejects_a_different_experiment_fingerprint(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'fingerprint'
    run_experiment(_config(mock_server, output, trials=1), authorize=True)
    changed = _config(mock_server, output, trials=2)
    with pytest.raises(ConfigurationError, match='fingerprint'):
        run_experiment(changed, authorize=True)


def test_request_budget_stops_further_trials(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    config = _config(mock_server, tmp_path / 'budget', trials=5, budget=1)
    report = run_experiment(config, authorize=True)
    assert len(report.trials) == 1
    assert report.summary['overall']['requests_used'] == 1
    assert report.summary['overall']['budget_exhausted'] is True
    assert report.summary['overall']['remaining_trials'] == 4


def test_authorization_is_required_before_any_request(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    config = _config(mock_server, tmp_path / 'auth', trials=1)
    with pytest.raises(Exception, match='authorization'):
        run_experiment(config)
    assert mock_server.requests == []


def test_csv_safe_neutralizes_formula_prefixes() -> None:
    for value in ('=cmd', '+cmd', '-cmd', '@cmd', '\tcmd', '\rcmd'):
        assert csv_safe(value).startswith("'")
    assert csv_safe('normal') == 'normal'
    assert csv_safe(None) == ''


def test_csv_export_neutralizes_target_name(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    config = _config(mock_server, tmp_path / 'csv', trials=1, names=('=formula',))
    report = run_experiment(config, authorize=True)
    results = (Path(report.output_dir) / 'results.csv').read_text(encoding='utf-8')
    requests = (Path(report.output_dir) / 'requests.csv').read_text(encoding='utf-8')
    assert "'=formula" in results
    assert "'=formula" in requests


def _trial_with_requests(count: int, latency_ms: float = 10.0) -> TrialResult:
    requests = tuple(
        {
            'success': True,
            'latency_ms': latency_ms,
            'prompt_tokens': 2,
            'completion_tokens': 3,
            'total_tokens': 5,
            'failure_kind': None,
        }
        for _ in range(count)
    )
    return TrialResult(
        trial_id='trial-0000',
        target_name='mock',
        provider='custom',
        base_url='https://example.com/v1',
        trial_index=0,
        order_index=0,
        requested_model='mock-chat',
        model='mock-chat',
        model_source='explicit',
        started_at='2026-01-01T00:00:00+00:00',
        finished_at='2026-01-01T00:00:01+00:00',
        duration_ms=1000.0,
        status='completed',
        verdict='compatible',
        summary={'counts': {'pass': 1}, 'executed_checks': 1, 'failure_kinds': {}},
        checks=({'status': 'pass', 'failure_kind': None},),
        requests=requests,
        conditions={},
    )


def test_requests_csv_records_the_response_model(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    report = run_experiment(_config(mock_server, tmp_path / 'models', trials=1), authorize=True)
    with (Path(report.output_dir) / 'requests.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]['response_model'] == 'mock-chat'
    assert rows[0]['model'] == 'mock-chat'


def test_results_csv_leaves_unreported_tokens_blank(tmp_path: Path) -> None:
    trial = _trial_with_requests(1)
    trial = TrialResult(**{**trial.__dict__, 'requests': ({'success': True, 'latency_ms': 2.0},)})
    path = tmp_path / 'results.csv'
    write_results_csv(path, [trial])
    with path.open(encoding='utf-8', newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['total_tokens'] == ''
    assert row['prompt_tokens'] == ''

def test_requests_csv_tolerates_missing_model_ids(tmp_path: Path) -> None:
    trial = _trial_with_requests(1)
    trial = TrialResult(
        **{
            **trial.__dict__,
            'requests': ({'success': True, 'latency_ms': 2.0, 'model_ids': None},),
        }
    )
    path = tmp_path / 'requests.csv'
    write_requests_csv(path, [trial])
    with path.open(encoding='utf-8', newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['model_ids'] == ''


def test_summary_does_not_claim_p95_below_twenty_samples() -> None:
    trial = _trial_with_requests(19)
    summary = build_summary([trial], request_budget=100, requests_used=19)
    assert summary['overall']['latency']['p95_ms'] is None
    assert summary['overall']['latency']['p95_eligible'] is False
    assert summary['overall']['requests_used'] == 19


def test_summary_reports_p95_at_twenty_samples() -> None:
    trial = _trial_with_requests(20)
    summary = build_summary([trial], request_budget=100, requests_used=20)
    assert summary['overall']['latency']['p95_ms'] == 10.0
    assert summary['overall']['latency']['p95_eligible'] is True


def test_summary_reports_tokens_only_when_provider_reports_them() -> None:
    trial = _trial_with_requests(2)
    summary = build_summary([trial], request_budget=100, requests_used=2)
    assert summary['overall']['tokens']['total_tokens'] == 10
    empty = TrialResult(
        **{
            **trial.__dict__,
            'requests': ({'success': True, 'latency_ms': 3.0},),
        }
    )
    missing = build_summary([empty], request_budget=100, requests_used=1)
    assert missing['overall']['tokens']['total_tokens'] is None
    assert missing['overall']['tokens']['total_tokens_reported_requests'] == 0


def test_research_validate_cli_makes_no_request(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'CLI validate',
                'targets': [
                    {'name': 'mock', 'base_url': mock_server.base_url, 'model': 'mock-chat'}
                ],
                'checks': ['chat'],
                'trials': 1,
                'output_dir': str(tmp_path / 'out'),
            }
        ),
        encoding='utf-8',
    )
    assert main(['research', 'validate', str(path), '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['config_fingerprint']
    assert mock_server.requests == []


def test_research_run_requires_authorization(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'CLI run',
                'targets': [
                    {'name': 'mock', 'base_url': mock_server.base_url, 'model': 'mock-chat'}
                ],
                'checks': ['chat'],
                'trials': 1,
                'output_dir': str(tmp_path / 'out'),
            }
        ),
        encoding='utf-8',
    )
    assert main(['research', 'run', str(path)]) == 2
    assert mock_server.requests == []
    assert 'explicit authorization' in capsys.readouterr().err


def test_research_run_cli_with_yes(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / 'cli-out'
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'CLI run',
                'targets': [
                    {'name': 'mock', 'base_url': mock_server.base_url, 'model': 'mock-chat'}
                ],
                'checks': ['chat'],
                'trials': 1,
                'output_dir': str(output),
            }
        ),
        encoding='utf-8',
    )
    code = main([
        'research',
        'run',
        str(path),
        '--yes',
        '--format',
        'json',
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['output_dir'] == str(output)
    assert (output / 'raw_results.json').exists()


def test_research_run_dry_run_is_offline(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'dry',
                'targets': [
                    {'name': 'mock', 'base_url': mock_server.base_url, 'model': 'mock-chat'}
                ],
                'checks': ['chat'],
                'trials': 2,
                'output_dir': str(tmp_path / 'out'),
            }
        ),
        encoding='utf-8',
    )
    override = tmp_path / 'override'
    assert (
        main([
            'research',
            'run',
            str(path),
            '--dry-run',
            '--output-dir',
            str(override),
            '--format',
            'json',
        ])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['estimated_requests'] == 2
    assert payload['output_dir'] == str(override)
    config = load_experiment(path)
    resolved = config.resolve_targets(require_credentials=False)
    assert payload['config_fingerprint'] == config.with_output_dir(
        str(override)
    ).fingerprint(resolved)


def test_all_checks_research_dry_run_is_offline(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'all checks dry',
                'targets': [
                    {'name': 'mock', 'base_url': mock_server.base_url, 'model': 'mock-chat'}
                ],
                'all_checks': True,
                'trials': 1,
                'output_dir': str(tmp_path / 'out'),
            }
        ),
        encoding='utf-8',
    )
    assert main(['research', 'run', str(path), '--dry-run', '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload['checks']) == 14
    assert payload['config_fingerprint']
    assert mock_server.requests == []


def test_load_experiment_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / 'experiment.json'
    path.write_text(
        json.dumps(
            {
                'name': 'bad',
                'targets': [{'name': 'local', 'base_url': 'https://example.com/v1'}],
                'surprise': True,
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(ConfigurationError, match='unknown field'):
        load_experiment(path)


def test_exported_environment_has_no_environment_values(
    mock_server: MockAPIServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('RESEARCH_KEY', 'sk-never-print-this')
    output = tmp_path / 'env'
    config = ExperimentConfig(
        name='env',
        targets=(_target(mock_server, api_key_env='RESEARCH_KEY'),),
        checks=('chat',),
        trials=1,
        samples=1,
        randomize_order=False,
        output_dir=str(output),
    )
    run_experiment(config, authorize=True)
    environment = json.loads((output / 'environment.json').read_text(encoding='utf-8'))
    assert 'sk-never-print-this' not in json.dumps(environment)


def test_copied_experiment_config_resumes_without_duplicates(
    mock_server: MockAPIServer,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'copy'
    config = _config(mock_server, output, trials=1)
    run_experiment(config, authorize=True)
    shutil.copyfile(output / 'experiment.json', tmp_path / 'experiment-copy.json')
    copied = load_experiment(tmp_path / 'experiment-copy.json')
    report = run_experiment(copied, authorize=True, output_dir=str(output))
    assert len(report.trials) == 1
