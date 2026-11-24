'''Multi-provider comparison API and CLI tests.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_compat.cli import main
from llm_compat.compare import compare_providers, render_comparison_text
from llm_compat.config import ProviderConfig
from llm_compat.errors import AuthorizationRequiredError, ConfigurationError
from llm_compat.runner import RunOptions
from tests.mock_server import MockAPIServer


def _config(server: MockAPIServer, name: str, **overrides: Any) -> ProviderConfig:
    values: dict[str, Any] = {
        'name': name,
        'base_url': server.base_url,
        'model': 'mock-chat',
        'timeout': 5.0,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def _write_config(path: Path, targets: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({'providers': targets}), encoding='utf-8')


def test_compare_requires_authorization_before_requests(
    mock_server: MockAPIServer,
) -> None:
    config = _config(mock_server, 'one')
    with pytest.raises(AuthorizationRequiredError):
        compare_providers([config, _config(mock_server, 'two')], RunOptions(authorize=False))
    assert mock_server.requests == []


def test_compare_rejects_duplicate_names(
    mock_server: MockAPIServer,
) -> None:
    config = _config(mock_server, 'same')
    with pytest.raises(ConfigurationError, match='unique'):
        compare_providers(
            [config, _config(mock_server, 'same')],
            RunOptions(authorize=True, only=('chat',)),
        )
    assert mock_server.requests == []


def test_compare_runs_each_independent_credential(
    mock_server: MockAPIServer,
    mock_server_two: MockAPIServer,
) -> None:
    mock_server.api_key = 'first-secret'
    mock_server_two.api_key = 'second-secret'
    one = _config(mock_server, 'one', api_key='first-secret')
    two = _config(mock_server_two, 'two', api_key='second-secret')
    report = compare_providers(
        [one, two],
        RunOptions(authorize=True, samples=1, only=('auth',)),
    )
    assert len(report.entries) == 2
    assert all(entry.summary.verdict == 'compatible' for entry in report.entries)
    assert report.entries[0].results[0].status.value == 'pass'
    assert report.entries[1].results[0].status.value == 'pass'
    assert mock_server.requests[0]['headers']['authorization'] == 'Bearer first-secret'
    assert mock_server_two.requests[0]['headers']['authorization'] == 'Bearer second-secret'


def test_compare_json_has_stable_provider_sections(
    mock_server: MockAPIServer,
) -> None:
    report = compare_providers(
        [_config(mock_server, 'one'), _config(mock_server, 'two')],
        RunOptions(authorize=True, samples=1, only=('chat',)),
    )
    payload = report.to_dict()
    assert payload['schema_version'] == '1.0'
    assert payload['comparison']['provider_count'] == 2
    assert len(payload['providers']) == 2
    assert payload['providers'][0]['provider']['name'] == 'one'
    assert payload['providers'][0]['summary']['counts']['pass'] == 1
    assert 'results' in payload['providers'][0]


def test_comparison_redacts_secrets_across_provider_entries(
    mock_server: MockAPIServer,
) -> None:
    one = _config(mock_server, 'one', api_key='cross-provider-secret')
    two = _config(mock_server, 'cross-provider-secret')
    report = compare_providers(
        [one, two],
        RunOptions(authorize=True, samples=1, only=('chat',)),
    )
    assert 'cross-provider-secret' not in json.dumps(report.to_dict())
    assert 'cross-provider-secret' not in render_comparison_text(report)


def test_comparison_text_lists_non_passing_checks(
    mock_server: MockAPIServer,
) -> None:
    mock_server.support_json = False
    report = compare_providers(
        [_config(mock_server, 'one')],
        RunOptions(authorize=True, samples=1, only=('json',)),
    )
    text = render_comparison_text(report)
    assert 'provider comparison' in text
    assert 'one' in text
    assert 'json' in text
    assert 'model_behavior' in text


def test_compare_config_rejects_literal_api_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_server: MockAPIServer,
) -> None:
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [
            {
                'name': 'one',
                'base_url': mock_server.base_url,
                'model': 'mock-chat',
                'api_key': 'must-not-leak',
            }
        ],
    )
    assert main(['compare', '--config', str(config_path), '--dry-run']) == 2
    captured = capsys.readouterr()
    assert 'api_key' in captured.err
    assert 'must-not-leak' not in captured.err
    assert mock_server.requests == []


def test_compare_config_requires_explicit_key_environment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    mock_server: MockAPIServer,
) -> None:
    monkeypatch.delenv('COMPARE_SECRET_KEY', raising=False)
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [
            {
                'name': 'one',
                'base_url': mock_server.base_url,
                'model': 'mock-chat',
                'api_key_env': 'COMPARE_SECRET_KEY',
            }
        ],
    )
    assert main(['compare', '--config', str(config_path), '--dry-run']) == 2
    captured = capsys.readouterr()
    assert 'COMPARE_SECRET_KEY' in captured.err
    assert mock_server.requests == []


def test_compare_dry_run_rejects_duplicate_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_server: MockAPIServer,
) -> None:
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [
            {'name': 'same', 'base_url': mock_server.base_url, 'model': 'mock-chat'},
            {'name': 'same', 'base_url': mock_server.base_url, 'model': 'mock-chat'},
        ],
    )
    assert main(['compare', '--config', str(config_path), '--dry-run']) == 2
    assert 'unique' in capsys.readouterr().err
    assert mock_server.requests == []


def test_compare_dry_run_json_makes_no_requests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    mock_server: MockAPIServer,
    mock_server_two: MockAPIServer,
) -> None:
    monkeypatch.setenv('COMPARE_ALPHA_KEY', 'alpha-secret')
    monkeypatch.setenv('COMPARE_BETA_KEY', 'beta-secret')
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [
            {
                'name': 'alpha',
                'base_url': mock_server.base_url,
                'model': 'mock-chat',
                'api_key_env': 'COMPARE_ALPHA_KEY',
            },
            {
                'name': 'beta',
                'base_url': mock_server_two.base_url,
                'model': 'mock-chat',
                'api_key_env': 'COMPARE_BETA_KEY',
            },
        ],
    )
    assert main(
        ['compare', '--config', str(config_path), '--dry-run', '--format', 'json']
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dry_run'] is True
    assert payload['comparison']['provider_count'] == 2
    assert [provider['name'] for provider in payload['providers']] == ['alpha', 'beta']
    assert 'alpha-secret' not in json.dumps(payload)
    assert 'beta-secret' not in json.dumps(payload)
    assert mock_server.requests == []
    assert mock_server_two.requests == []


def test_compare_cli_runs_two_providers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_server: MockAPIServer,
    mock_server_two: MockAPIServer,
) -> None:
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [
            {
                'name': 'alpha',
                'base_url': mock_server.base_url,
                'model': 'mock-chat',
            },
            {
                'name': 'beta',
                'base_url': mock_server_two.base_url,
                'model': 'mock-chat',
            },
        ],
    )
    code = main(
        [
            'compare',
            '--config',
            str(config_path),
            '--only',
            'chat',
            '--samples',
            '1',
            '--yes',
            '--format',
            'json',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload['providers']) == 2
    assert all(
        provider['summary']['verdict'] == 'compatible' for provider in payload['providers']
    )
    assert len(mock_server.requests) == 1
    assert len(mock_server_two.requests) == 1


def test_compare_cli_json_requires_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_server: MockAPIServer,
) -> None:
    config_path = tmp_path / 'targets.json'
    _write_config(
        config_path,
        [{'name': 'alpha', 'base_url': mock_server.base_url, 'model': 'mock-chat'}],
    )
    assert main(['compare', '--config', str(config_path), '--format', 'json']) == 2
    captured = capsys.readouterr()
    assert 'explicit authorization' in captured.err
    assert mock_server.requests == []


def test_compare_rejects_unknown_top_level_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'targets.json'
    path.write_text(
        json.dumps({'providers': [], 'unexpected': True}),
        encoding='utf-8',
    )
    assert main(['compare', '--config', str(path), '--dry-run']) == 2
    assert 'Unknown comparison config field' in capsys.readouterr().err


def test_compare_rejects_invalid_json_without_echoing_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / 'targets.json'
    path.write_text('{not-json with secret sk-abcdefghijklmnop', encoding='utf-8')
    assert main(['compare', '--config', str(path), '--dry-run']) == 2
    captured = capsys.readouterr()
    assert 'Invalid JSON' in captured.err
    assert 'sk-abcdefghijklmnop' not in captured.err


def test_compare_rejects_provider_sequence_length_mismatch(
    mock_server: MockAPIServer,
) -> None:
    config = _config(mock_server, 'one')
    with pytest.raises(ConfigurationError, match='match the configs'):
        compare_providers(
            [config],
            RunOptions(authorize=True, samples=1, only=('chat',)),
            providers=[],
        )