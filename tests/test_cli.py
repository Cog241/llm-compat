'''CLI behavior, authorization, JSON output, and exit codes.'''

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from llm_compat.cli import main
from tests.mock_server import MockAPIServer


def _run_args(server: MockAPIServer, *extra: str) -> list[str]:
    return [
        'run',
        '--provider',
        'custom',
        '--base-url',
        server.base_url,
        '--model',
        'mock-chat',
        '--samples',
        '1',
        *extra,
    ]


def test_unexpected_cli_error_is_reported_without_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom(args: object) -> int:
        raise RuntimeError('secret-value')

    monkeypatch.setattr('llm_compat.cli._cmd_checks', boom)

    assert main(['checks']) == 1
    captured = capsys.readouterr()
    assert 'unexpected internal error' in captured.err
    assert 'secret-value' not in captured.err


def test_checks_command_lists_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['checks']) == 0
    output = capsys.readouterr().out
    assert 'Built-in checks (14)' in output
    assert 'connectivity' in output
    assert 'requires a model' in output


def test_checks_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['checks', '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload['checks']) == 14
    ids = {check['id'] for check in payload['checks']}
    assert ids >= {'connectivity', 'tools', 'json', 'json_schema', 'reasoning'}
    assert any(not check['default_enabled'] for check in payload['checks'])


def test_providers_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['providers']) == 0
    output = capsys.readouterr().out
    assert 'openai' in output
    assert 'OPENAI_API_KEY' in output
    assert 'ollama' in output


def test_run_requires_authorization_without_yes(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_args(mock_server)) == 2
    captured = capsys.readouterr()
    assert 'explicit authorization' in captured.err
    assert mock_server.requests == []


def test_run_json_report(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_args(mock_server, '--yes', '--format', 'json')) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['summary']['verdict'] == 'compatible'
    assert payload['summary']['counts']['pass'] == 8
    assert payload['provider']['base_url'] == mock_server.base_url
    assert payload['provider']['model'] == 'mock-chat'


def test_run_writes_report_file(
    mock_server: MockAPIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / 'report.json'
    code = main(
        _run_args(
            mock_server,
            '--yes',
            '--format',
            'json',
            '--output',
            str(destination),
        )
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'Wrote output' in captured.err
    payload = json.loads(destination.read_text(encoding='utf-8'))
    assert payload['tool']['name'] == 'llm-compat'


def test_skipped_checks_make_the_cli_exit_nonzero(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_server.models = []
    code = main(
        [
            'run',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--samples',
            '1',
            '--only',
            'connectivity',
            '--only',
            'chat',
            '--yes',
        ]
    )
    assert code == 1
    assert 'SKIP' in capsys.readouterr().out


def test_dry_run_makes_no_requests(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_args(mock_server, '--dry-run')) == 0
    output = capsys.readouterr().out
    assert 'Dry run' in output
    assert 'no network requests' in output
    assert mock_server.requests == []


def test_dry_run_redacts_a_key_embedded_in_the_url_path(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = 'path-embedded-secret-value'
    monkeypatch.setenv('LLM_COMPAT_PATH_KEY', secret)
    code = main(
        [
            'run',
            '--provider',
            'custom',
            '--base-url',
            f'https://api.example.com/v1/{secret}',
            '--api-key-env',
            'LLM_COMPAT_PATH_KEY',
            '--model',
            'mock-chat',
            '--dry-run',
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert secret not in captured.out
    assert secret not in captured.err
    assert '[REDACTED]' in captured.out


def test_dry_run_json_reports_shared_models_request(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_args(mock_server, '--dry-run', '--format', 'json')) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['estimated_requests'] == 7
    assert mock_server.requests == []


def test_models_command_json(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            'models',
            '--provider',
            'custom',
            '--base-url',
            mock_server.base_url,
            '--yes',
            '--format',
            'json',
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['count'] == 2
    assert payload['models'][0]['id'] == 'mock-chat'


def test_missing_api_key_environment_variable_is_an_error(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('LLM_COMPAT_MISSING_KEY', raising=False)
    args = [
        'models',
        '--provider',
        'custom',
        '--base-url',
        mock_server.base_url,
        '--api-key-env',
        'LLM_COMPAT_MISSING_KEY',
        '--yes',
    ]
    assert main(args) == 2
    assert 'LLM_COMPAT_MISSING_KEY' in capsys.readouterr().err
    assert mock_server.requests == []


def test_api_key_is_used_but_never_printed(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = 'test-secret-value-123'
    mock_server.api_key = secret
    monkeypatch.setenv('LLM_COMPAT_TEST_KEY', secret)
    code = main(
        _run_args(
            mock_server,
            '--api-key-env',
            'LLM_COMPAT_TEST_KEY',
            '--yes',
            '--format',
            'json',
        )
    )
    captured = capsys.readouterr()
    assert code == 0
    assert secret not in captured.out
    assert secret not in captured.err
    request = mock_server.requests[0]
    assert request['headers']['authorization'] == f'Bearer {secret}'


def test_failure_results_in_nonzero_exit(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_server.support_tools = False
    code = main(_run_args(mock_server, '--only', 'tools', '--yes', '--format', 'json'))
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['results'][0]['status'] == 'fail'


def test_unknown_check_id_returns_configuration_error(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_run_args(mock_server, '--only', 'nope', '--yes')) == 2
    assert 'Unknown check id' in capsys.readouterr().err
    assert mock_server.requests == []


def test_custom_provider_requires_base_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['models', '--provider', 'custom', '--yes']) == 2
    assert '--base-url is required' in capsys.readouterr().err


def test_interactive_authorization_accepts_yes(
    mock_server: MockAPIServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr('builtins.input', lambda prompt='': 'yes')
    code = main(_run_args(mock_server, '--only', 'connectivity'))
    captured = capsys.readouterr()
    assert code == 0
    assert 'live API requests' in captured.err
    assert len(mock_server.requests) == 1


def test_interactive_authorization_rejects_no(
    mock_server: MockAPIServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr('builtins.input', lambda prompt='': 'no')
    assert main(_run_args(mock_server, '--only', 'connectivity')) == 2
    assert 'authorization was not granted' in capsys.readouterr().err
    assert mock_server.requests == []


@pytest.mark.parametrize('answer', ['y', 'YES', ' yes '])
def test_interactive_authorization_requires_exact_yes(
    mock_server: MockAPIServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
) -> None:
    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, 'stdin', TTY())
    monkeypatch.setattr('builtins.input', lambda prompt='': answer)
    assert main(_run_args(mock_server, '--only', 'connectivity')) == 2
    assert 'authorization was not granted' in capsys.readouterr().err
    assert mock_server.requests == []


def test_version_exits_successfully() -> None:
    with pytest.raises(SystemExit) as captured:
        main(['--version'])
    assert captured.value.code == 0


def test_models_can_be_printed_as_text(
    mock_server: MockAPIServer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        'models',
        '--provider',
        'custom',
        '--base-url',
        mock_server.base_url,
        '--yes',
    ]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert 'Models: 2' in output
    assert 'mock-chat' in output


def test_providers_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['providers', '--format', 'json']) == 0
    payload = json.loads(capsys.readouterr().out)
    keys = {preset['key'] for preset in payload['providers']}
    assert {'openai', 'ollama', 'deepseek'} <= keys


def test_runner_extra_import_guard() -> None:
    # Keep a direct import of public API symbols exercised by the test suite.
    import llm_compat

    assert llm_compat.__version__ == '0.2.0'
    assert callable(llm_compat.run_compatibility)
