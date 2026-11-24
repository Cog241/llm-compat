'''Security regression tests for credential handling and report redaction.'''

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_compat.checks._tooling import call_preview, parse_arguments
from llm_compat.cli import main
from llm_compat.config import ProviderConfig
from llm_compat.errors import ConfigurationError, NetworkError, build_http_error
from llm_compat.http import HttpResponse
from llm_compat.providers.base import ToolCall
from llm_compat.providers.registry import build_provider
from llm_compat.report import CompatibilityReport, render_text
from llm_compat.results import (
    CheckResult,
    ErrorInfo,
    FailureKind,
    Status,
    summarize,
)
from llm_compat.runner import RunOptions, run_compatibility
from llm_compat.utils import redact_sensitive, redact_value
from tests.mock_server import MockAPIServer


def test_remote_authenticated_http_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match='HTTPS'):
        ProviderConfig(
            base_url='http://api.example.com/v1',
            api_key='remote-secret',
        )


def test_remote_authenticated_http_override_is_explicit() -> None:
    config = ProviderConfig(
        base_url='http://api.example.com/v1',
        api_key='remote-secret',
        allow_insecure_http=True,
    )
    assert config.allow_insecure_http is True
    assert config.sensitive_values == ('remote-secret',)


def test_sensitive_header_also_requires_https() -> None:
    with pytest.raises(ConfigurationError, match='HTTPS'):
        ProviderConfig(
            base_url='http://api.example.com/v1',
            extra_headers={'Authorization': 'Bearer header-secret'},
        )


def test_non_sensitive_header_over_remote_http_is_allowed() -> None:
    config = ProviderConfig(
        base_url='http://api.example.com/v1',
        extra_headers={'X-Client': 'llm-compat'},
    )
    assert config.auth_configured is False


def test_sensitive_header_counts_as_configured_auth() -> None:
    config = ProviderConfig(
        base_url='https://api.example.com/v1',
        extra_headers={'Authorization': 'Bearer header-secret'},
    )
    assert config.auth_configured is True
    assert config.sensitive_values == ('Bearer header-secret', 'header-secret')


def test_authorization_header_exposes_scheme_credential_for_redaction() -> None:
    config = ProviderConfig(
        base_url='https://api.example.com/v1',
        extra_headers={'Authorization': 'Bearer header-secret'},
    )

    redacted = redact_sensitive('provider echoed header-secret', config.sensitive_values)

    assert 'header-secret' not in redacted


def test_loopback_http_with_key_is_allowed() -> None:
    config = ProviderConfig(
        base_url='http://127.0.0.1:8080/v1',
        api_key='local-secret',
    )
    assert config.auth_configured is True


def test_localhost_http_with_key_is_allowed() -> None:
    config = ProviderConfig(
        base_url='http://localhost:8080/v1',
        api_key='local-secret',
    )
    assert config.auth_configured is True


def test_plain_remote_http_without_credentials_is_allowed() -> None:
    config = ProviderConfig(base_url='http://api.example.com/v1')
    assert config.auth_configured is False


def test_redact_sensitive_covers_known_and_pattern_credentials() -> None:
    text = (
        'known=known-secret Bearer abcdefghijklmnop sk-abcdefghijklmnop '
        'coa_abcdefghijklmnop gsk_abcdefghijklmnop api_key=another-secret'
    )
    redacted = redact_sensitive(text, ('known-secret',))
    assert 'known-secret' not in redacted
    assert 'abcdefghijklmnop' not in redacted
    assert 'another-secret' not in redacted
    assert '[REDACTED]' in redacted


def test_redact_sensitive_prefers_the_longest_known_secret() -> None:
    assert redact_sensitive('abcdef', ('abc', 'abcdef')) == '[REDACTED]'


def test_redact_sensitive_covers_quoted_json_credentials() -> None:
    text = (
        '{"api_key": "plain-json-value", "password": "two words", '
        '"normal": "keep-me"}'
    )
    redacted = redact_sensitive(text)
    assert 'plain-json-value' not in redacted
    assert 'two words' not in redacted
    assert 'keep-me' in redacted


def test_tool_argument_previews_redact_before_truncating() -> None:
    secret = 'tool-secret-value'
    arguments = json.dumps(
        {'api_key': secret, 'padding': 'x' * 400}
    )
    preview = call_preview(
        ToolCall(id='call-1', name='get_weather', arguments=arguments),
        limit=40,
        secrets=(secret,),
    )
    assert secret not in preview['arguments']
    assert '[REDACTED]' in preview['arguments']


def test_redact_value_is_recursive() -> None:
    secret = 'sk-abcdefghijklmnop'
    value = {'outer': [secret, {'inner': f'prefix {secret}'}]}
    redacted = redact_value(value, (secret,))
    assert secret not in json.dumps(redacted)


def test_report_redacts_every_string_surface() -> None:
    secret = 'report-secret-value'
    result = CheckResult(
        id='chat',
        name='Chat',
        category='chat',
        status=Status.FAIL,
        message=f'The response contained {secret}',
        details={'nested': {'value': secret}, 'list': [secret]},
        error=ErrorInfo(
            type='HTTPStatusError',
            kind='http',
            message=f'Provider said {secret}',
            hint=f'Rotate {secret}',
            failure_kind=FailureKind.AUTHENTICATION,
        ),
        failure_kind=FailureKind.AUTHENTICATION,
    )
    report = CompatibilityReport(
        provider_name=f'provider {secret}',
        base_url='https://example.com/v1',
        model=f'model {secret}',
        model_source='explicit',
        auth_configured=True,
        started_at='2026-01-01T00:00:00.000Z',
        finished_at='2026-01-01T00:00:01.000Z',
        duration_ms=1000.0,
        results=(result,),
        summary=summarize((result,)),
        secrets=(secret,),
    )
    payload = report.to_dict()
    serialized = json.dumps(payload)
    text = render_text(report)
    assert secret not in serialized
    assert secret not in text
    assert '[REDACTED]' in serialized
    assert payload['summary']['failure_kinds'] == {'authentication': 1}
    assert 'Failure class: authentication' in text


def test_provider_error_body_does_not_leak_key(
    mock_server: MockAPIServer,
    config_factory: Any,
) -> None:
    secret = 'sk-abcdefghijklmnop'
    mock_server.fail_status = 401
    mock_server.fail_message = f'Rejected {secret}'
    config = config_factory(api_key=secret)
    report = run_compatibility(
        config,
        RunOptions(authorize=True, samples=1, only=('chat',)),
        provider=build_provider(config),
    )
    serialized = json.dumps(report.to_dict())
    assert secret not in serialized
    assert secret not in render_text(report)
    assert '[REDACTED]' in serialized


def test_http_error_redacts_body_and_message() -> None:
    secret = 'sk-abcdefghijklmnop'
    error = build_http_error(
        401,
        'https://example.com/v1/models?key=hidden',
        json.dumps({'error': {'message': f'bad key {secret}'}}),
        secrets=(secret,),
    )
    assert secret not in error.message
    assert secret not in error.body


def test_cli_does_not_echo_remote_http_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = 'sk-abcdefghijklmnop'
    monkeypatch.setenv('REMOTE_HTTP_KEY', secret)
    code = main(
        [
            'run',
            '--provider',
            'custom',
            '--base-url',
            'http://api.example.com/v1',
            '--model',
            'mock-chat',
            '--api-key-env',
            'REMOTE_HTTP_KEY',
            '--dry-run',
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert secret not in captured.err
    assert secret not in captured.out
    assert 'HTTPS' in captured.err


def test_response_read_error_redacts_known_secret() -> None:
    secret = 'known-response-secret'

    class FailingRaw:
        def read(self) -> bytes:
            raise OSError(f'connection lost while reading {secret}')

        def close(self) -> None:
            pass

    response = HttpResponse(
        200,
        {},
        raw=FailingRaw(),
        url='https://example.com/v1/chat/completions',
        secrets=(secret,),
    )

    with pytest.raises(NetworkError) as captured:
        _ = response.text

    assert secret not in str(captured.value)
    assert '[REDACTED]' in str(captured.value)


def test_tool_arguments_reject_nonstandard_json_constants() -> None:
    arguments, error = parse_arguments(
        ToolCall(
            id='call_1',
            name='get_weather',
            arguments='{"city":"Paris","value":NaN}',
        )
    )

    assert arguments is None
    assert error == 'The tool call arguments were not valid JSON.'