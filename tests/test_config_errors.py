'''Tests for configuration validation, safe URLs, and typed errors.'''

from __future__ import annotations

import pytest

from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.errors import (
    AuthorizationRequiredError,
    ConfigurationError,
    HTTPStatusError,
    build_http_error,
)
from llm_compat.utils import redact_secret, safe_url, truncate


def test_config_normalizes_values_and_hides_key() -> None:
    config = ProviderConfig(
        base_url=' https://example.com/v1/ ',
        name='custom',
        api_key='super-secret-key',
        model=' model-name ',
        timeout=12,
    )
    assert config.base_url == 'https://example.com/v1'
    assert config.model == 'model-name'
    assert config.timeout == 12
    assert config.auth_configured is True
    assert 'super-secret-key' not in repr(config)


@pytest.mark.parametrize(
    'base_url',
    [
        '',
        'api.example.com/v1',
        'ftp://example.com/v1',
        'https://',
        'https://user:password@example.com/v1',
        'https://example.com/v1?key=value',
        'https://example.com/v1#fragment',
        'https://example.com:not-a-port/v1',
        'https://exa mple.com/v1',
        'https://example.com/v1\x00',
        'https://example.com/v1\r\n',
    ],
)
def test_config_rejects_invalid_base_urls(base_url: str) -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(base_url=base_url)


def test_config_rejects_invalid_timeout() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(base_url='https://example.com/v1', timeout=0)


def test_config_rejects_header_injection() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test': 'safe\r\nInjected: true'},
        )


def test_config_rejects_control_characters_in_headers_and_keys() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test\x00': 'value'},
        )
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test\n': 'value'},
        )
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test': 'value\x00'},
        )
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test': 'value\n'},
        )
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            api_key='safe\x00Injected',
        )


def test_config_rejects_non_string_headers() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            extra_headers={'X-Test': 123},  # type: ignore[dict-item]
        )


def test_config_rejects_line_breaks_in_api_key() -> None:
    with pytest.raises(ConfigurationError):
        ProviderConfig(
            base_url='https://example.com/v1',
            api_key='safe\nInjected: true',
        )
    with pytest.raises(ConfigurationError):
        ProviderConfig(base_url='https://example.com/v1', api_key='safe\n')


@pytest.mark.parametrize(
    ('samples', 'max_tokens'),
    [(0, 128), (11, 128), (3, 0), (3, 4097)],
)
def test_run_settings_bounds(samples: int, max_tokens: int) -> None:
    with pytest.raises(ConfigurationError):
        RunSettings(samples=samples, max_tokens=max_tokens)



def test_redirect_errors_explain_the_same_origin_policy() -> None:
    error = build_http_error(302, 'https://example.com/v1/models', '')
    assert 'redirect' in error.message
    assert 'scheme, host, and port' in (error.hint or '')


def test_safe_url_strips_credentials_and_query() -> None:
    url = 'https://user:password@example.com:8443/v1?api_key=leaked#fragment'
    assert safe_url(url) == 'https://example.com:8443/v1'


def test_safe_url_never_returns_malformed_or_relative_urls() -> None:
    assert safe_url('https://user:password@example.com:bad/v1') == '<invalid URL>'
    assert safe_url('not a url') == '<invalid URL>'
    assert safe_url(None) == '<invalid URL>'  # type: ignore[arg-type]


def test_safe_url_removes_terminal_controls() -> None:
    rendered = safe_url('https://example.com/\x1b[31mred')
    assert '\x1b' not in rendered


def test_redact_and_truncate_helpers() -> None:
    assert redact_secret('key is abc123', 'abc123') == 'key is [REDACTED]'
    assert truncate('a' * 20, 10) == 'aaaaaaa...'
    assert truncate('safe\x1b[31mred\x00') == 'safe [31mred'


def test_build_http_error_extracts_message_and_redacts() -> None:
    error = build_http_error(
        401,
        'https://example.com/v1/models?key=hidden',
        '{"error": {"message": "bad key secret-value"}}',
        secret='secret-value',
        retry_after='120',
    )
    assert isinstance(error, HTTPStatusError)
    assert error.status_code == 401
    assert 'bad key [REDACTED]' in error.message
    assert 'secret-value' not in error.message
    assert 'secret-value' not in error.body
    assert '?key=hidden' not in error.message
    assert error.retry_after == 120
    assert error.retryable is False
    assert error.hint is not None


@pytest.mark.parametrize(
    ('status', 'retryable'),
    [(429, True), (500, True), (503, True), (400, False), (404, False)],
)
def test_http_error_retryability(status: int, retryable: bool) -> None:
    error = build_http_error(status, 'https://example.com/v1', '')
    assert error.retryable is retryable


def test_authorization_error_is_actionable() -> None:
    error = AuthorizationRequiredError()
    assert 'explicit authorization' in str(error)
    assert error.hint is not None
