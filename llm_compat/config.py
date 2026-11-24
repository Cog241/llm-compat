'''Validated configuration objects for a compatibility run.'''

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional
from urllib.parse import urlsplit

from llm_compat.errors import ConfigurationError

_TOKEN_SEPARATORS = frozenset('()<>@,;:\\"/[]?={} \t')


def _has_control_characters(value: str) -> bool:
    return any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in value)


def _is_header_name(value: str) -> bool:
    return bool(value) and value.isascii() and all(
        33 <= ord(char) <= 126 and char not in _TOKEN_SEPARATORS for char in value
    )


@dataclass(frozen=True)
class ProviderConfig:
    '''Connection settings for one OpenAI-compatible endpoint.'''

    base_url: str
    name: str = 'custom'
    api_key: Optional[str] = field(default=None, repr=False)
    model: Optional[str] = None
    timeout: float = 30.0
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str):
            raise ConfigurationError('base_url must be a string.')
        if _has_control_characters(self.base_url):
            raise ConfigurationError('base_url must not contain control characters.')
        base_url = self.base_url.strip().rstrip('/')
        if any(char.isspace() for char in base_url):
            raise ConfigurationError('base_url must not contain internal whitespace.')

        parts = urlsplit(base_url)
        if parts.scheme not in {'http', 'https'} or not parts.netloc or not parts.hostname:
            raise ConfigurationError(
                'base_url must be an absolute HTTP(S) URL, for example https://api.example.com/v1'
            )
        if parts.username is not None or parts.password is not None:
            raise ConfigurationError(
                'base_url must not contain credentials; use --api-key-env or api_key instead.'
            )
        if '?' in base_url or '#' in base_url:
            raise ConfigurationError('base_url must not contain a query string or fragment.')
        try:
            _ = parts.port
        except ValueError as exc:
            raise ConfigurationError('base_url contains an invalid port.') from exc
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise ConfigurationError('timeout must be a positive number of seconds.')
        timeout = float(self.timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError('timeout must be a finite number greater than zero.')
        if not isinstance(self.name, str):
            raise ConfigurationError('name must be a string.')
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ConfigurationError('api_key must be a string or None.')
        if self.model is not None and not isinstance(self.model, str):
            raise ConfigurationError('model must be a string or None.')
        name = self.name.strip() or 'custom'
        raw_api_key = self.api_key
        api_key = raw_api_key.strip() if isinstance(raw_api_key, str) else raw_api_key
        model = self.model.strip() if isinstance(self.model, str) else self.model
        if isinstance(raw_api_key, str) and _has_control_characters(raw_api_key):
            raise ConfigurationError('api_key must not contain control characters.')
        if not isinstance(self.extra_headers, MappingABC):
            raise ConfigurationError('extra_headers must be a mapping of strings.')
        headers = {}
        for key, value in self.extra_headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ConfigurationError('extra_headers keys and values must be strings.')
            header_name = key.strip()
            header_value = value.strip()
            if _has_control_characters(key):
                raise ConfigurationError(f'Invalid header name: {key!r}')
            if not _is_header_name(header_name):
                raise ConfigurationError(f'Invalid header name: {key!r}')
            if _has_control_characters(value):
                raise ConfigurationError(f'Invalid header value for {header_name}.')
            headers[header_name] = header_value
        object.__setattr__(self, 'base_url', base_url)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'api_key', api_key or None)
        object.__setattr__(self, 'model', model or None)
        object.__setattr__(self, 'timeout', timeout)
        object.__setattr__(self, 'extra_headers', MappingProxyType(headers))

    @property
    def auth_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class RunSettings:
    '''Controls optional request volume and token limits.'''

    samples: int = 3
    max_tokens: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise ConfigurationError('samples must be an integer.')
        if not 1 <= self.samples <= 10:
            raise ConfigurationError('samples must be between 1 and 10.')
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ConfigurationError('max_tokens must be an integer.')
        if not 1 <= self.max_tokens <= 4096:
            raise ConfigurationError('max_tokens must be between 1 and 4096.')
