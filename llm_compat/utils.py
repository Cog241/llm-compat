'''Small shared helpers with no project-specific dependencies.'''

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

_BEARER_RE = re.compile(r'(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{4,}')
_BASIC_RE = re.compile(r'(?i)\b(basic)\s+[A-Za-z0-9+/=]{4,}')
_KEY_VALUE_RE = re.compile(
    r'(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|token|client[_-]?secret|secret|password)'
    r'\b["\']?\s*[:=]\s*)'
    r'(?:"[^"\r\n]*"|\'[^\'\r\n]*\'|[^"\'\s,;&]{3,})'
)
_PREFIXED_KEY_RE = re.compile(
    r'\b(?:sk-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9_-]{8,}|coa_[A-Za-z0-9_-]{8,}'
    r'|xai-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_-]{8,}|github_pat_[A-Za-z0-9_-]{8,}'
    r'|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[0-9A-Za-z_-]{20,})'
)


def truncate(value: str, limit: int = 400) -> str:
    '''Collapse whitespace, remove control characters, and truncate for reports.'''
    limit = max(3, limit)
    printable = ''.join(char if char.isprintable() else ' ' for char in value)
    cleaned = ' '.join(printable.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + '...'


def redact_sensitive(value: str, secrets: Iterable[str] = ()) -> str:
    '''Remove known credentials and common credential patterns from text.

    Reports are also sanitized centrally, but errors and short response
    previews pass through this helper so an unsanitized value cannot leak
    into a log or exception first.
    '''
    if not isinstance(value, str):
        return value
    redacted = value
    for secret in sorted(
        {item for item in secrets if isinstance(item, str) and item},
        key=len,
        reverse=True,
    ):
        if secret in redacted:
            redacted = redacted.replace(secret, '[REDACTED]')
    redacted = _BEARER_RE.sub(lambda match: f'{match.group(1)} [REDACTED]', redacted)
    redacted = _BASIC_RE.sub(lambda match: f'{match.group(1)} [REDACTED]', redacted)
    redacted = _PREFIXED_KEY_RE.sub('[REDACTED]', redacted)
    redacted = _KEY_VALUE_RE.sub(lambda match: f'{match.group(1)}[REDACTED]', redacted)
    return redacted


def redact_secret(value: str, secret: str | None) -> str:
    '''Replace an API key if it appears in text that will be displayed.'''
    return redact_sensitive(value, (secret,) if secret else ())


def redact_value(value: Any, secrets: Iterable[str] = ()) -> Any:
    '''Recursively redact strings in a JSON-like report value.'''
    if isinstance(value, str):
        return redact_sensitive(value, secrets)
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets) for item in value)
    if isinstance(value, dict):
        return {
            redact_sensitive(key, secrets) if isinstance(key, str) else key: redact_value(
                item, secrets
            )
            for key, item in value.items()
        }
    return value


def safe_url(url: str) -> str:
    '''Return a URL suitable for logs, without credentials or query strings.'''
    if not isinstance(url, str):
        return '<invalid URL>'
    try:
        parts = urlsplit(url)
        host = parts.hostname or ''
        if not host:
            return '<invalid URL>'
        if ':' in host and not host.startswith('['):
            host = f'[{host}]'
        netloc = host
        port = parts.port
        if port is not None:
            netloc = f'{netloc}:{port}'
        rendered = urlunsplit((parts.scheme, netloc, parts.path, '', ''))
        return ''.join(char if char.isprintable() else ' ' for char in rendered)
    except (TypeError, ValueError):
        return '<invalid URL>'


def utc_now() -> str:
    '''Return an ISO-8601 UTC timestamp with millisecond precision.'''
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')