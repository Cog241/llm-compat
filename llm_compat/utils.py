'''Small shared helpers with no project-specific dependencies.'''

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


def truncate(value: str, limit: int = 400) -> str:
    '''Collapse whitespace, remove control characters, and truncate for reports.'''
    limit = max(3, limit)
    printable = ''.join(char if char.isprintable() else ' ' for char in value)
    cleaned = ' '.join(printable.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + '...'


def redact_secret(value: str, secret: str | None) -> str:
    '''Replace an API key if it appears in text that will be displayed.'''
    if secret and secret in value:
        return value.replace(secret, '[REDACTED]')
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
