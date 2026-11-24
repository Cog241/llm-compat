'''Typed errors that produce actionable compatibility reports.'''

from __future__ import annotations

import json
from typing import Any, Optional

from llm_compat.utils import redact_secret, safe_url, truncate

_STATUS_HINTS = {
    400: (
        'The provider rejected the request. Its message usually identifies '
        'the unsupported parameter.'
    ),
    401: 'Check that the API key is valid and that the configured environment variable is set.',
    403: 'The key is valid but does not have permission for this endpoint or model.',
    404: (
        'Verify the base URL includes the API version (for example /v1) '
        'and that the endpoint exists.'
    ),
    405: 'The endpoint does not allow this HTTP method.',
    429: 'The provider rate-limited the request. Retry later or reduce request volume.',
    500: 'The provider reported a server error. Retry later.',
    502: 'The provider or an upstream service returned a bad gateway response. Retry later.',
    503: 'The provider is temporarily unavailable. Retry later.',
}


class LLMCompatError(Exception):
    '''Base class for expected failures that should become report entries.'''

    kind = 'error'
    retryable = False
    status_code: Optional[int] = None
    hint: Optional[str] = None

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint

    @property
    def message(self) -> str:
        return str(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': type(self).__name__,
            'kind': self.kind,
            'message': self.message,
            'status_code': self.status_code,
            'retryable': self.retryable,
            'hint': self.hint,
        }


class ConfigurationError(LLMCompatError):
    '''The requested run cannot be configured as specified.'''

    kind = 'configuration'


class AuthorizationRequiredError(LLMCompatError):
    '''Live requests were attempted without explicit authorization.'''

    kind = 'authorization'

    def __init__(self) -> None:
        super().__init__(
            'Live API requests require explicit authorization because they may incur costs.',
            hint=(
                'Pass RunOptions(authorize=True) to run_compatibility() or use --yes '
                'with the CLI.'
            ),
        )


class NetworkError(LLMCompatError):
    '''A request could not reach or finish talking to the endpoint.'''

    kind = 'network'
    retryable = True

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(
            message,
            hint=hint
            or (
                'Check the base URL, DNS, TLS, proxy settings, and that the endpoint is reachable.'
            ),
        )


class ProtocolError(LLMCompatError):
    '''The endpoint responded, but not with a valid OpenAI-compatible payload.'''

    kind = 'protocol'

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(
            message,
            hint=hint
            or (
                'Verify that the endpoint implements the OpenAI-compatible REST and SSE format.'
            ),
        )


class HTTPStatusError(LLMCompatError):
    '''The endpoint returned a non-success HTTP status.'''

    kind = 'http'

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        body: str = '',
        retry_after: Optional[int] = None,
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


def _extract_error_message(body: str) -> Optional[str]:
    '''Pull a human-readable message out of common provider error payloads.'''
    if not body.strip():
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get('error')
    if isinstance(error, dict):
        message = error.get('message') or error.get('code')
        return truncate(str(message)) if message else None
    if isinstance(error, str):
        return truncate(error)
    message = payload.get('message')
    return truncate(str(message)) if message else None


def build_http_error(
    status_code: int,
    url: str,
    body: str,
    *,
    secret: Optional[str] = None,
    retry_after: Optional[str] = None,
) -> HTTPStatusError:
    '''Create a redacted and user-friendly error for an HTTP failure.'''
    detail = _extract_error_message(body)
    if detail is None:
        if status_code == 401:
            detail = 'Authentication failed.'
        elif status_code == 403:
            detail = 'The request was forbidden.'
        elif status_code == 404:
            detail = 'The endpoint was not found.'
        elif status_code == 429:
            detail = 'The request was rate-limited.'
        elif 300 <= status_code < 400:
            detail = 'The endpoint returned a redirect that was not followed.'
        elif status_code >= 500:
            detail = 'The provider returned a server error.'
        else:
            detail = 'The provider returned an error response.'
    message = redact_secret(f'HTTP {status_code} from {safe_url(url)}: {detail}', secret)
    parsed_retry_after: Optional[int] = None
    if retry_after:
        try:
            parsed_retry_after = int(retry_after)
        except ValueError:
            parsed_retry_after = None
    hint = _STATUS_HINTS.get(status_code)
    if 300 <= status_code < 400:
        hint = 'Redirects are followed only when the scheme, host, and port stay the same.'
    return HTTPStatusError(
        status_code,
        message,
        body=redact_secret(truncate(body, 2000), secret),
        retry_after=parsed_retry_after,
        hint=hint,
    )
