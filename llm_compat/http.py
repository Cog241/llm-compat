'''Minimal HTTP transport built on the Python standard library.'''

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import IO, Any, Iterator, Mapping, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from llm_compat.errors import NetworkError, ProtocolError
from llm_compat.utils import redact_secret, safe_url


@dataclass(frozen=True)
class HttpRequest:
    '''A transport-independent HTTP request.'''

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    timeout: float = 30.0


class HttpResponse:
    '''An HTTP response that can be read as text, JSON, or line by line.'''

    def __init__(
        self,
        status_code: int,
        headers: Mapping[str, str],
        *,
        body: Optional[bytes] = None,
        raw: Optional[IO[bytes]] = None,
        url: str = '',
    ) -> None:
        self.status_code = status_code
        self.headers = {str(key).lower(): str(value) for key, value in headers.items()}
        self.url = url
        self._body = body
        self._raw = raw

    @property
    def text(self) -> str:
        if self._body is None:
            if self._raw is None:
                return ''
            try:
                self._body = self._raw.read()
            except (OSError, ValueError, HTTPException) as exc:
                raise NetworkError(
                    f'Could not read the response from {safe_url(self.url)}: {exc}'
                ) from exc
            finally:
                self.close()
        return self._body.decode('utf-8', errors='replace')

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProtocolError(
                f'Expected JSON from {safe_url(self.url) or "the endpoint"} '
                'but received non-JSON content.'
            ) from exc

    def iter_lines(self) -> Iterator[str]:
        if self._body is not None:
            for line in self._body.splitlines():
                yield line.decode('utf-8', errors='replace')
            return
        if self._raw is None:
            return
        try:
            for line in self._raw:
                yield line.decode('utf-8', errors='replace').rstrip('\r\n')
        except (OSError, ValueError, HTTPException) as exc:
            raise NetworkError(
                f'Could not read the stream from {safe_url(self.url)}: {exc}'
            ) from exc

    def close(self) -> None:
        if self._raw is not None:
            try:
                self._raw.close()
            finally:
                self._raw = None

    def __enter__(self) -> 'HttpResponse':
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


class HttpClient(Protocol):
    '''Pluggable transport protocol used by provider adapters and tests.'''

    def send(self, request: HttpRequest) -> HttpResponse:
        '''Send one request and return its response.'''
        ...


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    '''Follow redirects only when scheme, host, and port are unchanged.'''

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urlsplit(req.full_url)
        new = urlsplit(newurl)
        try:
            old_origin = (old.scheme.lower(), (old.hostname or '').lower(), old.port)
            new_origin = (new.scheme.lower(), (new.hostname or '').lower(), new.port)
        except ValueError:
            return None
        if old_origin != new_origin:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_SameOriginRedirectHandler())


class UrllibHttpClient:
    '''Default transport. It uses TLS verification by default.'''

    def send(self, request: HttpRequest) -> HttpResponse:
        secret = None
        for name, value in request.headers.items():
            if (
                isinstance(name, str)
                and name.lower() == 'authorization'
                and isinstance(value, str)
                and value.lower().startswith('bearer ')
            ):
                secret = value[7:]
                break
        try:
            http_request = Request(
                request.url,
                data=request.body,
                headers=dict(request.headers),
                method=request.method,
            )
            raw = _OPENER.open(http_request, timeout=request.timeout)
        except HTTPError as exc:
            body = b''
            try:
                body = exc.read()
            except (OSError, ValueError):
                pass
            finally:
                exc.close()
            return HttpResponse(
                exc.code,
                dict(exc.headers.items()),
                body=body,
                url=request.url,
            )
        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            reason = redact_secret(str(getattr(exc, 'reason', exc)), secret)
            raise NetworkError(
                f'Could not reach {safe_url(request.url)}: {reason}.'
            ) from exc
        except (ValueError, TypeError, HTTPException) as exc:
            detail = redact_secret(str(exc), secret)
            raise NetworkError(
                f'Could not send the request to {safe_url(request.url)}: {detail}'
            ) from exc
        return HttpResponse(
            raw.getcode(),
            dict(raw.headers.items()),
            raw=raw,
            url=request.url,
        )
