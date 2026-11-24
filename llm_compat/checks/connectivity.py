'''Reachability and authentication checks that use the models endpoint.'''

from __future__ import annotations

import time

from llm_compat.checks.base import CheckContext, CompatibilityCheck, error_info
from llm_compat.errors import HTTPStatusError, ProtocolError
from llm_compat.results import CheckResult, FailureKind, Status


class ConnectivityCheck(CompatibilityCheck):
    id = 'connectivity'
    name = 'API connectivity'
    category = 'connectivity'
    description = 'Verify that the base URL is reachable and answers HTTP requests.'

    def run(self, context: CheckContext) -> CheckResult:
        started = time.perf_counter()
        try:
            models = context.provider.list_models()
            elapsed = (time.perf_counter() - started) * 1000.0
            return self.result(
                Status.PASS,
                f'Endpoint reachable; /models returned {len(models)} model(s).',
                details={
                    'status_code': 200,
                    'latency_ms': round(elapsed, 2),
                    'model_count': len(models),
                },
            )
        except HTTPStatusError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            return self.result(
                Status.PASS,
                f'Endpoint reachable; /models returned HTTP {exc.status_code}.',
                details={
                    'status_code': exc.status_code,
                    'latency_ms': round(elapsed, 2),
                },
            )
        except ProtocolError as exc:
            return self.result(
                Status.WARN,
                context.redact(
                    f'Endpoint reachable, but /models did not return valid OpenAI JSON: {exc}',
                    500,
                ),
                details={'status_code': 200},
                failure_kind=FailureKind.MALFORMED,
            )


class AuthenticationCheck(CompatibilityCheck):
    id = 'auth'
    name = 'Authentication'
    category = 'connectivity'
    description = 'Verify that the configured API key is accepted by the endpoint.'

    def run(self, context: CheckContext) -> CheckResult:
        try:
            context.provider.list_models()
        except HTTPStatusError as exc:
            if exc.status_code in {401, 403}:
                detail = context.redact(exc.message, 160)
                suffix = f': {detail}' if detail else ''
                return self.result(
                    Status.FAIL,
                    f'Authentication failed with HTTP {exc.status_code}{suffix}',
                    details={'status_code': exc.status_code},
                    error=error_info(
                        exc,
                        failure_kind=FailureKind.AUTHENTICATION,
                        secrets=context.provider.config.sensitive_values,
                    ),
                    failure_kind=FailureKind.AUTHENTICATION,
                )
            if exc.status_code == 404:
                return self.result(
                    Status.WARN,
                    'Authentication could not be verified because /models returned HTTP 404.',
                    details={'status_code': 404},
                    error=error_info(
                        exc,
                        failure_kind=FailureKind.UNSUPPORTED,
                        secrets=context.provider.config.sensitive_values,
                    ),
                    failure_kind=FailureKind.UNSUPPORTED,
                )
            raise
        except ProtocolError as exc:
            return self.result(
                Status.WARN,
                context.redact(
                    f'The request was accepted, but /models could not be parsed: {exc}',
                    500,
                ),
                details={'status_code': 200},
                failure_kind=FailureKind.MALFORMED,
            )
        if context.provider.auth_configured:
            message = 'API key accepted by the endpoint.'
        else:
            message = 'Endpoint accepted requests without an API key.'
        return self.result(Status.PASS, message, details={'status_code': 200})
