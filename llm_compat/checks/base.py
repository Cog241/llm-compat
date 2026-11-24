'''Base class that turns check failures into consistent report entries.'''

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional

from llm_compat.config import RunSettings
from llm_compat.errors import (
    ConfigurationError,
    HTTPStatusError,
    LLMCompatError,
    NetworkError,
    ProtocolError,
)
from llm_compat.providers.base import Provider
from llm_compat.results import CheckResult, ErrorInfo, FailureKind, Status
from llm_compat.utils import redact_sensitive, truncate


@dataclass(frozen=True)
class CheckContext:
    '''Everything a check needs for one run.'''

    provider: Provider
    model: Optional[str]
    settings: RunSettings

    def redact(self, value: Any, limit: int = 400) -> str:
        '''Sanitize a response preview before it reaches a report.'''
        return truncate(redact_sensitive(str(value), self.provider.config.sensitive_values), limit)


def failure_kind_for_exception(exc: BaseException) -> FailureKind:
    '''Explain why an exception prevented a clean compatibility check.'''
    if isinstance(exc, HTTPStatusError):
        if exc.status_code in {401, 403}:
            return FailureKind.AUTHENTICATION
        if exc.status_code in {408, 425, 429} or exc.status_code >= 500:
            return FailureKind.TEMPORARY
        if exc.status_code in {400, 404, 405, 406, 409, 415, 422}:
            return FailureKind.UNSUPPORTED
        return FailureKind.UNKNOWN
    if isinstance(exc, ProtocolError):
        return FailureKind.MALFORMED
    if isinstance(exc, NetworkError):
        return FailureKind.TEMPORARY
    if isinstance(exc, ConfigurationError):
        return FailureKind.CONFIGURATION
    return FailureKind.UNKNOWN


def error_info(
    exc: BaseException,
    *,
    failure_kind: Optional[FailureKind] = None,
    secrets: tuple[str, ...] = (),
) -> ErrorInfo:
    '''Build a stable, JSON-safe, redacted error description.'''
    hint = getattr(exc, 'hint', None)
    if not isinstance(exc, LLMCompatError):
        hint = 'This is likely an LLM Compat defect. Re-run and report it if the problem persists.'
    hint_text = str(hint) if hint else None
    return ErrorInfo(
        type=type(exc).__name__,
        kind=str(getattr(exc, 'kind', 'internal')),
        message=truncate(redact_sensitive(str(exc), secrets), 1000) or type(exc).__name__,
        status_code=getattr(exc, 'status_code', None),
        retryable=bool(getattr(exc, 'retryable', False)),
        hint=(
            truncate(redact_sensitive(hint_text, secrets), 500) if hint_text else None
        ),
        failure_kind=failure_kind,
    )


def status_for_exception(exc: BaseException) -> Status:
    '''Classify an error as a compatibility failure or a run error.'''
    if isinstance(exc, HTTPStatusError):
        if exc.status_code in {400, 401, 403, 404, 405, 406, 409, 415, 422}:
            return Status.FAIL
        return Status.ERROR
    if isinstance(exc, ProtocolError):
        return Status.FAIL
    if isinstance(exc, (ConfigurationError, NetworkError)):
        return Status.ERROR
    return Status.ERROR


class CompatibilityCheck(ABC):
    '''One independently runnable compatibility test.'''

    id: ClassVar[str]
    name: ClassVar[str]
    category: ClassVar[str]
    description: ClassVar[str]
    requires_model: ClassVar[bool] = False
    default_enabled: ClassVar[bool] = True

    @abstractmethod
    def run(self, context: CheckContext) -> CheckResult:
        '''Execute the check and return a result.'''

    def result(
        self,
        status: Status,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        error: Optional[ErrorInfo] = None,
        failure_kind: Optional[FailureKind] = None,
    ) -> CheckResult:
        return CheckResult(
            id=self.id,
            name=self.name,
            category=self.category,
            status=status,
            message=message,
            details=details or {},
            error=error,
            failure_kind=failure_kind,
        )

    def execute(self, context: CheckContext) -> CheckResult:
        '''Run the check with timing and uniform exception handling.'''
        if self.requires_model and not context.model:
            return self.result(
                Status.SKIP,
                'Skipped because no model was specified or discovered.',
            )
        started = time.perf_counter()
        try:
            result = self.run(context)
        except Exception as exc:  # noqa: BLE001 - converted into a report entry
            failure_kind = failure_kind_for_exception(exc)
            info = error_info(
                exc,
                failure_kind=failure_kind,
                secrets=context.provider.config.sensitive_values,
            )
            if isinstance(exc, ProtocolError):
                message = f'Compatibility failure: {exc}'
            elif isinstance(exc, HTTPStatusError):
                message = f'Request failed: {exc}'
            elif isinstance(exc, NetworkError):
                message = f'Network error: {exc}'
            elif isinstance(exc, LLMCompatError):
                message = f'Run error: {exc}'
            else:
                message = f'Unexpected {type(exc).__name__} while running the check.'
            message = truncate(
                redact_sensitive(message, context.provider.config.sensitive_values), 1000
            )
            result = self.result(
                status_for_exception(exc),
                message,
                details={'error_type': type(exc).__name__},
                error=info,
                failure_kind=failure_kind,
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return replace(result, duration_ms=elapsed_ms)