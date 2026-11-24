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
from llm_compat.results import CheckResult, ErrorInfo, Status
from llm_compat.utils import truncate


@dataclass(frozen=True)
class CheckContext:
    '''Everything a check needs for one run.'''

    provider: Provider
    model: Optional[str]
    settings: RunSettings


def error_info(exc: BaseException) -> ErrorInfo:
    '''Build a stable, JSON-safe error description.'''
    hint = getattr(exc, 'hint', None)
    if not isinstance(exc, LLMCompatError):
        hint = 'This is likely an LLM Compat defect. Re-run and report it if the problem persists.'
    return ErrorInfo(
        type=type(exc).__name__,
        kind=getattr(exc, 'kind', 'internal'),
        message=truncate(str(exc), 1000) or type(exc).__name__,
        status_code=getattr(exc, 'status_code', None),
        retryable=bool(getattr(exc, 'retryable', False)),
        hint=truncate(hint, 500) if hint else None,
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
    ) -> CheckResult:
        return CheckResult(
            id=self.id,
            name=self.name,
            category=self.category,
            status=status,
            message=message,
            details=details or {},
            error=error,
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
            info = error_info(exc)
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
            result = self.result(
                status_for_exception(exc),
                message,
                details={'error_type': type(exc).__name__},
                error=info,
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return replace(result, duration_ms=elapsed_ms)
