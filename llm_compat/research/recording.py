'''Provider wrapper that records sanitized request outcomes for research.'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Sequence

from llm_compat.checks.base import failure_kind_for_exception
from llm_compat.errors import LLMCompatError
from llm_compat.providers.base import (
    ChatResponse,
    ModelInfo,
    Provider,
    StreamEvent,
    ToolDefinition,
    Usage,
)
from llm_compat.utils import redact_sensitive, truncate, utc_now


class RequestBudgetExceeded(LLMCompatError):
    '''The experiment's configured request budget was exhausted.'''

    kind = 'budget'

    def __init__(self, budget: int) -> None:
        super().__init__(
            f'The request budget of {budget} was exhausted.',
            hint='Increase request_budget only after reviewing the cost and scope of the experiment.',
        )


class RequestController:
    '''Shared budget accounting and optional client-side pacing.'''

    def __init__(
        self,
        budget: int,
        rate_limit_per_second: Optional[float] = None,
        *,
        used: int = 0,
    ) -> None:
        self.budget = budget
        self.rate_limit_per_second = rate_limit_per_second
        self.used = used
        self._last_started: Optional[float] = None

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def before_request(self) -> int:
        '''Reserve one request, pacing first if configured.'''
        if self.used >= self.budget:
            raise RequestBudgetExceeded(self.budget)
        if self.rate_limit_per_second:
            minimum_interval = 1.0 / self.rate_limit_per_second
            now = time.monotonic()
            if self._last_started is not None:
                wait = minimum_interval - (now - self._last_started)
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
            self._last_started = now
        self.used += 1
        return self.used


@dataclass
class RequestRecord:
    '''One sanitized model-discovery, chat, or stream attempt.'''

    sequence: int
    operation: str
    check_id: Optional[str]
    started_at: str
    finished_at: str
    latency_ms: float
    requested_model: Optional[str]
    success: bool
    completed: bool
    response_model: Optional[str] = None
    status_code: Optional[int] = None
    failure_kind: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    stream_events: Optional[int] = None
    tool_call_count: Optional[int] = None
    model_count: Optional[int] = None
    model_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'sequence': self.sequence,
            'operation': self.operation,
            'check_id': self.check_id,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'latency_ms': round(self.latency_ms, 2),
            'requested_model': self.requested_model,
            'response_model': self.response_model,
            'success': self.success,
            'completed': self.completed,
            'status_code': self.status_code,
            'failure_kind': self.failure_kind,
            'error_type': self.error_type,
            'error': self.error,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'finish_reason': self.finish_reason,
            'stream_events': self.stream_events,
            'tool_call_count': self.tool_call_count,
            'model_count': self.model_count,
            'model_ids': list(self.model_ids),
        }


def _usage_fields(usage: Optional[Usage]) -> dict[str, Optional[int]]:
    if usage is None:
        return {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
        }
    return {
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens,
    }


class RecordingProvider(Provider):
    '''Delegate to a provider while recording outcomes and enforcing limits.'''

    def __init__(self, inner: Provider, controller: RequestController) -> None:
        super().__init__(inner.config)
        self.inner = inner
        self.controller = controller
        self.check_id: Optional[str] = None
        self.records: list[RequestRecord] = []

    def _start(self, operation: str, model: Optional[str]) -> tuple[int, str, float, str]:
        sequence = self.controller.before_request()
        started_at = utc_now()
        started = time.perf_counter()
        return sequence, operation, started, started_at

    def _error_fields(self, exc: BaseException) -> dict[str, Any]:
        failure_kind = (
            'budget' if isinstance(exc, RequestBudgetExceeded) else failure_kind_for_exception(exc).value
        )
        return {
            'status_code': getattr(exc, 'status_code', None),
            'failure_kind': failure_kind,
            'error_type': type(exc).__name__,
            'error': truncate(
                redact_sensitive(str(exc), self.config.sensitive_values), 1000
            ),
        }

    def _append(self, record: RequestRecord) -> None:
        self.records.append(record)

    def list_models(self) -> list[ModelInfo]:
        sequence, _, started, started_at = self._start('models', None)
        try:
            models = self.inner.list_models()
        except Exception as exc:
            self._append(
                RequestRecord(
                    sequence=sequence,
                    operation='models',
                    check_id=self.check_id,
                    started_at=started_at,
                    finished_at=utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    requested_model=None,
                    success=False,
                    completed=True,
                    **self._error_fields(exc),
                )
            )
            raise
        ids = tuple(model.id for model in models if getattr(model, 'id', None))
        self._append(
            RequestRecord(
                sequence=sequence,
                operation='models',
                check_id=self.check_id,
                started_at=started_at,
                finished_at=utc_now(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                requested_model=None,
                success=True,
                completed=True,
                model_count=len(models),
                model_ids=ids[:200],
            )
        )
        return models

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        reasoning: bool = False,
    ) -> ChatResponse:
        sequence, _, started, started_at = self._start('chat', model)
        try:
            response = self.inner.chat(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning=reasoning,
            )
        except Exception as exc:
            self._append(
                RequestRecord(
                    sequence=sequence,
                    operation='chat',
                    check_id=self.check_id,
                    started_at=started_at,
                    finished_at=utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    requested_model=model,
                    success=False,
                    completed=True,
                    **self._error_fields(exc),
                )
            )
            raise
        usage = _usage_fields(response.usage)
        self._append(
            RequestRecord(
                sequence=sequence,
                operation='chat',
                check_id=self.check_id,
                started_at=started_at,
                finished_at=utc_now(),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                requested_model=model,
                success=True,
                completed=True,
                response_model=response.model,
                finish_reason=response.finish_reason,
                tool_call_count=len(response.tool_calls),
                **usage,
            )
        )
        return response

    def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        include_usage: bool = False,
    ) -> Iterator[StreamEvent]:
        sequence, _, started, started_at = self._start('stream', model)
        try:
            stream = self.inner.stream_chat(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                include_usage=include_usage,
            )
        except Exception as exc:
            self._append(
                RequestRecord(
                    sequence=sequence,
                    operation='stream',
                    check_id=self.check_id,
                    started_at=started_at,
                    finished_at=utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    requested_model=model,
                    success=False,
                    completed=True,
                    **self._error_fields(exc),
                )
            )
            raise
        return self._observe_stream(stream, sequence, started, started_at, model)

    def _observe_stream(
        self,
        stream: Iterator[StreamEvent],
        sequence: int,
        started: float,
        started_at: str,
        model: str,
    ) -> Iterator[StreamEvent]:
        completed = False
        success = True
        error_fields: dict[str, Any] = {}
        events = 0
        usage: Optional[Usage] = None
        finish_reason: Optional[str] = None
        tool_calls = 0
        try:
            for event in stream:
                events += 1
                if event.usage is not None:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason
                tool_calls += len(event.tool_calls)
                yield event
            completed = True
        except GeneratorExit:
            raise
        except Exception as exc:
            success = False
            error_fields = self._error_fields(exc)
            raise
        finally:
            usage_fields = _usage_fields(usage)
            self._append(
                RequestRecord(
                    sequence=sequence,
                    operation='stream',
                    check_id=self.check_id,
                    started_at=started_at,
                    finished_at=utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    requested_model=model,
                    success=success,
                    completed=completed,
                    stream_events=events,
                    finish_reason=finish_reason,
                    tool_call_count=tool_calls,
                    **usage_fields,
                    **error_fields,
                )
            )


__all__ = [
    'RecordingProvider',
    'RequestBudgetExceeded',
    'RequestController',
    'RequestRecord',
]