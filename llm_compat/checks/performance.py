'''Latency, time-to-first-token, and token-throughput measurement.'''

from __future__ import annotations

import math
import time
from statistics import mean, median
from typing import Any, Optional

from llm_compat.checks.base import (
    CheckContext,
    CompatibilityCheck,
    error_info,
    failure_kind_for_exception,
)
from llm_compat.errors import LLMCompatError
from llm_compat.providers.base import Usage
from llm_compat.results import CheckResult, FailureKind, Status

P95_MIN_SAMPLES = 20


def _percentile(values: list[float], percentile: int) -> Optional[float]:
    '''Nearest-rank percentile, reported only when the sample is large enough.'''
    if len(values) < P95_MIN_SAMPLES:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _stats(values: list[float]) -> Optional[dict[str, Optional[float]]]:
    if not values:
        return None
    p95 = _percentile(values, 95)
    return {
        'avg': round(mean(values), 2),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'median': round(median(values), 2),
        'p95': round(p95, 2) if p95 is not None else None,
        'samples': len(values),
    }


class PerformanceCheck(CompatibilityCheck):
    id = 'latency'
    name = 'Latency and generation speed'
    category = 'performance'
    description = (
        'Measure time to first token, total latency, and generation speed over repeated streams.'
    )
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        samples = context.settings.samples
        latencies: list[float] = []
        ttfts: list[float] = []
        generation_speeds: list[float] = []
        end_to_end_speeds: list[float] = []
        prompt_tokens = 0
        completion_tokens = 0
        failures: list[dict[str, Any]] = []
        incomplete = 0
        usage_missing = 0
        last_error: Optional[LLMCompatError] = None
        include_usage = True
        stream_options_supported = True
        for index in range(samples):
            try:
                metric = self._measure_stream(
                    context,
                    include_usage=include_usage,
                )
            except LLMCompatError as exc:
                if include_usage and self._is_stream_options_rejection(exc):
                    include_usage = False
                    stream_options_supported = False
                    try:
                        metric = self._measure_stream(context, include_usage=False)
                    except LLMCompatError as retry_exc:
                        last_error = retry_exc
                        failures.append(
                            {
                                'sample': index + 1,
                                'error': self._failure(context, retry_exc),
                            }
                        )
                        continue
                else:
                    last_error = exc
                    failures.append(
                        {
                            'sample': index + 1,
                            'error': self._failure(context, exc),
                        }
                    )
                    continue
            total_ms, ttft_ms, usage, done, finish_reason = metric
            if ttft_ms is None:
                incomplete += 1
                failures.append(
                    {
                        'sample': index + 1,
                        'error': {
                            'type': 'EmptyStream',
                            'kind': 'protocol',
                            'message': 'The stream contained no content chunks.',
                            'status_code': None,
                            'retryable': False,
                            'hint': None,
                            'failure_kind': FailureKind.MODEL_BEHAVIOR.value,
                        },
                    }
                )
                continue
            latencies.append(total_ms)
            ttfts.append(ttft_ms)
            if not done or finish_reason is None:
                incomplete += 1
            tokens = usage.completion_tokens
            sample_usage_missing = (
                usage.prompt_tokens is None or tokens is None or tokens <= 0
            )
            if usage.prompt_tokens is not None:
                prompt_tokens += usage.prompt_tokens
            if tokens is not None:
                completion_tokens += tokens
            if sample_usage_missing:
                usage_missing += 1
            if tokens is not None and tokens > 0:
                generation_ms = max(total_ms - ttft_ms, 0.001)
                generation_speeds.append(tokens / (generation_ms / 1000.0))
                end_to_end_speeds.append(tokens / (max(total_ms, 0.001) / 1000.0))
        if not latencies:
            if last_error is not None:
                raise last_error
            return self.result(
                Status.FAIL,
                'No performance sample produced text content.',
                details={
                    'samples_requested': samples,
                    'failures': failures,
                },
                failure_kind=(
                    FailureKind.MODEL_BEHAVIOR if failures else FailureKind.UNKNOWN
                ),
            )
        average_latency = mean(latencies)
        latency_stats = _stats(latencies)
        ttft_stats = _stats(ttfts)
        generation_stats = _stats(generation_speeds)
        end_to_end_stats = _stats(end_to_end_speeds)
        p95_latency = _percentile(latencies, 95)
        details: dict[str, Any] = {
            'samples_requested': samples,
            'samples_succeeded': len(latencies),
            'stream_options_supported': stream_options_supported,
            'latency_ms': latency_stats,
            'total_latency_ms': round(average_latency, 2),
            'median_latency_ms': round(median(latencies), 2),
            'p95_latency_ms': round(p95_latency, 2) if p95_latency is not None else None,
            'p95_min_samples': P95_MIN_SAMPLES,
            'ttft_ms': ttft_stats,
            'generation_tokens_per_sec': generation_stats,
            'throughput_tokens_per_sec': generation_stats,
            'end_to_end_tokens_per_sec': end_to_end_stats,
            'usage_totals': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
            },
            'incomplete_samples': incomplete,
            'usage_missing_samples': usage_missing,
            'failures': failures,
        }
        latency_summary = (
            f'{len(latencies)}/{samples} sample(s); median latency '
            f'{median(latencies):.0f} ms, avg {average_latency:.0f} ms'
        )
        if failures:
            failure_kind = (
                failure_kind_for_exception(last_error)
                if last_error is not None
                else (
                    FailureKind.MODEL_BEHAVIOR
                    if any(
                        failure['error'].get('type') == 'EmptyStream'
                        for failure in failures
                    )
                    else FailureKind.UNKNOWN
                )
            )
            return self.result(
                Status.WARN,
                f'{latency_summary}; {len(failures)} sample(s) failed.',
                details=details,
                failure_kind=failure_kind,
            )
        if generation_stats is None:
            return self.result(
                Status.WARN,
                f'{latency_summary}; generation speed unavailable because '
                'completion token usage was missing or zero.',
                details=details,
                failure_kind=FailureKind.MALFORMED,
            )
        if incomplete:
            return self.result(
                Status.WARN,
                f'{latency_summary}; {incomplete} stream(s) ended without complete metadata.',
                details=details,
                failure_kind=FailureKind.MALFORMED,
            )
        if usage_missing:
            return self.result(
                Status.WARN,
                f'{latency_summary}; token usage was incomplete for {usage_missing} sample(s).',
                details=details,
                failure_kind=FailureKind.MALFORMED,
            )
        return self.result(
            Status.PASS,
            (
                f'{latency_summary}; first token '
                f'{ttft_stats["median"] if ttft_stats else 0:.0f} ms; '
                f'{generation_stats["avg"]:.1f} output tokens/sec.'
            ),
            details=details,
        )

    def _measure_stream(
        self,
        context: CheckContext,
        *,
        include_usage: bool,
    ) -> tuple[float, Optional[float], Usage, bool, Optional[str]]:
        assert context.model is not None
        started = time.perf_counter()
        first_token_ms: Optional[float] = None
        usage = Usage()
        done = False
        finish_reason: Optional[str] = None
        stream = context.provider.stream_chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Write one short sentence about reliable API design.',
                }
            ],
            max_tokens=context.settings.max_tokens,
            temperature=0.0,
            include_usage=include_usage,
        )
        try:
            for event in stream:
                now_ms = (time.perf_counter() - started) * 1000.0
                if event.content and first_token_ms is None:
                    first_token_ms = now_ms
                if event.usage is not None:
                    usage = event.usage
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.done:
                    done = True
                    break
        finally:
            close = getattr(stream, 'close', None)
            if callable(close):
                close()
        total_ms = (time.perf_counter() - started) * 1000.0
        return total_ms, first_token_ms, usage, done, finish_reason

    @staticmethod
    def _is_stream_options_rejection(exc: LLMCompatError) -> bool:
        if getattr(exc, 'status_code', None) not in {400, 404, 405, 422}:
            return False
        detail = f'{getattr(exc, "body", "")} {exc}'.lower()
        return 'stream_options' in detail or 'include_usage' in detail

    @staticmethod
    def _failure(context: CheckContext, exc: LLMCompatError) -> dict[str, Any]:
        return error_info(
            exc,
            failure_kind=failure_kind_for_exception(exc),
            secrets=context.provider.config.sensitive_values,
        ).to_dict()