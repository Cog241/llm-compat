'''Latency and token-throughput measurement.'''

from __future__ import annotations

from statistics import mean
from typing import Any

from llm_compat.checks.base import CheckContext, CompatibilityCheck, error_info
from llm_compat.errors import LLMCompatError
from llm_compat.results import CheckResult, Status


class PerformanceCheck(CompatibilityCheck):
    id = 'latency'
    name = 'Latency and throughput'
    category = 'performance'
    description = 'Measure request latency and output-token throughput over repeated calls.'
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        samples = context.settings.samples
        latencies: list[float] = []
        throughputs: list[float] = []
        prompt_tokens = 0
        completion_tokens = 0
        failures: list[dict[str, Any]] = []
        last_error: LLMCompatError | None = None
        for index in range(samples):
            try:
                response = context.provider.chat(
                    model=context.model,
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Write one short sentence about reliable API design.',
                        }
                    ],
                    max_tokens=context.settings.max_tokens,
                    temperature=0.0,
                )
            except LLMCompatError as exc:
                last_error = exc
                failures.append({'sample': index + 1, 'error': error_info(exc).to_dict()})
                continue
            latencies.append(response.latency_ms)
            usage = response.usage
            if usage.prompt_tokens is not None:
                prompt_tokens += usage.prompt_tokens
            if usage.completion_tokens is not None:
                completion_tokens += usage.completion_tokens
                if response.latency_ms > 0:
                    throughputs.append(usage.completion_tokens / (response.latency_ms / 1000.0))
        if not latencies:
            if last_error is not None:
                raise last_error
            return self.result(Status.ERROR, 'No performance samples completed.')
        average_latency = mean(latencies)
        average_throughput = mean(throughputs) if throughputs else None
        details: dict[str, Any] = {
            'samples_requested': samples,
            'samples_succeeded': len(latencies),
            'latency_ms': {
                'avg': round(average_latency, 2),
                'min': round(min(latencies), 2),
                'max': round(max(latencies), 2),
            },
            'throughput_tokens_per_sec': (
                {
                    'avg': round(average_throughput, 2),
                    'min': round(min(throughputs), 2),
                    'max': round(max(throughputs), 2),
                }
                if average_throughput is not None
                else None
            ),
            'usage_totals': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
            },
            'failures': failures,
        }
        latency_summary = (
            f'{len(latencies)}/{samples} sample(s); avg latency {average_latency:.0f} ms '
            f'(min {min(latencies):.0f}, max {max(latencies):.0f})'
        )
        if failures:
            return self.result(
                Status.WARN,
                f'{latency_summary}; {len(failures)} sample(s) failed.',
                details=details,
            )
        if average_throughput is None:
            return self.result(
                Status.WARN,
                f'{latency_summary}; token throughput unavailable because usage was not reported.',
                details=details,
            )
        return self.result(
            Status.PASS,
            f'{latency_summary}; {average_throughput:.1f} output tokens/sec.',
            details=details,
        )
