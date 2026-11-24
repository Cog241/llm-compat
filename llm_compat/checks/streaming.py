'''Server-Sent Events streaming compatibility check.'''

from __future__ import annotations

import time

from llm_compat.checks.base import (
    CheckContext,
    CompatibilityCheck,
    error_info,
    status_for_exception,
)
from llm_compat.errors import LLMCompatError
from llm_compat.results import CheckResult, Status
from llm_compat.utils import truncate


class StreamingCheck(CompatibilityCheck):
    id = 'streaming'
    name = 'Streaming compatibility'
    category = 'streaming'
    description = 'Verify SSE chunk delivery, incremental content, and a clean stream termination.'
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        started = time.perf_counter()
        parts: list[str] = []
        chunks = 0
        first_content_ms = None
        finish_reason = None
        usage = None
        done = False
        try:
            stream = context.provider.stream_chat(
                model=context.model,
                messages=[
                    {
                        'role': 'user',
                        'content': (
                            'Count from 1 to 5, separated by spaces. '
                            'Reply with only the numbers.'
                        ),
                    }
                ],
                max_tokens=min(context.settings.max_tokens, 64),
                temperature=0.0,
            )
            for event in stream:
                now_ms = (time.perf_counter() - started) * 1000.0
                if event.content:
                    chunks += 1
                    if first_content_ms is None:
                        first_content_ms = now_ms
                    parts.append(event.content)
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.usage is not None:
                    usage = event.usage
                if event.done:
                    done = True
                    break
        except LLMCompatError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            if not parts:
                raise
            return self.result(
                status_for_exception(exc),
                f'Stream failed after {chunks} content chunk(s): {exc}',
                details={
                    'content_chunks': chunks,
                    'first_content_ms': (
                        round(first_content_ms, 2) if first_content_ms is not None else None
                    ),
                    'duration_ms': round(elapsed, 2),
                    'finish_reason': finish_reason,
                    'done': done,
                    'content_preview': truncate(''.join(parts), 200),
                },
                error=error_info(exc),
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        content = ''.join(parts)
        details = {
            'content_chunks': chunks,
            'first_content_ms': (
                round(first_content_ms, 2) if first_content_ms is not None else None
            ),
            'duration_ms': round(elapsed, 2),
            'finish_reason': finish_reason,
            'done': done,
            'content_preview': truncate(content, 200),
            'usage': usage.to_dict() if usage is not None else None,
        }
        if not content:
            return self.result(
                Status.FAIL,
                'The stream ended without any text content.',
                details=details,
            )
        ttft = f'{first_content_ms:.0f} ms' if first_content_ms is not None else 'unknown'
        if not done:
            return self.result(
                Status.WARN,
                f'Received {chunks} chunk(s) with first content in {ttft}, '
                'but the stream ended without a [DONE] marker.',
                details=details,
            )
        if finish_reason is None:
            return self.result(
                Status.WARN,
                f'Received {chunks} chunk(s) with first content in {ttft}, '
                'but no chunk reported a finish reason.',
                details=details,
            )
        if chunks < 2:
            return self.result(
                Status.WARN,
                f'Received only one content chunk with first content in {ttft}; '
                'incremental streaming could not be confirmed.',
                details=details,
            )
        return self.result(
            Status.PASS,
            f'Received {chunks} content chunk(s); first content in {ttft}; '
            f'stream completed in {elapsed:.0f} ms.',
            details=details,
        )
