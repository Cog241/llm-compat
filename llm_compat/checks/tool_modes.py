'''Advanced tool-calling mode checks. These are opt-in.'''

from __future__ import annotations

from typing import Any

from llm_compat.checks._tooling import (
    CLOCK_TOOL,
    WEATHER_TOOL,
    call_preview,
    parse_arguments,
    validate_weather_call,
)
from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.results import CheckResult, FailureKind, Status


def _calls_detail(
    calls: tuple[Any, ...],
    context: CheckContext,
) -> list[dict[str, Any]]:
    secrets = context.provider.config.sensitive_values
    return [call_preview(call, secrets=secrets) for call in calls]


class ForcedToolSelectionCheck(CompatibilityCheck):
    id = 'tools_forced'
    name = 'Forced tool selection'
    category = 'tools'
    description = 'Require a specific function with tool_choice and validate the returned call.'
    requires_model = True
    default_enabled = False

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Get the weather in Paris and pass Paris as the city.',
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[WEATHER_TOOL],
            tool_choice={'type': 'function', 'function': {'name': 'get_weather'}},
        )
        details = {
            'finish_reason': response.finish_reason,
            'content_preview': context.redact(response.content, 200),
            'tool_calls': _calls_detail(response.tool_calls, context),
            'usage': response.usage.to_dict(),
            'latency_ms': round(response.latency_ms, 2),
            'tool_choice': 'get_weather',
        }
        if not response.tool_calls:
            return self.result(
                Status.FAIL,
                'The model ignored the forced tool_choice and returned no tool call.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if len(response.tool_calls) != 1 or response.tool_calls[0].name != 'get_weather':
            return self.result(
                Status.FAIL,
                'The model did not return exactly the forced get_weather tool call.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        call = response.tool_calls[0]
        failure = validate_weather_call(call)
        if failure:
            return self.result(
                Status.FAIL,
                failure,
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        return self.result(
            Status.PASS,
            'Honored the forced get_weather tool choice with valid arguments.',
            details=details,
        )


class ParallelToolCallsCheck(CompatibilityCheck):
    id = 'tools_parallel'
    name = 'Parallel tool calls'
    category = 'tools'
    description = 'Request two function calls in one assistant turn and validate both calls.'
    requires_model = True
    default_enabled = False

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        'For Paris, call get_weather and get_time together in this turn. '
                        'Do not answer in text.'
                    ),
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[WEATHER_TOOL, CLOCK_TOOL],
            tool_choice='auto',
        )
        calls = response.tool_calls
        details = {
            'finish_reason': response.finish_reason,
            'content_preview': context.redact(response.content, 200),
            'tool_calls': _calls_detail(calls, context),
            'usage': response.usage.to_dict(),
            'latency_ms': round(response.latency_ms, 2),
        }
        if len(calls) != 2:
            return self.result(
                Status.FAIL,
                f'Expected exactly two parallel tool calls but received {len(calls)}.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        names = [call.name for call in calls]
        if 'get_weather' not in names or 'get_time' not in names:
            return self.result(
                Status.FAIL,
                'The parallel tool calls did not include both get_weather and get_time.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        ids = [call.id for call in calls]
        if any(not identifier or not identifier.strip() for identifier in ids) or len(
            set(ids)
        ) != len(ids):
            return self.result(
                Status.FAIL,
                'Parallel tool calls must each have a unique, usable id.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        for call in calls:
            arguments, error = parse_arguments(call)
            if error:
                return self.result(
                    Status.FAIL,
                    f'{call.name or "A tool"} returned invalid arguments: {error}',
                    details=details,
                    failure_kind=FailureKind.MODEL_BEHAVIOR,
                )
            city = arguments.get('city') if arguments else None
            if not isinstance(city, str) or not city.strip():
                return self.result(
                    Status.FAIL,
                    f'{call.name or "A tool"} did not include a city argument.',
                    details=details,
                    failure_kind=FailureKind.MODEL_BEHAVIOR,
                )
        return self.result(
            Status.PASS,
            'Returned two parallel tool calls with unique ids and valid arguments.',
            details=details,
        )


class StreamedToolCallsCheck(CompatibilityCheck):
    id = 'tools_stream'
    name = 'Streamed tool calls'
    category = 'streaming'
    description = 'Verify that tool-call deltas are assembled correctly during SSE streaming.'
    requires_model = True
    default_enabled = False

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        calls = []
        content_parts: list[str] = []
        chunks = 0
        finish_reason = None
        done = False
        stream = context.provider.stream_chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Use the get_weather tool to get the weather in Paris.',
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[WEATHER_TOOL],
            tool_choice='auto',
        )
        try:
            for event in stream:
                if event.content:
                    chunks += 1
                    content_parts.append(event.content)
                if event.tool_calls:
                    calls.extend(event.tool_calls)
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.done:
                    done = True
                    break
        finally:
            close = getattr(stream, 'close', None)
            if callable(close):
                close()
        details = {
            'content_chunks': chunks,
            'content_preview': context.redact(''.join(content_parts), 200),
            'tool_calls': _calls_detail(tuple(calls), context),
            'finish_reason': finish_reason,
            'done': done,
        }
        if not calls:
            return self.result(
                Status.FAIL,
                'The streamed response did not contain an assembled tool call.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        call = next(
            (item for item in calls if item.name == 'get_weather'),
            calls[0],
        )
        failure = validate_weather_call(call)
        if failure:
            return self.result(
                Status.FAIL,
                failure,
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if not done:
            return self.result(
                Status.WARN,
                'Assembled a streamed tool call, but the stream ended without a [DONE] marker.',
                details=details,
                failure_kind=FailureKind.MALFORMED,
            )
        if finish_reason not in {'tool_calls', 'function_call'}:
            return self.result(
                Status.WARN,
                'Assembled a streamed tool call, but the stream did not report a tool-call finish reason.',
                details=details,
                failure_kind=FailureKind.MALFORMED,
            )
        return self.result(
            Status.PASS,
            'Assembled a complete streamed get_weather call with valid arguments.',
            details=details,
        )