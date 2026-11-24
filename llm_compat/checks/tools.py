'''Tool-calling compatibility check, including the result round trip.'''

from __future__ import annotations

from typing import Any

from llm_compat.checks._tooling import (
    WEATHER_TOOL,
    call_as_message,
    call_preview,
    tool_result_message,
    validate_weather_call,
)
from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.providers.base import ChatResponse
from llm_compat.results import CheckResult, FailureKind, Status


def _response_details(context: CheckContext, response: ChatResponse) -> dict[str, Any]:
    return {
        'finish_reason': response.finish_reason,
        'content_preview': context.redact(response.content, 200),
        'tool_calls': [
            call_preview(
                call,
                secrets=context.provider.config.sensitive_values,
            )
            for call in response.tool_calls
        ],
        'usage': response.usage.to_dict(),
        'latency_ms': round(response.latency_ms, 2),
    }


class ToolCallingCheck(CompatibilityCheck):
    id = 'tools'
    name = 'Tool calling (multi-turn)'
    category = 'tools'
    description = (
        'Request a tool call, return its result, and validate the follow-up answer.'
    )
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        first = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Use the get_weather tool to get the current weather in Paris.',
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[WEATHER_TOOL],
            tool_choice='auto',
        )
        details = _response_details(context, first)
        calls = first.tool_calls
        if not calls:
            return self.result(
                Status.FAIL,
                'The model did not return a tool call even though the request asked for one.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        call = next(
            (candidate for candidate in calls if candidate.name == 'get_weather'), calls[0]
        )
        failure = validate_weather_call(call)
        if failure:
            return self.result(
                Status.FAIL,
                failure,
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )

        tool_result = {
            'city': 'Paris',
            'temperature_c': 18,
            'condition': 'sunny',
            'marker': 'TOOL_RESULT_18C',
        }
        assistant_message = {
            'role': 'assistant',
            'content': first.content or None,
            'tool_calls': [call_as_message(call)],
        }
        tool_messages = [tool_result_message(call, tool_result)]
        follow_up = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Use the get_weather tool to get the current weather in Paris.',
                },
                assistant_message,
                *tool_messages,
                {
                    'role': 'user',
                    'content': (
                        'Use the tool result to answer. Include the exact marker '
                        'TOOL_RESULT_18C and do not call any more tools.'
                    ),
                },
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[WEATHER_TOOL],
            tool_choice='auto',
        )
        details['follow_up'] = _response_details(context, follow_up)
        details['tool_result'] = tool_result
        if follow_up.tool_calls:
            return self.result(
                Status.FAIL,
                'The model requested another tool instead of using the returned tool result.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        content = follow_up.content.strip()
        if not content:
            return self.result(
                Status.FAIL,
                'The follow-up response contained no text after the tool result was returned.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if 'TOOL_RESULT_18C' not in content:
            return self.result(
                Status.FAIL,
                'The follow-up response did not use the returned tool result marker.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        return self.result(
            Status.PASS,
            'Completed a get_weather round trip and used the returned tool result.',
            details=details,
        )