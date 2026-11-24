'''Tool-calling compatibility check.'''

from __future__ import annotations

import json

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.results import CheckResult, Status
from llm_compat.utils import truncate

_WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'City name, for example Paris.',
                }
            },
            'required': ['city'],
            'additionalProperties': False,
        },
    },
}


class ToolCallingCheck(CompatibilityCheck):
    id = 'tools'
    name = 'Tool calling'
    category = 'tools'
    description = 'Request a tool call and validate the function name and JSON arguments.'
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'Use the get_weather tool to get the current weather in Paris.',
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            tools=[_WEATHER_TOOL],
            tool_choice='auto',
        )
        calls = response.tool_calls
        details = {
            'finish_reason': response.finish_reason,
            'content_preview': truncate(response.content, 200),
            'tool_calls': [
                {'name': call.name, 'arguments': truncate(call.arguments, 300)} for call in calls
            ],
            'usage': response.usage.to_dict(),
            'latency_ms': round(response.latency_ms, 2),
        }
        if not calls:
            return self.result(
                Status.FAIL,
                'The model did not return a tool call even though the request asked for one.',
                details=details,
            )
        call = next((candidate for candidate in calls if candidate.name == 'get_weather'), calls[0])
        if call.name != 'get_weather':
            return self.result(
                Status.FAIL,
                f'The model called {call.name or "an unnamed function"} instead of get_weather.',
                details=details,
            )
        if not call.id or not call.id.strip():
            return self.result(
                Status.FAIL,
                'The tool call did not include a usable id for a follow-up tool result.',
                details=details,
            )
        try:
            arguments = json.loads(call.arguments) if call.arguments else {}
        except (json.JSONDecodeError, ValueError):
            return self.result(
                Status.FAIL,
                'The tool call arguments were not valid JSON.',
                details=details,
            )
        if not isinstance(arguments, dict):
            return self.result(
                Status.FAIL,
                'The tool call arguments were not a JSON object.',
                details=details,
            )
        city = arguments.get('city')
        if not isinstance(city, str) or not city.strip():
            return self.result(
                Status.FAIL,
                'The tool call arguments did not contain a city string.',
                details=details,
            )
        return self.result(
            Status.PASS,
            'Returned get_weather with valid JSON arguments.',
            details=details,
        )
