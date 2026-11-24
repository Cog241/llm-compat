'''Shared tool definitions and validation helpers.'''

from __future__ import annotations

import json
from typing import Any, Optional

from llm_compat.providers.base import ToolCall
from llm_compat.utils import redact_sensitive, truncate

WEATHER_TOOL = {
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

CLOCK_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_time',
        'description': 'Get the current local time for a city.',
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


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-standard JSON constant: {value}')


def parse_arguments(call: ToolCall) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    '''Return parsed JSON object arguments and an error message, if any.'''
    if not call.arguments:
        return {}, None
    try:
        parsed = json.loads(call.arguments, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None, 'The tool call arguments were not valid JSON.'
    if not isinstance(parsed, dict):
        return None, 'The tool call arguments were not a JSON object.'
    return parsed, None


def call_preview(
    call: ToolCall,
    limit: int = 300,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        'id': truncate(redact_sensitive(call.id, secrets), 200) if call.id else None,
        'name': truncate(redact_sensitive(call.name, secrets), 200),
        'arguments': truncate(redact_sensitive(call.arguments, secrets), limit),
    }


def call_as_message(call: ToolCall) -> dict[str, Any]:
    '''Render a validated tool call as an assistant message for the next turn.'''
    return {
        'id': call.id,
        'type': 'function',
        'function': {'name': call.name, 'arguments': call.arguments or '{}'},
    }


def tool_result_message(call: ToolCall, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'role': 'tool',
        'tool_call_id': call.id,
        'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
    }


def validate_weather_call(call: ToolCall) -> Optional[str]:
    '''Return a failure message, or None when the call is usable weather data.'''
    if call.name != 'get_weather':
        called = truncate(call.name, 80) if call.name else 'an unnamed function'
        return f'The model called {called} instead of get_weather.'
    if not call.id or not call.id.strip():
        return 'The tool call did not include a usable id for a follow-up tool result.'
    arguments, error = parse_arguments(call)
    if error:
        return error
    city = arguments.get('city') if arguments else None
    if not isinstance(city, str) or not city.strip():
        return 'The tool call arguments did not contain a city string.'
    return None