'''Structured JSON output compatibility checks.'''

from __future__ import annotations

import json

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.providers.base import ChatResponse
from llm_compat.results import CheckResult, FailureKind, Status


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-standard JSON constant: {value}')


def _load_json(content: str) -> object:
    return json.loads(content, parse_constant=_reject_json_constant)


def _base_details(context: CheckContext, content: str, response: ChatResponse) -> dict[str, object]:
    return {
        'finish_reason': response.finish_reason,
        'content_preview': context.redact(content, 300),
        'usage': response.usage.to_dict(),
        'latency_ms': round(response.latency_ms, 2),
    }


class StructuredJSONCheck(CompatibilityCheck):
    id = 'json'
    name = 'Structured JSON output'
    category = 'structured-output'
    description = 'Request response_format=json_object and validate required fields.'
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': (
                        'Return only a JSON object with exactly two fields: '
                        'name (a string) and version (an integer). '
                        'Do not wrap the JSON in markdown fences.'
                    ),
                }
            ],
            max_tokens=min(context.settings.max_tokens, 64),
            temperature=0.0,
            response_format={'type': 'json_object'},
        )
        content = response.content.strip()
        details = _base_details(context, content, response)
        if not content:
            return self.result(
                Status.FAIL,
                'The structured-output response was empty.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        try:
            payload = _load_json(content)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return self.result(
                Status.FAIL,
                'The structured-output response was not valid JSON.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if not isinstance(payload, dict):
            return self.result(
                Status.FAIL,
                'The structured-output response was JSON but not an object.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        details['keys'] = sorted(payload.keys())
        if not isinstance(payload.get('name'), str):
            return self.result(
                Status.FAIL,
                'The JSON object did not contain a string name field.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        version = payload.get('version')
        if isinstance(version, bool) or not isinstance(version, int):
            return self.result(
                Status.FAIL,
                'The JSON object did not contain an integer version field.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        return self.result(
            Status.PASS,
            'Returned valid JSON with the requested name and version fields.',
            details=details,
        )


_STRICT_SCHEMA = {
    'name': 'compat_payload',
    'strict': True,
    'schema': {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'version': {'type': 'integer'},
            'tags': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['name', 'version', 'tags'],
        'additionalProperties': False,
    },
}


class StrictJSONSchemaCheck(CompatibilityCheck):
    id = 'json_schema'
    name = 'Strict JSON Schema output'
    category = 'structured-output'
    description = 'Request response_format=json_schema with strict=true and validate every field.'
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
                        'Return one JSON object with name (string), version (integer), '
                        'and tags (array of strings). Use the name "compat" and version 2. '
                        'Do not add extra fields or markdown.'
                    ),
                }
            ],
            max_tokens=min(context.settings.max_tokens, 128),
            temperature=0.0,
            response_format={'type': 'json_schema', 'json_schema': _STRICT_SCHEMA},
        )
        content = response.content.strip()
        details = _base_details(context, content, response)
        details['response_format'] = 'json_schema'
        details['strict'] = True
        if not content:
            return self.result(
                Status.FAIL,
                'The strict-schema response was empty.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        try:
            payload = _load_json(content)
        except (json.JSONDecodeError, ValueError, RecursionError):
            return self.result(
                Status.FAIL,
                'The strict-schema response was not valid JSON.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if not isinstance(payload, dict):
            return self.result(
                Status.FAIL,
                'The strict-schema response was not a JSON object.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        expected = {'name', 'version', 'tags'}
        if set(payload) != expected:
            return self.result(
                Status.FAIL,
                'The response object did not contain exactly name, version, and tags.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if not isinstance(payload.get('name'), str):
            return self.result(
                Status.FAIL,
                'The strict-schema name field was not a string.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        version = payload.get('version')
        if isinstance(version, bool) or not isinstance(version, int):
            return self.result(
                Status.FAIL,
                'The strict-schema version field was not an integer.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        tags = payload.get('tags')
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            return self.result(
                Status.FAIL,
                'The strict-schema tags field was not an array of strings.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        details['keys'] = sorted(payload.keys())
        return self.result(
            Status.PASS,
            'Returned a JSON object satisfying the strict json_schema request.',
            details=details,
        )