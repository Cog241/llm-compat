'''Structured JSON output compatibility check.'''

from __future__ import annotations

import json

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.results import CheckResult, Status
from llm_compat.utils import truncate


class StructuredJSONCheck(CompatibilityCheck):
    id = 'json'
    name = 'Structured JSON output'
    category = 'structured-output'
    description = 'Request response_format=json_object and validate the returned JSON.'
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
        details = {
            'finish_reason': response.finish_reason,
            'content_preview': truncate(content, 300),
            'usage': response.usage.to_dict(),
            'latency_ms': round(response.latency_ms, 2),
        }
        if not content:
            return self.result(
                Status.FAIL,
                'The structured-output response was empty.',
                details=details,
            )
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return self.result(
                Status.FAIL,
                'The structured-output response was not valid JSON.',
                details=details,
            )
        if not isinstance(payload, dict):
            return self.result(
                Status.FAIL,
                'The structured-output response was JSON but not an object.',
                details=details,
            )
        details['keys'] = sorted(payload.keys())
        if not isinstance(payload.get('name'), str):
            return self.result(
                Status.FAIL,
                'The JSON object did not contain a string name field.',
                details=details,
            )
        version = payload.get('version')
        if isinstance(version, bool) or not isinstance(version, int):
            return self.result(
                Status.FAIL,
                'The JSON object did not contain an integer version field.',
                details=details,
            )
        return self.result(
            Status.PASS,
            'Returned a JSON object that satisfies the requested schema.',
            details=details,
        )
