'''Model discovery compatibility check.'''

from __future__ import annotations

from llm_compat.checks.base import CheckContext, CompatibilityCheck, error_info
from llm_compat.errors import HTTPStatusError
from llm_compat.results import CheckResult, FailureKind, Status


class ModelDiscoveryCheck(CompatibilityCheck):
    id = 'models'
    name = 'Model discovery'
    category = 'discovery'
    description = 'Verify that GET /models returns a usable OpenAI-style model list.'

    def run(self, context: CheckContext) -> CheckResult:
        try:
            models = context.provider.list_models()
        except HTTPStatusError as exc:
            if exc.status_code == 404:
                return self.result(
                    Status.FAIL,
                    'Model discovery is not available: /models returned HTTP 404.',
                    details={'status_code': 404},
                    error=error_info(
                        exc,
                        failure_kind=FailureKind.UNSUPPORTED,
                        secrets=context.provider.config.sensitive_values,
                    ),
                    failure_kind=FailureKind.UNSUPPORTED,
                )
            raise
        if not models:
            return self.result(
                Status.FAIL,
                'The endpoint returned an empty model list.',
                details={'count': 0, 'models': []},
                failure_kind=FailureKind.UNSUPPORTED,
            )
        identifiers = [model.id for model in models]
        sample = identifiers[:50]
        details = {
            'count': len(identifiers),
            'models': sample,
            'truncated': len(identifiers) > len(sample),
        }
        return self.result(
            Status.PASS,
            f'Discovered {len(identifiers)} model(s).',
            details=details,
        )
