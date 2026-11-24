'''Basic non-streaming chat completion check.'''

from __future__ import annotations

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.results import CheckResult, Status
from llm_compat.utils import truncate


class ChatCompletionCheck(CompatibilityCheck):
    id = 'chat'
    name = 'Chat completion'
    category = 'chat'
    description = 'Send one standard chat completion request and validate the response shape.'
    requires_model = True

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[{'role': 'user', 'content': 'Reply with the single word: pong.'}],
            max_tokens=min(context.settings.max_tokens, 32),
            temperature=0.0,
        )
        content = response.content.strip()
        details = {
            'response_model': response.model or context.model,
            'finish_reason': response.finish_reason,
            'content_preview': truncate(content, 200),
            'usage': response.usage.to_dict(),
            'latency_ms': round(response.latency_ms, 2),
        }
        if not content:
            return self.result(
                Status.FAIL,
                'The completion response contained no text content.',
                details=details,
            )
        return self.result(
            Status.PASS,
            f'Received a valid completion in {response.latency_ms:.0f} ms.',
            details=details,
        )
