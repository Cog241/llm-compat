'''Opt-in checks for system messages and reasoning-model request shapes.'''

from __future__ import annotations

from typing import Any

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.providers.base import ChatResponse
from llm_compat.results import CheckResult, FailureKind, Status


def _chat_details(context: CheckContext, response: ChatResponse) -> dict[str, Any]:
    return {
        'response_model': response.model or context.model,
        'finish_reason': response.finish_reason,
        'content_preview': context.redact(response.content, 240),
        'usage': response.usage.to_dict(),
        'latency_ms': round(response.latency_ms, 2),
    }


class SystemMessageCheck(CompatibilityCheck):
    id = 'system'
    name = 'System messages'
    category = 'messages'
    description = 'Verify that a system-role instruction influences the completion.'
    requires_model = True
    default_enabled = False

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        system_prompt = 'You must answer with exactly one word: azure.'
        response = context.provider.chat(
            model=context.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': 'What single word must you answer with?'},
            ],
            max_tokens=min(context.settings.max_tokens, 32),
            temperature=0.0,
        )
        details = _chat_details(context, response)
        details['system_preview'] = context.redact(system_prompt, 160)
        content = response.content.strip()
        if not content:
            return self.result(
                Status.FAIL,
                'The system-message request returned no text content.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if 'azure' not in content.casefold():
            return self.result(
                Status.FAIL,
                'The response did not follow the system-role instruction.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        return self.result(
            Status.PASS,
            'Accepted the system message and followed its instruction.',
            details=details,
        )


class ReasoningModelCheck(CompatibilityCheck):
    id = 'reasoning'
    name = 'Reasoning request shape'
    category = 'reasoning'
    description = (
        'Send the request shape used by reasoning models: max_completion_tokens and no temperature.'
    )
    requires_model = True
    default_enabled = False

    def run(self, context: CheckContext) -> CheckResult:
        assert context.model is not None
        response = context.provider.chat(
            model=context.model,
            messages=[
                {
                    'role': 'user',
                    'content': 'What is 2 + 2? Reply with only the number.',
                }
            ],
            max_tokens=min(context.settings.max_tokens, 64),
            reasoning=True,
        )
        details = _chat_details(context, response)
        details['request_shape'] = {
            'max_completion_tokens': min(context.settings.max_tokens, 64),
            'temperature_omitted': True,
        }
        content = response.content.strip()
        if not content:
            return self.result(
                Status.FAIL,
                'The reasoning-shaped request returned no text content.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        if '4' not in content:
            return self.result(
                Status.FAIL,
                'The reasoning-shaped request returned an unexpected answer.',
                details=details,
                failure_kind=FailureKind.MODEL_BEHAVIOR,
            )
        return self.result(
            Status.PASS,
            'Accepted max_completion_tokens without a temperature parameter.',
            details=details,
        )