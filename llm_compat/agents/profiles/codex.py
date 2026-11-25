'''Codex CLI profile. Basis: https://github.com/openai/codex model-provider docs.'''

from __future__ import annotations

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles._common import capability

_RESPONSES_GAP = (
    'Current Codex uses the Responses API (POST /responses), not Chat Completions. '
    'LLM Compat 0.3.0 only tests the Chat Completions surface, so this capability '
    'cannot be inferred from the checks in this release.'
)

PROFILE = AgentProfile(
    key='codex',
    name='Codex CLI',
    summary='OpenAI coding agent whose current custom-provider wire API is Responses.',
    docs_url='https://github.com/openai/codex/blob/main/codex-rs/model-provider-info/src/lib.rs',
    protocol='responses',
    capabilities=(
        capability(
            'responses_api',
            'Responses API',
            None,
            True,
            'Use the Responses API endpoint and response shape expected by Codex.',
            'Configure a provider that implements /responses; Chat Completions alone is not sufficient.',
            unavailable_reason=_RESPONSES_GAP,
        ),
        capability(
            'responses_streaming',
            'Responses streaming',
            None,
            True,
            'Consume the Responses API streaming event protocol.',
            'Verify Responses SSE events against the Codex contract before configuring Codex.',
            unavailable_reason=_RESPONSES_GAP,
        ),
        capability(
            'responses_tools',
            'Responses function tools',
            None,
            True,
            'Accept Responses-style function tools and return consumable tool calls.',
            'Verify the provider supports Responses function tools before use.',
            unavailable_reason=_RESPONSES_GAP,
        ),
    ),
    notes=(
        'Codex removed the legacy wire_api="chat" path; an OpenAI Chat Completions test is not evidence of Codex compatibility.',
        'This profile is intentionally marked inconclusive until Responses API checks are implemented.',
    ),
    verified_by='not verified: the current Codex protocol is outside this release\'s chat-completions checks',
)