'''Aider profile. Basis: https://aider.chat/docs/llms/openai-compat.html.'''

from __future__ import annotations

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles._common import capability

PROFILE = AgentProfile(
    key='aider',
    name='Aider',
    summary='Terminal pair-programming agent that uses OpenAI-compatible chat completions.',
    docs_url='https://aider.chat/docs/llms/openai-compat.html',
    protocol='chat_completions',
    capabilities=(
        capability(
            'chat',
            'Chat completions',
            'chat',
            True,
            'Send a normal /chat/completions response with assistant text.',
            'Set the endpoint with --openai-api-base and verify the model id.',
        ),
        capability(
            'streaming',
            'Streaming',
            'streaming',
            False,
            'Consume Server-Sent Events until finish or [DONE].',
            'Streaming is recommended; Aider can fall back to non-streaming requests.',
        ),
        capability(
            'system',
            'System messages',
            'system',
            True,
            'Honor an OpenAI system message in the conversation.',
            'Aider relies on strong system/instruction following; verify it before editing code.',
        ),
        capability(
            'tools',
            'Tool calling',
            'tools',
            False,
            'Return a tool call and use the returned result.',
            'Aider primarily uses edit formats rather than function tools; tool use is optional.',
        ),
        capability(
            'tools_stream',
            'Streamed tool calls',
            'tools_stream',
            False,
            'Assemble tool-call argument fragments from streaming deltas.',
            'Optional for Aider workflows that do not use function tools.',
        ),
        capability(
            'json_schema',
            'Structured output',
            'json_schema',
            False,
            'Return JSON matching a strict JSON Schema response_format.',
            'Optional; use a model and workflow that do not depend on strict schema output.',
        ),
        capability(
            'reasoning',
            'Reasoning-model request shape',
            'reasoning',
            False,
            'Accept max_completion_tokens without temperature for reasoning models.',
            'Use a non-reasoning model or a provider adapter that translates the request shape.',
        ),
    ),
    notes=(
        'Aider model warnings are separate from protocol compatibility; use a model Aider recognizes or configure it explicitly.',
    ),
)