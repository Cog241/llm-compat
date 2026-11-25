'''OpenCode profile. Basis: https://opencode.ai/docs/providers/ (custom openai-compatible package).'''

from __future__ import annotations

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles._common import capability

PROFILE = AgentProfile(
    key='opencode',
    name='OpenCode',
    summary='Terminal coding agent using an OpenAI-compatible provider package.',
    docs_url='https://opencode.ai/docs/providers/',
    protocol='chat_completions',
    capabilities=(
        capability(
            'chat',
            'Chat completions',
            'chat',
            True,
            'Send a normal /chat/completions response with assistant text.',
            'Enable the OpenAI-compatible chat endpoint and verify the model id.',
        ),
        capability(
            'streaming',
            'Streaming',
            'streaming',
            True,
            'Consume Server-Sent Events until finish or [DONE].',
            'Enable SSE streaming and check the endpoint does not buffer the response.',
        ),
        capability(
            'system',
            'System messages',
            'system',
            True,
            'Honor an OpenAI system message in the conversation.',
            'Verify that system/developer instructions are accepted and followed.',
        ),
        capability(
            'tools',
            'Multi-turn tool calling',
            'tools',
            True,
            'Return a tool call, accept the tool result, and produce a final answer.',
            'Enable OpenAI function tools and verify the model can use returned tool results.',
        ),
        capability(
            'tools_stream',
            'Streamed tool calls',
            'tools_stream',
            True,
            'Assemble tool-call argument fragments from streaming deltas.',
            'Enable streamed tool deltas; if unavailable, use an adapter or a non-streaming tool path.',
        ),
        capability(
            'json_schema',
            'Structured output',
            'json_schema',
            False,
            'Return JSON matching a strict JSON Schema response_format.',
            'Use a model with strict-schema support, or parse/validate JSON in the agent layer.',
        ),
        capability(
            'tools_parallel',
            'Parallel tool calls',
            'tools_parallel',
            False,
            'Return more than one independent tool call in one turn.',
            'Disable parallel tool calls or use a model/provider that supports them.',
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
        'OpenCode also supports many catalog providers; this profile checks the documented OpenAI-compatible path.',
        'Tool and streaming requirements are inferred from the agent provider contract, not from an actual OpenCode run.',
    ),
)