'''Roo Code profile. Basis: Roo Code OpenAI-compatible provider documentation.'''

from __future__ import annotations

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles._common import capability

PROFILE = AgentProfile(
    key='roo-code',
    name='Roo Code',
    summary='VS Code coding agent that requires native OpenAI-compatible tool calling.',
    docs_url='https://roocodeinc.github.io/Roo-Code/providers/openai-compatible/',
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
            'Enable SSE streaming and verify the endpoint does not buffer the response.',
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
            'Native multi-turn tool calling',
            'tools',
            True,
            'Return a tool call, accept the tool result, and produce a final answer.',
            'Roo Code requires native tool calling; switch to a model/provider that supports it.',
        ),
        capability(
            'tools_stream',
            'Streamed native tool calls',
            'tools_stream',
            True,
            'Assemble tool-call argument fragments from streaming deltas.',
            'Roo Code requires streamed native tool calls; test and enable the provider path.',
        ),
        capability(
            'json_schema',
            'Structured output',
            'json_schema',
            False,
            'Return JSON matching a strict JSON Schema response_format.',
            'Optional; use a model with strict-schema support if the workflow needs it.',
        ),
        capability(
            'tools_parallel',
            'Parallel tool calls',
            'tools_parallel',
            False,
            'Return more than one independent tool call in one turn.',
            'Disable parallel tool calls or use a provider that supports them.',
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
        'Roo Code documents native tool calling as a hard requirement for OpenAI-compatible providers.',
        'Image and prompt-caching settings are outside these protocol checks.',
    ),
)