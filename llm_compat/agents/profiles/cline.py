'''Cline profile. Basis: Cline OpenAI-compatible provider and chat-completions docs.'''

from __future__ import annotations

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles._common import capability

PROFILE = AgentProfile(
    key='cline',
    name='Cline',
    summary='VS Code coding agent using an OpenAI-compatible chat endpoint.',
    docs_url='https://docs.cline.bot/api/chat-completions',
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
            'Cline defaults to stream=true; enable SSE or use a gateway that provides it.',
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
            'Enable streamed tool deltas; Cline needs streamed native tool calls for its normal flow.',
        ),
        capability(
            'json_schema',
            'Structured output',
            'json_schema',
            False,
            'Return JSON matching a strict JSON Schema response_format.',
            'Use a model with strict-schema support, or validate JSON in the agent layer.',
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
        'Image and computer-use capabilities are provider/model settings in Cline and are outside these protocol checks.',
    ),
)