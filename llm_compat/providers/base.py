'''Provider contracts and shared response objects.'''

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence

from llm_compat.config import ProviderConfig

Message = Mapping[str, Any]
ToolDefinition = Mapping[str, Any]


@dataclass(frozen=True)
class ModelInfo:
    '''One model returned by an endpoint model-discovery call.'''

    id: str
    owned_by: Optional[str] = None
    created: Optional[int] = None


@dataclass(frozen=True)
class Usage:
    '''Token usage reported by a provider. Missing values stay None.'''

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def available(self) -> bool:
        return any(
            value is not None
            for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        )

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
        }


@dataclass(frozen=True)
class ToolCall:
    '''A tool call requested by a chat completion.'''

    id: Optional[str]
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResponse:
    '''Normalized response from a non-streaming chat completion.'''

    content: str
    model: str
    finish_reason: Optional[str]
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    latency_ms: float = 0.0


@dataclass(frozen=True)
class StreamEvent:
    '''One meaningful Server-Sent Events chunk.'''

    content: str = ''
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    done: bool = False


class Provider(ABC):
    '''Base class for provider adapters.'''

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def auth_configured(self) -> bool:
        return self.config.auth_configured

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        '''Return models advertised by the endpoint.'''

    @abstractmethod
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float = 0.0,
        tools: Optional[Sequence[ToolDefinition]] = None,
        tool_choice: Optional[str | Mapping[str, Any]] = None,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> ChatResponse:
        '''Send one non-streaming chat completion request.'''

    @abstractmethod
    def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Iterator[StreamEvent]:
        '''Send one streaming chat completion request.'''


_EXCLUDED_MODEL_MARKERS = (
    'embedding',
    'embed-',
    'whisper',
    'transcribe',
    'speech',
    'tts',
    'audio',
    'realtime',
    'moderation',
    'rerank',
    'dall-e',
    'dalle',
    'image',
    'video',
    'sora',
    'codex',
)

_CHAT_MODEL_HINTS = (
    'gpt-5',
    'gpt-4.1',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'o4',
    'o3',
    'o1',
    'chat',
    'instruct',
    'claude',
    'gemini',
    'llama',
    'mistral',
    'qwen',
    'deepseek',
    'command',
    'sonar',
    'gemma',
    'phi',
)


def select_chat_model(models: Sequence[ModelInfo]) -> Optional[str]:
    '''Choose a plausible chat model for automatic testing, or return None.'''
    scored: list[tuple[int, int, str]] = []
    for position, model in enumerate(models):
        identifier = model.id.strip()
        lowered = identifier.lower()
        if not identifier:
            continue
        if any(marker in lowered for marker in _EXCLUDED_MODEL_MARKERS):
            continue
        score = 0
        for index, hint in enumerate(_CHAT_MODEL_HINTS):
            if hint in lowered:
                score = max(score, len(_CHAT_MODEL_HINTS) - index)
        if 'chat' in lowered:
            score += 2
        scored.append((score, position, identifier))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]
