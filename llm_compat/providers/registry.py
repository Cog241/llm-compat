'''Built-in OpenAI-compatible provider presets and construction.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_compat.config import ProviderConfig
from llm_compat.http import HttpClient
from llm_compat.providers.base import Provider
from llm_compat.providers.openai import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderPreset:
    '''Non-secret connection defaults for a known provider.'''

    key: str
    display_name: str
    base_url: str
    api_key_env: Optional[str] = None


PRESETS: dict[str, ProviderPreset] = {
    'openai': ProviderPreset(
        'openai', 'OpenAI', 'https://api.openai.com/v1', 'OPENAI_API_KEY'
    ),
    'groq': ProviderPreset(
        'groq', 'Groq', 'https://api.groq.com/openai/v1', 'GROQ_API_KEY'
    ),
    'together': ProviderPreset(
        'together', 'Together AI', 'https://api.together.xyz/v1', 'TOGETHER_API_KEY'
    ),
    'openrouter': ProviderPreset(
        'openrouter', 'OpenRouter', 'https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'
    ),
    'deepseek': ProviderPreset(
        'deepseek', 'DeepSeek', 'https://api.deepseek.com/v1', 'DEEPSEEK_API_KEY'
    ),
    'ollama': ProviderPreset(
        'ollama', 'Ollama (local)', 'http://localhost:11434/v1', None
    ),
}


def get_preset(key: str) -> Optional[ProviderPreset]:
    '''Look up a provider preset, case-insensitively.'''
    if not isinstance(key, str):
        return None
    return PRESETS.get(key.strip().lower())


def list_presets() -> list[ProviderPreset]:
    '''Return presets in the order they are shown by the CLI.'''
    return list(PRESETS.values())


def build_provider(
    config: ProviderConfig,
    http: Optional[HttpClient] = None,
) -> Provider:
    '''Build the provider adapter for a run.'''
    return OpenAICompatibleProvider(config, http=http)
