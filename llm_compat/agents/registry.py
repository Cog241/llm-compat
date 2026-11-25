'''Registry of built-in agent profiles, in stable display order.'''

from __future__ import annotations

from typing import Optional

from llm_compat.agents.base import AgentProfile
from llm_compat.agents.profiles import PROFILES


def list_profiles() -> list[AgentProfile]:
    '''Return profiles in the order shown by the CLI.'''
    return list(PROFILES)


def get_profile(key: str) -> Optional[AgentProfile]:
    '''Look up a profile by key, case-insensitively.'''
    if not isinstance(key, str):
        return None
    normalized = key.strip().lower().replace('_', '-')
    for profile in PROFILES:
        if profile.key == normalized:
            return profile
    aliases = {'roo': 'roo-code', 'roocode': 'roo-code', 'open-code': 'opencode'}
    alias = aliases.get(normalized)
    if alias is None:
        return None
    for profile in PROFILES:
        if profile.key == alias:
            return profile
    return None


__all__ = ['get_profile', 'list_profiles']