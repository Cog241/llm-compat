'''Shared helpers for built-in agent profile modules.'''

from __future__ import annotations

from typing import Optional

from llm_compat.agents.base import AgentCapability


def capability(
    key: str,
    name: str,
    check_id: Optional[str],
    required: bool,
    description: str,
    suggestion: str,
    unavailable_reason: Optional[str] = None,
) -> AgentCapability:
    return AgentCapability(
        key=key,
        name=name,
        check_id=check_id,
        required=required,
        description=description,
        suggestion=suggestion,
        unavailable_reason=unavailable_reason,
    )