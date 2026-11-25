'''Coding-agent compatibility profiles and evaluation.'''

from llm_compat.agents.base import (
    AgentCapability,
    AgentExecution,
    AgentProfile,
    AgentReport,
    CapabilityResult,
    evaluate_agent,
    render_agent_text,
)
from llm_compat.agents.registry import get_profile, list_profiles

__all__ = [
    'AgentCapability',
    'AgentExecution',
    'AgentProfile',
    'AgentReport',
    'CapabilityResult',
    'evaluate_agent',
    'get_profile',
    'list_profiles',
    'render_agent_text',
]