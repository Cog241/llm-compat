'''The ordered catalog of built-in compatibility checks.'''

from __future__ import annotations

from typing import Sequence

from llm_compat.checks.base import CompatibilityCheck
from llm_compat.checks.chat import ChatCompletionCheck
from llm_compat.checks.connectivity import AuthenticationCheck, ConnectivityCheck
from llm_compat.checks.models import ModelDiscoveryCheck
from llm_compat.checks.performance import PerformanceCheck
from llm_compat.checks.streaming import StreamingCheck
from llm_compat.checks.structured import StructuredJSONCheck
from llm_compat.checks.tools import ToolCallingCheck
from llm_compat.errors import ConfigurationError

ALL_CHECKS: tuple[CompatibilityCheck, ...] = (
    ConnectivityCheck(),
    AuthenticationCheck(),
    ModelDiscoveryCheck(),
    ChatCompletionCheck(),
    StreamingCheck(),
    PerformanceCheck(),
    ToolCallingCheck(),
    StructuredJSONCheck(),
)


def available_checks() -> tuple[CompatibilityCheck, ...]:
    return ALL_CHECKS


def select_checks(
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> list[CompatibilityCheck]:
    '''Filter the catalog while preserving its intentional execution order.'''
    known = {check.id for check in ALL_CHECKS}
    requested = set(only) if only else known
    unknown = sorted(requested - known)
    if unknown:
        raise ConfigurationError(
            f'Unknown check id(s): {", ".join(unknown)}. Use llm-compat checks to list ids.'
        )
    unknown_skips = sorted(set(skip) - known)
    if unknown_skips:
        raise ConfigurationError(
            f'Unknown check id(s) in --skip: {", ".join(unknown_skips)}.'
        )
    selected = [
        check for check in ALL_CHECKS if check.id in requested and check.id not in set(skip)
    ]
    if not selected:
        raise ConfigurationError('No checks were selected for this run.')
    return selected
