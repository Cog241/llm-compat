'''Built-in compatibility checks.'''

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.checks.chat import ChatCompletionCheck
from llm_compat.checks.connectivity import AuthenticationCheck, ConnectivityCheck
from llm_compat.checks.models import ModelDiscoveryCheck
from llm_compat.checks.performance import PerformanceCheck
from llm_compat.checks.registry import ALL_CHECKS, available_checks, select_checks
from llm_compat.checks.streaming import StreamingCheck
from llm_compat.checks.structured import StructuredJSONCheck
from llm_compat.checks.tools import ToolCallingCheck

__all__ = [
    'ALL_CHECKS',
    'AuthenticationCheck',
    'ChatCompletionCheck',
    'CheckContext',
    'CompatibilityCheck',
    'ConnectivityCheck',
    'ModelDiscoveryCheck',
    'PerformanceCheck',
    'StreamingCheck',
    'StructuredJSONCheck',
    'ToolCallingCheck',
    'available_checks',
    'select_checks',
]
