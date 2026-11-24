'''LLM Compat: compatibility testing for OpenAI-compatible AI APIs.'''

from llm_compat.compare import ComparisonReport, compare_providers
from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.report import CompatibilityReport
from llm_compat.results import CheckResult, FailureKind, Status, Summary
from llm_compat.runner import RunOptions, run_compatibility
from llm_compat.version import __version__

__all__ = [
    'CheckResult',
    'ComparisonReport',
    'CompatibilityReport',
    'FailureKind',
    'ProviderConfig',
    'RunOptions',
    'RunSettings',
    'Status',
    'Summary',
    '__version__',
    'compare_providers',
    'run_compatibility',
]