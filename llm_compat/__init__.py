'''LLM Compat: compatibility testing for OpenAI-compatible AI APIs.'''

from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.report import CompatibilityReport
from llm_compat.results import CheckResult, Status, Summary
from llm_compat.runner import RunOptions, run_compatibility
from llm_compat.version import __version__

__all__ = [
    'CheckResult',
    'CompatibilityReport',
    'ProviderConfig',
    'RunOptions',
    'RunSettings',
    'Status',
    'Summary',
    '__version__',
    'run_compatibility',
]
