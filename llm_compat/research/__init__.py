'''Configurable, reproducible provider/model research experiments.'''

from llm_compat.research.config import (
    SCHEMA_VERSION,
    ExperimentConfig,
    ResearchConfig,
    ResearchTarget,
    ResolvedTarget,
    experiment_from_dict,
    load_experiment,
)
from llm_compat.research.runner import run_experiment

__all__ = [
    'SCHEMA_VERSION',
    'ExperimentConfig',
    'ResearchConfig',
    'ResearchTarget',
    'ResolvedTarget',
    'experiment_from_dict',
    'load_experiment',
    'run_experiment',
]