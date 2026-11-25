'''Serializable research trial and experiment result models.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from llm_compat.research.config import ExperimentConfig, ResolvedTarget


@dataclass
class TrialResult:
    '''One target's complete trial, with sanitized check and request data.'''

    trial_id: str
    target_name: str
    provider: str
    base_url: str
    trial_index: int
    order_index: int
    requested_model: Optional[str]
    model: Optional[str]
    model_source: str
    started_at: str
    finished_at: str
    duration_ms: float
    status: str
    verdict: str
    summary: dict[str, Any]
    checks: tuple[dict[str, Any], ...]
    requests: tuple[dict[str, Any], ...]
    conditions: dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'trial_id': self.trial_id,
            'target_name': self.target_name,
            'provider': self.provider,
            'base_url': self.base_url,
            'trial_index': self.trial_index,
            'order_index': self.order_index,
            'requested_model': self.requested_model,
            'model': self.model,
            'model_source': self.model_source,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'duration_ms': round(self.duration_ms, 2),
            'status': self.status,
            'verdict': self.verdict,
            'summary': self.summary,
            'checks': list(self.checks),
            'requests': list(self.requests),
            'conditions': self.conditions,
            'error': self.error,
        }


@dataclass
class ExperimentReport:
    '''The final in-memory experiment result used by all exports.'''

    config: ExperimentConfig
    resolved_targets: Sequence[ResolvedTarget]
    fingerprint: str
    output_dir: str
    started_at: str
    finished_at: str
    environment: dict[str, Any]
    trials: list[TrialResult]
    summary: dict[str, Any]
    secrets: tuple[str, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        from llm_compat.utils import redact_value
        from llm_compat.version import __version__

        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'config_fingerprint': self.fingerprint,
            'experiment': self.config.to_dict(),
            'resolved_targets': [target.to_safe_dict() for target in self.resolved_targets],
            'environment': self.environment,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'authorization': {
                'explicit': True,
                'statement': (
                    'The operator explicitly authorized live requests before this run.'
                ),
            },
            'summary': self.summary,
            'trials': [trial.to_dict() for trial in self.trials],
        }
        return redact_value(payload, self.secrets)


__all__ = ['ExperimentReport', 'TrialResult']