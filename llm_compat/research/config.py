'''Validated, reproducible research experiment configuration.'''

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from llm_compat.checks.registry import select_checks
from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.errors import ConfigurationError
from llm_compat.providers.registry import get_preset
from llm_compat.utils import safe_url
from llm_compat.version import __version__

SCHEMA_VERSION = '1.0'
_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_TOP_LEVEL_KEYS = {
    'schema_version',
    'name',
    'description',
    'seed',
    'trials',
    'samples',
    'max_tokens',
    'randomize_order',
    'request_budget',
    'rate_limit_per_second',
    'checks',
    'all_checks',
    'output_dir',
    'resume',
    'targets',
}
_TARGET_KEYS = {
    'name',
    'provider',
    'base_url',
    'model',
    'model_env',
    'api_key_env',
    'timeout',
    'allow_insecure_http',
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(str(key) for key in set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f'{where} contains unknown field(s): {", ".join(unknown)}.')


def _require_string(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f'{where}.{key} must be a non-empty string.')
    return value.strip()


def _optional_string(mapping: Mapping[str, Any], key: str, where: str) -> Optional[str]:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f'{where}.{key} must be a non-empty string or null.')
    return value.strip()


@dataclass(frozen=True)
class ResearchTarget:
    '''One independently configured provider/model under test.'''

    name: str
    provider: str = 'custom'
    base_url: Optional[str] = None
    model: Optional[str] = None
    model_env: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: float = 30.0
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError('target.name must be a non-empty string.')
        object.__setattr__(self, 'name', self.name.strip())
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ConfigurationError(f'target {self.name!r}: provider must be a non-empty string.')
        object.__setattr__(self, 'provider', self.provider.strip())
        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ConfigurationError(f'target {self.name!r}: base_url must be a string or null.')
        if self.model is not None and not isinstance(self.model, str):
            raise ConfigurationError(f'target {self.name!r}: model must be a string or null.')
        if self.model_env is not None and not isinstance(self.model_env, str):
            raise ConfigurationError(f'target {self.name!r}: model_env must be a string or null.')
        if self.api_key_env is not None and not isinstance(self.api_key_env, str):
            raise ConfigurationError(
                f'target {self.name!r}: api_key_env must be a string or null.'
            )
        if self.model is not None and self.model_env is not None:
            raise ConfigurationError(
                f'target {self.name!r}: set either model or model_env, not both.'
            )
        if isinstance(self.timeout, bool) or not _is_number(self.timeout):
            raise ConfigurationError(f'target {self.name!r}: timeout must be a positive number.')
        timeout = float(self.timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError(
                f'target {self.name!r}: timeout must be a finite number greater than zero.'
            )
        if not isinstance(self.allow_insecure_http, bool):
            raise ConfigurationError(
                f'target {self.name!r}: allow_insecure_http must be a boolean.'
            )
        object.__setattr__(self, 'timeout', timeout)
        base_url = self.base_url.strip() if self.base_url else None
        model = self.model.strip() if self.model else None
        object.__setattr__(self, 'base_url', base_url or None)
        object.__setattr__(self, 'model', model or None)
        object.__setattr__(
            self, 'model_env', self.model_env.strip() if self.model_env else None
        )
        object.__setattr__(
            self, 'api_key_env', self.api_key_env.strip() if self.api_key_env else None
        )
        for label, env_name in (('model_env', self.model_env), ('api_key_env', self.api_key_env)):
            if env_name is not None and not _ENV_NAME.fullmatch(env_name):
                raise ConfigurationError(
                    f'target {self.name!r}: {label} must be a valid environment variable name.'
                )

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any], index: int) -> 'ResearchTarget':
        where = f'targets[{index}]'
        if not isinstance(mapping, Mapping):
            raise ConfigurationError(f'{where} must be an object.')
        if 'api_key' in mapping:
            raise ConfigurationError(
                f'{where}.api_key is not accepted. Use api_key_env with the name of an '
                'environment variable; literal credentials must not be stored in config files.'
            )
        _reject_unknown(mapping, _TARGET_KEYS, where)
        raw_timeout = mapping.get('timeout', 30.0)
        if isinstance(raw_timeout, bool) or not _is_number(raw_timeout):
            raise ConfigurationError(f'{where}.timeout must be a positive number.')
        raw_insecure = mapping.get('allow_insecure_http', False)
        if not isinstance(raw_insecure, bool):
            raise ConfigurationError(f'{where}.allow_insecure_http must be a boolean.')
        return cls(
            name=_require_string(mapping, 'name', where),
            provider=_optional_string(mapping, 'provider', where) or 'custom',
            base_url=_optional_string(mapping, 'base_url', where),
            model=_optional_string(mapping, 'model', where),
            model_env=_optional_string(mapping, 'model_env', where),
            api_key_env=_optional_string(mapping, 'api_key_env', where),
            timeout=float(raw_timeout),
            allow_insecure_http=raw_insecure,
        )

    def resolve(self, *, require_credentials: bool = True) -> 'ResolvedTarget':
        preset = get_preset(self.provider)
        if preset is None and self.provider != 'custom':
            raise ConfigurationError(
                f'target {self.name!r}: unknown provider preset {self.provider!r}; use custom '
                'with a base_url or one of the built-in presets.'
            )
        base_url = self.base_url or (preset.base_url if preset else '')
        if not base_url:
            raise ConfigurationError(
                f'target {self.name!r}: base_url is required for provider {self.provider!r}.'
            )
        requested_model = self.model
        model_source = 'explicit' if requested_model else 'auto'
        if self.model_env is not None:
            env_model = os.environ.get(self.model_env)
            if not env_model or not env_model.strip():
                raise ConfigurationError(
                    f'target {self.name!r}: environment variable {self.model_env} is not set '
                    'or is empty.'
                )
            requested_model = env_model.strip()
            model_source = 'environment'
        api_key_env = self.api_key_env
        if api_key_env is None and preset is not None:
            api_key_env = preset.api_key_env
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if (
            require_credentials
            and self.api_key_env is not None
            and (not api_key or not api_key.strip())
        ):
            raise ConfigurationError(
                f'target {self.name!r}: environment variable {api_key_env} is not set or is empty.'
            )
        config = ProviderConfig(
            name=self.name,
            base_url=base_url,
            api_key=api_key,
            model=requested_model,
            timeout=self.timeout,
            allow_insecure_http=self.allow_insecure_http,
        )
        return ResolvedTarget(
            target=self,
            config=config,
            requested_model=requested_model,
            model_source=model_source,
            api_key_env=api_key_env,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'provider': self.provider,
            'base_url': self.base_url,
            'model': self.model,
            'model_env': self.model_env,
            'api_key_env': self.api_key_env,
            'timeout': self.timeout,
            'allow_insecure_http': self.allow_insecure_http,
        }


@dataclass(frozen=True)
class ResolvedTarget:
    '''A target plus its non-secret resolved connection settings.'''

    target: ResearchTarget
    config: ProviderConfig
    requested_model: Optional[str]
    model_source: str
    api_key_env: Optional[str]

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            'name': self.target.name,
            'provider': self.target.provider,
            'base_url': safe_url(self.config.base_url),
            'requested_model': self.requested_model,
            'model_source': self.model_source,
            'api_key_env': self.api_key_env,
            'timeout': self.config.timeout,
            'allow_insecure_http': self.config.allow_insecure_http,
            'auth_configured': self.config.auth_configured,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    '''A complete, validated research experiment definition.'''

    name: str
    targets: tuple[ResearchTarget, ...]
    description: str = ''
    seed: int = 0
    trials: int = 3
    samples: int = 3
    max_tokens: int = 128
    randomize_order: bool = True
    request_budget: int = 100
    rate_limit_per_second: Optional[float] = None
    checks: tuple[str, ...] = ()
    all_checks: bool = False
    output_dir: str = 'research-output'
    resume: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigurationError(
                f'Unsupported research schema_version {self.schema_version!r}; '
                f'expected {SCHEMA_VERSION!r}.'
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError('name must be a non-empty string.')
        object.__setattr__(self, 'name', self.name.strip())
        if not isinstance(self.description, str):
            raise ConfigurationError('description must be a string.')
        if not isinstance(self.all_checks, bool):
            raise ConfigurationError('all_checks must be a boolean.')
        if not isinstance(self.randomize_order, bool):
            raise ConfigurationError('randomize_order must be a boolean.')
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ConfigurationError('targets must be a non-empty list.')
        if not all(isinstance(target, ResearchTarget) for target in self.targets):
            raise ConfigurationError('targets must contain ResearchTarget values.')
        names = [target.name for target in self.targets]
        if len(set(names)) != len(names):
            raise ConfigurationError('target names must be unique.')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigurationError('seed must be a non-negative integer.')
        if isinstance(self.trials, bool) or not isinstance(self.trials, int):
            raise ConfigurationError('trials must be an integer.')
        if not 1 <= self.trials <= 1000:
            raise ConfigurationError('trials must be between 1 and 1000.')
        if isinstance(self.request_budget, bool) or not isinstance(self.request_budget, int):
            raise ConfigurationError('request_budget must be an integer.')
        if not 1 <= self.request_budget <= 1_000_000:
            raise ConfigurationError('request_budget must be between 1 and 1000000.')
        if self.rate_limit_per_second is not None:
            if not _is_number(self.rate_limit_per_second) or not math.isfinite(
                float(self.rate_limit_per_second)
            ):
                raise ConfigurationError(
                    'rate_limit_per_second must be a finite positive number or null.'
                )
            if float(self.rate_limit_per_second) <= 0:
                raise ConfigurationError(
                    'rate_limit_per_second must be a finite positive number or null.'
                )
            object.__setattr__(
                self, 'rate_limit_per_second', float(self.rate_limit_per_second)
            )
        if self.all_checks and self.checks:
            raise ConfigurationError('Use either checks or all_checks, not both.')
        if not isinstance(self.checks, tuple):
            raise ConfigurationError('checks must be a list of check ids.')
        if not all(isinstance(item, str) and item.strip() for item in self.checks):
            raise ConfigurationError('checks must be a list of non-empty check ids.')
        try:
            selected = select_checks(
                only=self.checks,
                skip=(),
                include_optional=self.all_checks and not self.checks,
            )
        except ConfigurationError:
            raise
        if not selected:
            raise ConfigurationError('The experiment selected no compatibility checks.')
        object.__setattr__(self, 'checks', tuple(check.id for check in selected))
        self.run_settings()
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConfigurationError('output_dir must be a non-empty string.')
        object.__setattr__(self, 'output_dir', self.output_dir.strip())
        if not isinstance(self.resume, bool):
            raise ConfigurationError('resume must be a boolean.')

    def run_settings(self) -> RunSettings:
        '''Validate and return the per-check sample/token settings.'''
        return RunSettings(samples=self.samples, max_tokens=self.max_tokens)

    def with_output_dir(self, output_dir: str) -> 'ExperimentConfig':
        '''Return a validated copy targeting a different output directory.'''
        # all_checks configs hold the expanded check ids internally. Clearing them
        # before the dataclass rebuild avoids tripping the all_checks/checks conflict
        # while preserving the exact same effective check selection.
        return replace(
            self,
            output_dir=output_dir,
            checks=() if self.all_checks else self.checks,
        )

    def resolve_targets(self, *, require_credentials: bool = True) -> tuple[ResolvedTarget, ...]:
        return tuple(
            target.resolve(require_credentials=require_credentials) for target in self.targets
        )

    def fingerprint(self, resolved: Sequence[ResolvedTarget]) -> str:
        def safe_target(target: ResolvedTarget) -> dict[str, Any]:
            payload = target.to_safe_dict()
            # Credential presence is runtime environment state, not an experiment
            # condition. Excluding it keeps validate/dry-run fingerprints stable and
            # allows key rotation without invalidating a resumable experiment.
            payload.pop('auth_configured', None)
            return payload

        payload = {
            'schema_version': self.schema_version,
            'tool_version': __version__,
            'config': self.to_dict(),
            'resolved_targets': [safe_target(target) for target in resolved],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'name': self.name,
            'description': self.description,
            'seed': self.seed,
            'trials': self.trials,
            'samples': self.samples,
            'max_tokens': self.max_tokens,
            'randomize_order': self.randomize_order,
            'request_budget': self.request_budget,
            'rate_limit_per_second': self.rate_limit_per_second,
            # An all_checks config is stored without the expanded list so the exported
            # wrapper remains a valid, re-runnable config rather than combining both.
            'checks': [] if self.all_checks else list(self.checks),
            'all_checks': self.all_checks,
            'output_dir': self.output_dir,
            'resume': self.resume,
            'targets': [target.to_dict() for target in self.targets],
        }


def experiment_from_dict(mapping: Mapping[str, Any]) -> ExperimentConfig:
    '''Validate a JSON-like mapping and build an experiment config.'''
    if not isinstance(mapping, Mapping):
        raise ConfigurationError('The experiment config must be a JSON object.')
    _reject_unknown(mapping, _TOP_LEVEL_KEYS, 'config')
    raw_targets = mapping.get('targets')
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ConfigurationError('targets must be a non-empty list.')
    targets = tuple(
        ResearchTarget.from_dict(item, index) for index, item in enumerate(raw_targets)
    )
    raw_checks = mapping.get('checks', [])
    if not isinstance(raw_checks, list) or any(not isinstance(item, str) for item in raw_checks):
        raise ConfigurationError('checks must be a list of check ids.')
    raw_description = mapping.get('description', '')
    if not isinstance(raw_description, str):
        raise ConfigurationError('config.description must be a string.')
    return ExperimentConfig(
        name=_require_string(mapping, 'name', 'config'),
        description=raw_description,
        seed=mapping.get('seed', 0),
        trials=mapping.get('trials', 3),
        samples=mapping.get('samples', 3),
        max_tokens=mapping.get('max_tokens', 128),
        randomize_order=mapping.get('randomize_order', True),
        request_budget=mapping.get('request_budget', 100),
        rate_limit_per_second=mapping.get('rate_limit_per_second'),
        checks=tuple(raw_checks),
        all_checks=mapping.get('all_checks', False),
        output_dir=mapping.get('output_dir', 'research-output'),
        resume=mapping.get('resume', True),
        schema_version=mapping.get('schema_version', SCHEMA_VERSION),
        targets=targets,
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    '''Read and validate an experiment JSON file without reading credentials.'''
    try:
        text = Path(path).read_text(encoding='utf-8')
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'Could not read experiment config {path!s}: {exc}') from exc
    try:
        mapping = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ConfigurationError(f'Experiment config {path!s} is not valid JSON: {exc}') from exc
    # experiment.json and raw_results.json wrap the config with run metadata. Accepting
    # either makes an exported experiment directly re-runnable.
    if isinstance(mapping, Mapping):
        for key in ('config', 'experiment'):
            nested = mapping.get(key)
            if isinstance(nested, Mapping):
                mapping = nested
                break
    return experiment_from_dict(mapping)


__all__ = [
    'SCHEMA_VERSION',
    'ExperimentConfig',
    'ResearchConfig',
    'ResearchTarget',
    'ResolvedTarget',
    'experiment_from_dict',
    'load_experiment',
]
ResearchConfig = ExperimentConfig
