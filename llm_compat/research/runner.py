'''Resumable, explicitly authorized research experiment runner.'''

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from llm_compat.checks.base import CheckContext
from llm_compat.checks.registry import select_checks
from llm_compat.errors import AuthorizationRequiredError, ConfigurationError
from llm_compat.providers.base import select_chat_model
from llm_compat.providers.registry import build_provider
from llm_compat.research.config import ExperimentConfig, ResolvedTarget
from llm_compat.research.export import (
    build_summary,
    environment_metadata,
    render_methodology,
    write_json,
    write_standard_exports,
    write_text,
)
from llm_compat.research.models import ExperimentReport, TrialResult
from llm_compat.research.recording import (
    RecordingProvider,
    RequestBudgetExceeded,
    RequestController,
)
from llm_compat.results import summarize
from llm_compat.utils import redact_sensitive, redact_value, safe_url, truncate, utc_now


def _prepare_output_dir(
    output_dir: Path,
    *,
    resume: bool,
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigurationError(f'Research output path {output_dir} is not a directory.')
    if output_dir.exists() and not resume:
        entries = list(output_dir.iterdir())
        if entries:
            raise ConfigurationError(
                f'Research output directory {output_dir} is not empty. Set resume=true to '
                'append to an existing experiment, or choose a new output_dir.'
            )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f'Could not create research output directory: {exc}') from exc


def _load_existing(
    output_dir: Path,
    *,
    fingerprint: str,
) -> tuple[list[dict[str, Any]], str]:
    path = output_dir / 'raw_results.json'
    if not path.exists():
        return [], utc_now()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError) as exc:
        raise ConfigurationError(f'Could not read existing raw_results.json: {exc}') from exc
    if not isinstance(payload, dict):
        raise ConfigurationError('Existing raw_results.json is not a JSON object.')
    existing_fingerprint = payload.get('config_fingerprint')
    if existing_fingerprint != fingerprint:
        raise ConfigurationError(
            'Existing raw_results.json has a different config fingerprint. '
            'Choose a new output directory or restore the original config before resuming.'
        )
    trials = payload.get('trials', [])
    if not isinstance(trials, list):
        raise ConfigurationError('Existing raw_results.json has an invalid trials list.')
    seen_ids: set[str] = set()
    for trial in trials:
        if not isinstance(trial, dict) or not isinstance(trial.get('trial_id'), str):
            raise ConfigurationError('Existing raw_results.json contains an invalid trial.')
        trial_id = trial['trial_id']
        if trial_id in seen_ids:
            raise ConfigurationError(
                f'Existing raw_results.json contains duplicate trial id {trial_id!r}.'
            )
        seen_ids.add(trial_id)
    started_at = payload.get('started_at')
    if not isinstance(started_at, str) or not started_at:
        started_at = utc_now()
    return trials, started_at


def _tasks(
    config: ExperimentConfig,
) -> list[tuple[int, int, int]]:
    tasks = [
        (target_index, trial_index)
        for target_index in range(len(config.targets))
        for trial_index in range(1, config.trials + 1)
    ]
    if config.randomize_order:
        random.Random(config.seed).shuffle(tasks)
    return [
        (target_index, trial_index, order_index)
        for order_index, (target_index, trial_index) in enumerate(tasks)
    ]


def _trial_id(target_index: int, trial_index: int) -> str:
    return f't{target_index + 1:03d}-r{trial_index:03d}'


def _conditions(
    config: ExperimentConfig,
    resolved: ResolvedTarget,
    *,
    trial_index: int,
    order_index: int,
    model_error: Optional[str],
) -> dict[str, Any]:
    conditions = {
        'seed': config.seed,
        'trial_index': trial_index,
        'order_index': order_index,
        'samples': config.samples,
        'max_tokens': config.max_tokens,
        'checks': list(config.checks),
        'all_checks': config.all_checks,
        'randomize_order': config.randomize_order,
        'request_budget': config.request_budget,
        'rate_limit_per_second': config.rate_limit_per_second,
        'target': resolved.to_safe_dict(),
    }
    if model_error:
        conditions['model_discovery_error'] = model_error
    return conditions


def _run_trial(
    config: ExperimentConfig,
    resolved: ResolvedTarget,
    target_index: int,
    trial_index: int,
    order_index: int,
    controller: RequestController,
) -> TrialResult:
    started_at = utc_now()
    started = time.perf_counter()
    provider = build_provider(resolved.config)
    recorder = RecordingProvider(provider, controller)
    settings = config.run_settings()
    model = resolved.requested_model
    model_source = resolved.model_source
    model_error: Optional[str] = None
    if model is None:
        try:
            models = recorder.list_models()
            model = select_chat_model(models)
        except Exception as exc:  # noqa: BLE001 - failure is recorded and reported
            model_error = truncate(
                redact_sensitive(str(exc), resolved.config.sensitive_values), 1000
            )
    checks = select_checks(only=config.checks)
    results: list[Any] = []
    status = 'completed'
    trial_error: Optional[str] = None
    try:
        context = CheckContext(provider=recorder, model=model, settings=settings)
        for check in checks:
            if controller.exhausted:
                status = 'budget_exhausted'
                break
            recorder.check_id = check.id
            results.append(check.execute(context))
        if controller.exhausted and len(results) < len(checks):
            status = 'budget_exhausted'
    except RequestBudgetExceeded as exc:
        status = 'budget_exhausted'
        trial_error = truncate(
            redact_sensitive(str(exc), resolved.config.sensitive_values), 1000
        )
    except Exception as exc:  # noqa: BLE001 - produce a resumable error trial
        status = 'error'
        trial_error = truncate(
            redact_sensitive(str(exc), resolved.config.sensitive_values), 1000
        )
    finally:
        recorder.check_id = None
    finished_at = utc_now()
    duration_ms = (time.perf_counter() - started) * 1000.0
    summary = summarize(results)
    secrets = resolved.config.sensitive_values
    protocol_payload = {
        'results': [result.to_dict() for result in results],
        'summary': summary.to_dict(),
    }
    return TrialResult(
        trial_id=_trial_id(target_index, trial_index),
        target_name=resolved.target.name,
        provider=resolved.target.provider,
        base_url=safe_url(resolved.config.base_url),
        trial_index=trial_index,
        order_index=order_index,
        requested_model=resolved.requested_model,
        model=model,
        model_source=model_source,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        status=status,
        verdict=summary.verdict,
        summary=redact_value(protocol_payload['summary'], secrets),
        checks=tuple(redact_value(protocol_payload['results'], secrets)),
        requests=tuple(redact_value(record.to_dict(), secrets) for record in recorder.records),
        conditions=_conditions(
            config,
            resolved,
            trial_index=trial_index,
            order_index=order_index,
            model_error=model_error,
        ),
        error=trial_error,
    )


def _raw_payload(report: ExperimentReport) -> dict[str, Any]:
    return report.to_dict()


def _experiment_file(
    config: ExperimentConfig,
    resolved: Sequence[ResolvedTarget],
    fingerprint: str,
) -> dict[str, Any]:
    return {
        'schema_version': '1.0',
        'config_fingerprint': fingerprint,
        'config': config.to_dict(),
        'resolved_targets': [target.to_safe_dict() for target in resolved],
    }


def _report(
    config: ExperimentConfig,
    resolved: Sequence[ResolvedTarget],
    fingerprint: str,
    output_dir: Path,
    started_at: str,
    environment: dict[str, Any],
    trials: list[TrialResult],
    *,
    requests_used: int,
) -> ExperimentReport:
    planned = len(config.targets) * config.trials
    summary = build_summary(
        trials,
        request_budget=config.request_budget,
        requests_used=requests_used,
        budget_exhausted=(
            requests_used >= config.request_budget
            and (
                len(trials) < planned
                or any(trial.status == 'budget_exhausted' for trial in trials)
            )
        ),
    )
    summary['overall']['planned_trials'] = planned
    summary['overall']['remaining_trials'] = max(0, planned - len(trials))
    return ExperimentReport(
        config=config,
        resolved_targets=resolved,
        fingerprint=fingerprint,
        output_dir=str(output_dir),
        started_at=started_at,
        finished_at=utc_now(),
        environment=environment,
        trials=trials,
        summary=summary,
        secrets=tuple(
            dict.fromkeys(
                secret for target in resolved for secret in target.config.sensitive_values
            )
        ),
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    authorize: bool = False,
    output_dir: Optional[str | Path] = None,
) -> ExperimentReport:
    '''Run a research experiment after explicit authorization.'''
    if authorize is not True:
        raise AuthorizationRequiredError()
    resolved = config.resolve_targets()
    destination = Path(output_dir or config.output_dir).expanduser()
    resolved_config = config.with_output_dir(str(destination))
    fingerprint = resolved_config.fingerprint(resolved)
    _prepare_output_dir(destination, resume=resolved_config.resume)
    existing, started_at = _load_existing(destination, fingerprint=fingerprint)
    existing_ids = {str(trial['trial_id']) for trial in existing}
    tasks = _tasks(resolved_config)
    task_ids = {_trial_id(target_index, trial_index) for target_index, trial_index, _ in tasks}
    unknown = existing_ids - task_ids
    if unknown:
        raise ConfigurationError(
            'Existing raw_results.json contains trials that do not belong to this configuration: '
            + ', '.join(sorted(unknown)[:5])
        )
    requests_used = sum(
        len(trial.get('requests', []))
        for trial in existing
        if isinstance(trial.get('requests'), list)
    )
    controller = RequestController(
        resolved_config.request_budget,
        resolved_config.rate_limit_per_second,
        used=requests_used,
    )
    environment = environment_metadata()
    trials: list[TrialResult] = []
    for trial in existing:
        trials.append(_trial_from_existing(trial))
    initial = _report(
        resolved_config,
        resolved,
        fingerprint,
        destination,
        started_at,
        environment,
        trials,
        requests_used=controller.used,
    )
    write_json(destination / 'raw_results.json', _raw_payload(initial), initial.secrets)
    write_json(
        destination / 'experiment.json',
        _experiment_file(resolved_config, resolved, fingerprint),
        initial.secrets,
    )
    for target_index, trial_index, order_index in tasks:
        trial_id = _trial_id(target_index, trial_index)
        if trial_id in existing_ids:
            continue
        if controller.exhausted:
            break
        trial = _run_trial(
            resolved_config,
            resolved[target_index],
            target_index,
            trial_index,
            order_index,
            controller,
        )
        trials.append(trial)
        write_json(
            destination / 'raw_results.json',
            _raw_payload(
                _report(
                    resolved_config,
                    resolved,
                    fingerprint,
                    destination,
                    started_at,
                    environment,
                    trials,
                    requests_used=controller.used,
                )
            ),
            initial.secrets,
        )
        if trial.status == 'budget_exhausted':
            break
    final = _report(
        resolved_config,
        resolved,
        fingerprint,
        destination,
        started_at,
        environment,
        trials,
        requests_used=controller.used,
    )
    raw = _raw_payload(final)
    write_json(destination / 'raw_results.json', raw, final.secrets)
    methodology = render_methodology(
        resolved_config,
        resolved,
        fingerprint=fingerprint,
        summary=final.summary,
        requests_used=controller.used,
    )
    write_text(destination / 'methodology.md', methodology)
    write_standard_exports(
        destination,
        trials=trials,
        summary=final.summary,
        methodology=methodology,
        environment=environment,
        experiment=_experiment_file(resolved_config, resolved, fingerprint),
        secrets=final.secrets,
    )
    return final


def _trial_from_existing(raw: dict[str, Any]) -> TrialResult:
    '''Rehydrate only the fields needed for summary and exports.'''
    checks = raw.get('checks', [])
    requests = raw.get('requests', [])
    try:
        return _build_existing_trial(raw, checks, requests)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f'Existing raw_results.json contains an invalid trial: {exc}'
        ) from exc


def _build_existing_trial(
    raw: dict[str, Any],
    checks: object,
    requests: object,
) -> TrialResult:
    return TrialResult(
        trial_id=str(raw['trial_id']),
        target_name=str(raw.get('target_name', '')),
        provider=str(raw.get('provider', '')),
        base_url=str(raw.get('base_url', '')),
        trial_index=int(raw.get('trial_index', 0)),
        order_index=int(raw.get('order_index', 0)),
        requested_model=raw.get('requested_model'),
        model=raw.get('model'),
        model_source=str(raw.get('model_source', 'unknown')),
        started_at=str(raw.get('started_at', '')),
        finished_at=str(raw.get('finished_at', '')),
        duration_ms=float(raw.get('duration_ms', 0.0)),
        status=str(raw.get('status', 'completed')),
        verdict=str(raw.get('verdict', 'inconclusive')),
        summary=raw.get('summary') if isinstance(raw.get('summary'), dict) else {},
        checks=tuple(item for item in checks if isinstance(item, dict)) if isinstance(checks, list) else (),
        requests=(
            tuple(item for item in requests if isinstance(item, dict))
            if isinstance(requests, list)
            else ()
        ),
        conditions=raw.get('conditions') if isinstance(raw.get('conditions'), dict) else {},
        error=raw.get('error') if isinstance(raw.get('error'), str) else None,
    )


__all__ = ['run_experiment']