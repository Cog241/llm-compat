'''Sanitized JSON, CSV, environment, and methodology exports.'''

from __future__ import annotations

import csv
import json
import locale
import math
import os
import platform
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from llm_compat.research.config import ExperimentConfig, ResolvedTarget
from llm_compat.research.models import TrialResult
from llm_compat.utils import redact_sensitive, redact_value, safe_url, utc_now
from llm_compat.version import __version__


def environment_metadata() -> dict[str, Any]:
    '''Collect non-identifying runtime context for reproducibility.'''
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    return {
        'tool': {'name': 'llm-compat', 'version': __version__},
        'python': {
            'implementation': platform.python_implementation(),
            'version': platform.python_version(),
            'compiler': platform.python_compiler(),
        },
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
        },
        'locale': locale.getpreferredencoding(False),
        'timezone': now.tzname(),
        'utc_offset_minutes': int(offset.total_seconds() // 60) if offset else 0,
        'runtime_dependencies': [],
        'generated_at': utc_now(),
    }


def safe_json_dumps(payload: Any, secrets: Iterable[str] = ()) -> str:
    '''Serialize after central recursive credential redaction.'''
    cleaned = redact_value(payload, secrets)
    return json.dumps(cleaned, indent=2, ensure_ascii=False) + '\n'


def write_text(path: Path, text: str) -> None:
    '''Write a UTF-8 output file atomically.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any, secrets: Iterable[str] = ()) -> None:
    write_text(path, safe_json_dumps(payload, secrets))


def csv_safe(value: Any) -> str:
    '''Neutralize spreadsheet formula prefixes in a text CSV cell.'''
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    text = str(value)
    if text and text[0] in {'=', '+', '-', '@', '\t', '\r'}:
        return "'" + text
    return text


def _csv(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), extrasaction='ignore', lineterminator='\n'
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_safe(row.get(key)) for key in columns})
    return buffer.getvalue()


def _reported_token_sum(requests: Sequence[dict[str, Any]], field: str) -> Optional[int]:
    '''Sum provider-reported integers, preserving missing data as blank.'''
    values = [
        request[field]
        for request in requests
        if isinstance(request.get(field), int) and not isinstance(request.get(field), bool)
    ]
    return sum(values) if values else None


def write_results_csv(
    path: Path,
    trials: Sequence[TrialResult],
    *,
    secrets: Iterable[str] = (),
) -> None:
    columns = [
        'trial_id', 'order_index', 'target_name', 'provider', 'requested_model',
        'model', 'started_at', 'finished_at', 'duration_ms', 'status', 'verdict',
        'checks_run', 'passed', 'failed', 'warnings', 'errors', 'skipped',
        'request_count', 'prompt_tokens', 'completion_tokens', 'total_tokens',
        'failure_kinds',
    ]
    rows = []
    for trial in trials:
        counts = trial.summary.get('counts')
        if not isinstance(counts, dict):
            counts = {}
        failure_kinds = trial.summary.get('failure_kinds')
        if not isinstance(failure_kinds, dict):
            failure_kinds = {}
        requests = trial.requests
        rows.append(
            {
                'trial_id': trial.trial_id,
                'order_index': trial.order_index,
                'target_name': trial.target_name,
                'provider': trial.provider,
                'requested_model': trial.requested_model,
                'model': trial.model,
                'started_at': trial.started_at,
                'finished_at': trial.finished_at,
                'duration_ms': trial.duration_ms,
                'status': trial.status,
                'verdict': trial.verdict,
                'checks_run': trial.summary.get('executed_checks', 0),
                'passed': counts.get('pass', 0),
                'failed': counts.get('fail', 0),
                'warnings': counts.get('warn', 0),
                'errors': counts.get('error', 0),
                'skipped': counts.get('skip', 0),
                'request_count': len(requests),
                'prompt_tokens': _reported_token_sum(requests, 'prompt_tokens'),
                'completion_tokens': _reported_token_sum(requests, 'completion_tokens'),
                'total_tokens': _reported_token_sum(requests, 'total_tokens'),
                'failure_kinds': ';'.join(
                    f'{key}={value}' for key, value in sorted(failure_kinds.items())
                ),
            }
        )
    write_text(path, _csv(redact_value(rows, tuple(secrets)), columns))


def write_requests_csv(
    path: Path,
    trials: Sequence[TrialResult],
    *,
    secrets: Iterable[str] = (),
) -> None:
    columns = [
        'trial_id', 'order_index', 'target_name', 'provider', 'requested_model',
        'model', 'check_id', 'operation', 'sequence', 'started_at', 'finished_at',
        'latency_ms', 'success', 'completed', 'status_code', 'failure_kind',
        'error_type', 'error', 'prompt_tokens', 'completion_tokens', 'total_tokens',
        'finish_reason', 'stream_events', 'tool_call_count', 'response_model',
        'model_count', 'model_ids',
    ]
    rows = []
    for trial in trials:
        for request in trial.requests:
            row = dict(request)
            model_ids = request.get('model_ids')
            if not isinstance(model_ids, (list, tuple)):
                model_ids = ()
            row.update(
                {
                    'trial_id': trial.trial_id,
                    'order_index': trial.order_index,
                    'target_name': trial.target_name,
                    'provider': trial.provider,
                    'model': request.get('response_model') or trial.model,
                    'response_model': request.get('response_model'),
                    'model_ids': ';'.join(
                        str(item) for item in model_ids if isinstance(item, str)
                    ),
                }
            )
            rows.append(row)
    write_text(path, _csv(redact_value(rows, tuple(secrets)), columns))


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _latency_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            'count': 0,
            'mean_ms': None,
            'median_ms': None,
            'p95_ms': None,
            'p95_eligible': False,
            'min_ms': None,
            'max_ms': None,
        }
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    return {
        'count': len(ordered),
        'mean_ms': round(sum(ordered) / len(ordered), 2),
        'median_ms': round(median, 2),
        'p95_ms': round(_percentile(ordered, 95), 2) if len(ordered) >= 20 else None,
        'p95_eligible': len(ordered) >= 20,
        'min_ms': round(ordered[0], 2),
        'max_ms': round(ordered[-1], 2),
    }


def _token_totals(requests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [
            request[field]
            for request in requests
            if isinstance(request.get(field), int)
            and not isinstance(request.get(field), bool)
        ]
        totals[field] = sum(values) if values else None
        totals[f'{field}_reported_requests'] = len(values)
    return totals


def build_summary(
    trials: Sequence[TrialResult],
    *,
    request_budget: int,
    requests_used: int,
    budget_exhausted: Optional[bool] = None,
) -> dict[str, Any]:
    '''Compute descriptive statistics without inventing missing measurements.'''
    by_target: dict[str, list[TrialResult]] = {}
    for trial in trials:
        by_target.setdefault(trial.target_name, []).append(trial)

    def aggregate(group: Sequence[TrialResult]) -> dict[str, Any]:
        checks = [check for trial in group for check in trial.checks]
        requests = [request for trial in group for request in trial.requests]
        trial_statuses: dict[str, int] = {}
        verdicts: dict[str, int] = {}
        check_statuses: dict[str, int] = {}
        failure_kinds: dict[str, int] = {}
        for check in checks:
            status = str(check.get('status'))
            check_statuses[status] = check_statuses.get(status, 0) + 1
            failure = check.get('failure_kind')
            if failure:
                failure_kinds[str(failure)] = failure_kinds.get(str(failure), 0) + 1
        for request in requests:
            failure = request.get('failure_kind')
            if failure:
                failure_kinds[str(failure)] = failure_kinds.get(str(failure), 0) + 1
        for trial in group:
            trial_statuses[trial.status] = trial_statuses.get(trial.status, 0) + 1
            verdicts[trial.verdict] = verdicts.get(trial.verdict, 0) + 1
        latencies = [
            float(request['latency_ms'])
            for request in requests
            if isinstance(request.get('latency_ms'), (int, float))
            and not isinstance(request.get('latency_ms'), bool)
            and math.isfinite(float(request['latency_ms']))
        ]
        successful = sum(1 for request in requests if request.get('success') is True)
        return {
            'trials': len(group),
            'trial_statuses': trial_statuses,
            'verdicts': verdicts,
            'checks': len(checks),
            'check_statuses': check_statuses,
            'requests': len(requests),
            'successful_requests': successful,
            'failed_requests': len(requests) - successful,
            'request_success_rate': (
                round(successful / len(requests), 4) if requests else None
            ),
            'latency': _latency_stats(latencies),
            'tokens': _token_totals(requests),
            'models': sorted({str(trial.model) for trial in group if trial.model is not None}),
            'requested_models': sorted(
                {
                    str(trial.requested_model)
                    for trial in group
                    if trial.requested_model is not None
                }
            ),
            'failure_kinds': failure_kinds,
        }

    overall = aggregate(trials)
    overall['request_budget'] = request_budget
    overall['requests_used'] = requests_used
    overall['budget_exhausted'] = (
        requests_used >= request_budget
        if budget_exhausted is None
        else bool(budget_exhausted)
    )
    overall['targets'] = len(by_target)
    return {
        'overall': overall,
        'by_target': {name: aggregate(group) for name, group in sorted(by_target.items())},
    }


def _md_cell(value: Any) -> str:
    '''Escape user-controlled text for one Markdown table cell.'''
    text = (
        str(value)
        .replace('\\', '\\\\')
        .replace('\r', ' ')
        .replace('\n', ' ')
        .replace('|', '\\|')
    )
    return text


def render_methodology(
    config: ExperimentConfig,
    resolved_targets: Sequence[ResolvedTarget],
    *,
    fingerprint: str,
    summary: dict[str, Any],
    requests_used: int,
) -> str:
    '''Render a methods document that makes claims and limits explicit.'''
    secrets = tuple(
        secret for target in resolved_targets for secret in target.config.sensitive_values
    )

    def safe(value: object) -> str:
        return _md_cell(redact_sensitive(str(value), secrets))

    lines = [f'# {safe(config.name)}', '']
    if config.description:
        lines.extend([redact_sensitive(config.description, secrets), ''])
    rate = (
        f'{config.rate_limit_per_second:g} requests/second'
        if config.rate_limit_per_second
        else 'none beyond sequential check execution'
    )
    lines.extend(
        [
            '## Scope and authorization',
            '',
            'This document describes a protocol compatibility experiment, not a benchmark '
            'or a model-quality ranking. Every live request in this run required explicit '
            'operator authorization. The tool does not automatically retry transient '
            'failures; the latency check may retry once without streamed usage when a '
            'provider rejects that option.',
            '',
            '## Design',
            '',
            f'- Trials per target: {config.trials}',
            f'- Samples per latency check: {config.samples}',
            f'- Maximum completion tokens: {config.max_tokens}',
            f'- Checks: {", ".join(config.checks)}',
            f'- Randomized order: {"yes" if config.randomize_order else "no"}',
            f'- Order seed: {config.seed}',
            f'- Request budget: {config.request_budget}',
            f'- Requests used: {requests_used}',
            f'- Rate limit: {rate}',
            '',
            '## Targets',
            '',
            '| Name | Provider | Base URL | Requested model | API key env |',
            '| --- | --- | --- | --- | --- |',
        ]
    )
    for target in resolved_targets:
        lines.append(
            '| {name} | {provider} | {url} | {model} | {key} |'.format(
                name=safe(target.target.name),
                provider=safe(target.target.provider),
                url=safe(safe_url(target.config.base_url)),
                model=safe(target.requested_model or 'auto-discovered'),
                key=safe(target.api_key_env or 'none'),
            )
        )
    lines.extend(
        [
            '',
            '## Measurements',
            '',
            '- Latency is wall-clock time around one provider call, in milliseconds.',
            '- Success means no transport or protocol exception was raised for that call.',
            '- Token counts are provider-reported values; missing usage is never estimated.',
            '- p95 latency is reported only when at least 20 request latencies exist for a '
            '  target. Smaller samples show count, mean, median, min, and max instead.',
            '- Each request record includes its check id, operation, timestamps, model, '
            '  finish reason, token usage when available, and classified failure.',
            '',
            '## Output files',
            '',
            '- `raw_results.json`: sanitized experiment metadata, trials, checks, and requests.',
            '- `results.csv`: one row per trial.',
            '- `requests.csv`: one row per recorded request attempt.',
            '- `summary.json`: descriptive statistics by target and overall.',
            '- `environment.json`: tool, Python, platform, locale, and timezone metadata.',
            '- `experiment.json`: resolved config and fingerprint for resume validation.',
            '',
            '## Limitations',
            '',
            '- These checks exercise protocol behavior through a synthetic client; they do '
            '  not launch OpenCode, Codex, Cline, Aider, or Roo Code.',
            '- Provider behavior, model routing, rate limits, and prices can change after the run.',
            '- The experiment does not retry, so transient failures remain in the dataset.',
            '- A compatible protocol result is not a guarantee that an agent workflow will '
            '  succeed on complex tasks.',
            '',
            '## Reproducibility',
            '',
            f'- Config fingerprint: `{fingerprint}`',
            '- Resume skips trials already present in `raw_results.json` only when the '
            '  fingerprint matches.',
            '- Order is deterministic for the recorded seed and target list.',
            '',
        ]
    )
    overall = summary.get('overall', {})
    if overall:
        lines.extend(
            [
                '## Recorded result summary',
                '',
                f'- Targets: {overall.get("targets", 0)}',
                f'- Trials: {overall.get("trials", 0)}',
                f'- Requests: {overall.get("requests", 0)}',
                f'- Successful requests: {overall.get("successful_requests", 0)}',
                f'- Budget exhausted: {"yes" if overall.get("budget_exhausted") else "no"}',
                '',
            ]
        )
    return '\n'.join(lines)


def write_standard_exports(
    output_dir: Path,
    *,
    trials: Sequence[TrialResult],
    summary: dict[str, Any],
    methodology: str,
    environment: dict[str, Any],
    experiment: dict[str, Any],
    secrets: Iterable[str],
) -> None:
    secrets = tuple(secrets)
    write_results_csv(output_dir / 'results.csv', trials, secrets=secrets)
    write_requests_csv(output_dir / 'requests.csv', trials, secrets=secrets)
    write_json(output_dir / 'summary.json', summary, secrets)
    write_text(output_dir / 'methodology.md', redact_sensitive(methodology, secrets))
    write_json(output_dir / 'environment.json', environment, secrets)
    write_json(output_dir / 'experiment.json', experiment, secrets)


__all__ = [
    'build_summary',
    'csv_safe',
    'environment_metadata',
    'render_methodology',
    'safe_json_dumps',
    'write_json',
    'write_requests_csv',
    'write_results_csv',
    'write_standard_exports',
    'write_text',
]
