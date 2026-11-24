'''Multi-provider comparison with independent endpoints and credentials.'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from llm_compat.checks.base import CompatibilityCheck
from llm_compat.checks.registry import select_checks
from llm_compat.config import ProviderConfig
from llm_compat.errors import AuthorizationRequiredError, ConfigurationError
from llm_compat.providers.base import Provider
from llm_compat.report import CompatibilityReport
from llm_compat.runner import RunOptions, run_compatibility
from llm_compat.utils import redact_sensitive, redact_value, truncate, utc_now
from llm_compat.version import __version__


@dataclass(frozen=True)
class ComparisonReport:
    '''Reports from multiple independently configured providers.'''

    started_at: str
    finished_at: str
    duration_ms: float
    entries: tuple[CompatibilityReport, ...]
    schema_version: str = '1.0'

    def to_dict(self) -> dict[str, object]:
        providers = []
        for report in self.entries:
            data = report.to_dict()
            providers.append(
                {
                    'provider': data['provider'],
                    'timing': data['timing'],
                    'summary': data['summary'],
                    'results': data['results'],
                }
            )
        payload = {
            'schema_version': self.schema_version,
            'tool': {'name': 'llm-compat', 'version': __version__},
            'comparison': {
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'duration_ms': round(self.duration_ms, 2),
                'provider_count': len(self.entries),
            },
            'providers': providers,
        }
        secrets = tuple(
            secret for entry in self.entries for secret in entry.secrets
        )
        return redact_value(payload, secrets)


def _duration(duration_ms: float) -> str:
    if duration_ms < 1000.0:
        return f'{duration_ms:.0f} ms'
    return f'{duration_ms / 1000.0:.2f} s'


def render_comparison_text(report: ComparisonReport, *, color: bool = False) -> str:
    '''Render a compact comparison table followed by non-passing checks.'''
    lines = [
        f'LLM Compat {__version__} - provider comparison',
        f'Providers: {len(report.entries)}  Duration: {_duration(report.duration_ms)}',
        '',
        'Providers',
    ]
    secrets = tuple(
        secret for entry in report.entries for secret in entry.secrets
    )
    rows = []
    for entry in report.entries:
        name = truncate(redact_sensitive(entry.provider_name, secrets), 24)
        model = truncate(redact_sensitive(entry.model or '(none)', secrets), 28)
        counts = entry.summary.counts
        rows.append(
            (
                name,
                model,
                entry.summary.verdict.upper(),
                f'{entry.summary.score:.1f}%',
                str(counts.get('pass', 0)),
                str(counts.get('fail', 0)),
                str(counts.get('warn', 0)),
                str(counts.get('error', 0)),
                str(counts.get('skip', 0)),
            )
        )
    headers = ('NAME', 'MODEL', 'VERDICT', 'SCORE', 'PASS', 'FAIL', 'WARN', 'ERR', 'SKIP')
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines.append('  ' + '  '.join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rows:
        lines.append('  ' + '  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))
    issues = []
    for entry in report.entries:
        name = truncate(redact_sensitive(entry.provider_name, secrets), 40)
        for result in entry.results:
            if result.status.value == 'pass':
                continue
            message = truncate(redact_sensitive(result.message, secrets), 200)
            kind = f' [{result.failure_kind.value}]' if result.failure_kind else ''
            issues.append(f'  {name}: {result.id} {result.status.label}{kind} - {message}')
    if issues:
        lines.extend(['', 'Issues'] + issues)
    return '\n'.join(lines)


def compare_providers(
    configs: Sequence[ProviderConfig],
    options: Optional[RunOptions] = None,
    *,
    providers: Optional[Sequence[Provider]] = None,
    checks: Optional[Sequence[CompatibilityCheck]] = None,
) -> ComparisonReport:
    '''Run the same checks against multiple independent provider configurations.'''
    options = options or RunOptions()
    if options.authorize is not True:
        raise AuthorizationRequiredError()
    if not configs:
        raise ConfigurationError('At least one provider configuration is required.')
    if any(not isinstance(config, ProviderConfig) for config in configs):
        raise ConfigurationError('Every comparison target must be a ProviderConfig.')
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ConfigurationError('Comparison target names must be unique.')
    if providers is not None and len(providers) != len(configs):
        raise ConfigurationError('The providers sequence must match the configs sequence.')
    selected = (
        list(checks)
        if checks is not None
        else select_checks(
            options.only,
            options.skip,
            include_optional=options.all_checks,
        )
    )
    started_at = utc_now()
    started = time.perf_counter()
    entries = []
    for index, config in enumerate(configs):
        active_provider = providers[index] if providers is not None else None
        entries.append(
            run_compatibility(
                config,
                options,
                provider=active_provider,
                checks=selected,
            )
        )
    duration_ms = (time.perf_counter() - started) * 1000.0
    return ComparisonReport(
        started_at=started_at,
        finished_at=utc_now(),
        duration_ms=duration_ms,
        entries=tuple(entries),
    )