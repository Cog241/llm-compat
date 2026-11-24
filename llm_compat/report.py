'''JSON exports and human-readable terminal reports.'''

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from llm_compat.results import CheckResult, Status, Summary
from llm_compat.utils import redact_sensitive, redact_value, safe_url, truncate
from llm_compat.version import __version__

_COLORS = {
    Status.PASS: '\x1b[32m',
    Status.FAIL: '\x1b[31m',
    Status.WARN: '\x1b[33m',
    Status.ERROR: '\x1b[31m',
    Status.SKIP: '\x1b[2m',
}
_RESET = '\x1b[0m'


@dataclass(frozen=True)
class CompatibilityReport:
    '''A complete run result that can be rendered or exported.'''

    provider_name: str
    base_url: str
    model: Optional[str]
    model_source: str
    auth_configured: bool
    started_at: str
    finished_at: str
    duration_ms: float
    results: tuple[CheckResult, ...]
    summary: Summary
    schema_version: str = '1.0'
    secrets: tuple[str, ...] = field(default=(), repr=False)

    @property
    def display_base_url(self) -> str:
        return safe_url(self.base_url)

    def to_dict(self) -> dict[str, object]:
        '''Return a JSON-safe export with credentials redacted centrally.'''
        payload = {
            'schema_version': self.schema_version,
            'tool': {'name': 'llm-compat', 'version': __version__},
            'provider': {
                'name': self.provider_name,
                'base_url': self.display_base_url,
                'model': self.model,
                'model_source': self.model_source,
                'auth_configured': self.auth_configured,
            },
            'timing': {
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'duration_ms': round(self.duration_ms, 2),
            },
            'summary': self.summary.to_dict(),
            'results': [result.to_dict() for result in self.results],
        }
        return redact_value(payload, self.secrets)


def _format_duration(duration_ms: float) -> str:
    if duration_ms < 1000.0:
        return f'{duration_ms:.0f} ms'
    return f'{duration_ms / 1000.0:.2f} s'


def _paint(text: str, status: Status, enabled: bool) -> str:
    if not enabled:
        return text
    return f'{_COLORS[status]}{text}{_RESET}'


def render_text(report: CompatibilityReport, *, color: bool = False) -> str:
    '''Render a compact, human-readable report for a terminal.'''
    secrets = report.secrets
    provider_name = truncate(redact_sensitive(report.provider_name, secrets), 120)
    base_url = truncate(redact_sensitive(report.display_base_url, secrets), 300)
    lines = [
        f'LLM Compat {__version__}',
        f'Provider: {provider_name}  {base_url}',
    ]
    if report.model:
        suffix = ' (auto-selected)' if report.model_source == 'auto' else ''
        lines.append(f'Model: {truncate(redact_sensitive(report.model, secrets), 200)}{suffix}')
    else:
        lines.append('Model: (not selected)')
    lines.append(f'API key: {"configured" if report.auth_configured else "not configured"}')
    lines.append(f'Duration: {_format_duration(report.duration_ms)}')
    lines.append('')
    lines.append('Checks')

    names = [truncate(redact_sensitive(item.name, secrets), 40) for item in report.results]
    name_width = max(24, min(40, max((len(name) for name in names), default=0)))
    for item, name in zip(report.results, names):
        label = f'{item.status.label:<5}'
        rendered_label = _paint(label, item.status, color)
        lines.append(
            f'  {rendered_label}  {name:<{name_width}}  '
            f'{_format_duration(item.duration_ms):>9}'
        )
        indent = ' ' * (11 + name_width)
        available = max(40, 100 - len(indent))
        message = truncate(redact_sensitive(item.message, secrets), 1000)
        for wrapped in textwrap.wrap(message, width=available) or ['']:
            lines.append(f'{indent}{wrapped}')
        if item.failure_kind is not None:
            lines.append(f'{indent}Failure class: {item.failure_kind.value}')
        if item.error and item.error.hint:
            hint = truncate(redact_sensitive(item.error.hint, secrets), 500)
            lines.append(f'{indent}Hint: {hint}')

    lines.append('')
    lines.append('Summary')
    lines.append(f'  Verdict: {report.summary.verdict.upper()}')
    if report.summary.executed:
        lines.append(f'  Compatibility score: {report.summary.score:.1f}%')
    else:
        lines.append('  Compatibility score: not scored (no checks executed)')
    counts = report.summary.counts
    warnings = counts.get('warn', 0)
    errors = counts.get('error', 0)
    lines.append(
        '  Checks: '
        f"{counts.get('pass', 0)} passed, "
        f"{counts.get('fail', 0)} failed, "
        f"{warnings} warning{'s' if warnings != 1 else ''}, "
        f"{errors} error{'s' if errors != 1 else ''}, "
        f"{counts.get('skip', 0)} skipped"
    )
    if report.summary.failure_kinds:
        classes = ', '.join(
            f'{name}={count}' for name, count in sorted(report.summary.failure_kinds.items())
        )
        lines.append(f'  Failure classes: {classes}')
    return '\n'.join(lines)


def render_json(report: CompatibilityReport) -> str:
    '''Render a stable, indented JSON document for automation.'''
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
