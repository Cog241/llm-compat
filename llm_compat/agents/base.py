'''Agent compatibility profiles, evaluation, and report structures.'''

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from llm_compat.report import CompatibilityReport
from llm_compat.results import CheckResult, FailureKind, Status
from llm_compat.utils import redact_sensitive, truncate, utc_now
from llm_compat.version import __version__


@dataclass(frozen=True)
class AgentCapability:
    '''One protocol capability an agent needs or may use.'''

    key: str
    name: str
    check_id: Optional[str]
    required: bool
    description: str
    suggestion: str
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'name': self.name,
            'check_id': self.check_id,
            'required': self.required,
            'description': self.description,
            'suggestion': self.suggestion,
            'unavailable_reason': self.unavailable_reason,
        }


@dataclass(frozen=True)
class AgentProfile:
    '''Documented protocol requirements for one coding agent.'''

    key: str
    name: str
    summary: str
    docs_url: str
    protocol: str
    capabilities: tuple[AgentCapability, ...]
    notes: tuple[str, ...] = ()
    verified_by: str = 'inferred protocol tests only; the agent itself is not run'

    @property
    def check_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                capability.check_id
                for capability in self.capabilities
                if capability.check_id is not None
            )
        )

    @property
    def required_capabilities(self) -> tuple[AgentCapability, ...]:
        return tuple(capability for capability in self.capabilities if capability.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            'key': self.key,
            'name': self.name,
            'summary': self.summary,
            'docs_url': self.docs_url,
            'protocol': self.protocol,
            'capabilities': [capability.to_dict() for capability in self.capabilities],
            'notes': list(self.notes),
            'verified_by': self.verified_by,
        }


@dataclass(frozen=True)
class AgentExecution:
    '''A user-authorized command that actually ran a coding agent.'''

    command: str
    exit_code: Optional[int]
    timed_out: bool
    duration_ms: float
    output_preview: str

    @property
    def status(self) -> str:
        if self.timed_out:
            return 'timed_out'
        return 'passed' if self.exit_code == 0 else 'failed'

    def to_dict(self) -> dict[str, Any]:
        return {
            'command': self.command,
            'status': self.status,
            'exit_code': self.exit_code,
            'timed_out': self.timed_out,
            'duration_ms': round(self.duration_ms, 2),
            'output_preview': self.output_preview,
        }


@dataclass(frozen=True)
class CapabilityResult:
    '''Evaluation of one documented capability against protocol checks.'''

    capability: AgentCapability
    status: Status
    message: str
    failure_kind: Optional[FailureKind] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'key': self.capability.key,
            'name': self.capability.name,
            'check_id': self.capability.check_id,
            'required': self.capability.required,
            'status': self.status.value,
            'failure_kind': self.failure_kind.value if self.failure_kind else None,
            'message': self.message,
            'suggestion': self.suggestion,
            'description': self.capability.description,
        }


_KIND_ACTION = {
    FailureKind.UNSUPPORTED: (
        'Confirm that the endpoint implements this OpenAI-compatible feature, or route '
        'the agent through a compatible gateway.'
    ),
    FailureKind.MALFORMED: (
        'Inspect the raw response and endpoint version; the provider returned data that '
        'did not follow the expected OpenAI schema.'
    ),
    FailureKind.MODEL_BEHAVIOR: (
        'Try a stronger or more instruction-following model; the transport worked but '
        'the model did not follow the requested protocol.'
    ),
    FailureKind.AUTHENTICATION: (
        'Check the API-key environment variable, account permissions, and endpoint '
        'authentication requirements.'
    ),
    FailureKind.TEMPORARY: (
        'Retry later and check provider status; this may be a transient network or '
        'rate-limit problem.'
    ),
    FailureKind.CONFIGURATION: (
        'Fix the endpoint URL, model id, timeout, or authorization configuration before '
        'running the agent.'
    ),
    FailureKind.UNKNOWN: (
        'Inspect the underlying check error and run the actual agent to reproduce the '
        'failure with its own logs.'
    ),
}


def _suggestion_for(capability: AgentCapability, result: CheckResult) -> Optional[str]:
    if result.status == Status.PASS:
        return None
    pieces = [capability.suggestion]
    if result.failure_kind is not None:
        pieces.append(_KIND_ACTION[result.failure_kind])
    if result.status == Status.SKIP:
        pieces.append(
            'Provide a model or enable the required check so this capability can be evaluated.'
        )
    return ' '.join(piece for piece in pieces if piece)


def _capability_result(
    profile: AgentProfile,
    capability: AgentCapability,
    by_id: dict[str, CheckResult],
) -> CapabilityResult:
    if capability.check_id is None:
        reason = capability.unavailable_reason or 'This capability is not implemented by LLM Compat.'
        return CapabilityResult(
            capability,
            Status.SKIP,
            f'Not evaluated: {reason}',
            suggestion=capability.suggestion,
        )
    result = by_id.get(capability.check_id)
    if result is None:
        return CapabilityResult(
            capability,
            Status.SKIP,
            f'The {capability.check_id!r} check was not run, so this capability is unverified.',
            suggestion='Run the complete agent profile so every documented capability is exercised.',
        )
    return CapabilityResult(
        capability,
        result.status,
        result.message,
        result.failure_kind,
        _suggestion_for(capability, result),
    )


def _verdict(
    results: Sequence[CapabilityResult],
    execution: Optional[AgentExecution],
) -> str:
    if execution is not None and execution.status != 'passed':
        return 'incompatible'
    required = [item for item in results if item.capability.required]
    tested = [item for item in required if item.capability.check_id is not None]
    passed = [item for item in tested if item.status == Status.PASS]
    failed = [item for item in tested if item.status in {Status.FAIL, Status.ERROR}]
    warned = [item for item in tested if item.status == Status.WARN]
    skipped = [item for item in tested if item.status == Status.SKIP]
    optional_issues = [
        item
        for item in results
        if not item.capability.required
        and item.capability.check_id is not None
        and item.status in {Status.FAIL, Status.ERROR, Status.WARN}
    ]
    if not tested or (not passed and not failed and not warned):
        return 'inconclusive'
    if failed and not passed:
        return 'incompatible'
    if failed or warned or skipped or optional_issues:
        return 'partial'
    return 'compatible'


@dataclass(frozen=True)
class AgentReport:
    '''Protocol inference plus any explicit actual-agent execution.'''

    profile: AgentProfile
    protocol_report: CompatibilityReport
    capabilities: tuple[CapabilityResult, ...]
    verdict: str
    execution: Optional[AgentExecution] = None
    generated_at: str = field(default_factory=utc_now)

    @property
    def verification_method(self) -> str:
        if self.execution is not None:
            return 'agent_execution'
        if self.protocol_report.results:
            return 'inferred_protocol'
        return 'not_run'

    @property
    def agent_execution_verified(self) -> bool:
        return self.execution is not None and self.execution.status == 'passed'

    def to_dict(self) -> dict[str, Any]:
        protocol = self.protocol_report.to_dict()
        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'agent': self.profile.to_dict(),
            'provider': protocol['provider'],
            'verdict': self.verdict,
            'verification': {
                'method': self.verification_method,
                'agent_execution_verified': self.agent_execution_verified,
                'agent_execution': self.execution.to_dict() if self.execution else None,
                'note': (
                    self.profile.verified_by
                    if self.execution is None
                    else (
                        'An explicitly authorized actual-agent command ran; protocol checks '
                        'are reported separately.'
                    )
                ),
            },
            'capabilities': [item.to_dict() for item in self.capabilities],
            'protocol_report': protocol,
            'generated_at': self.generated_at,
        }
        from llm_compat.utils import redact_value

        return redact_value(payload, self.protocol_report.secrets)


def evaluate_agent(
    profile: AgentProfile,
    report: CompatibilityReport,
    *,
    execution: Optional[AgentExecution] = None,
) -> AgentReport:
    '''Map protocol check results to documented agent capabilities.'''
    by_id = {result.id: result for result in report.results}
    capabilities = tuple(
        _capability_result(profile, capability, by_id)
        for capability in profile.capabilities
    )
    return AgentReport(
        profile=profile,
        protocol_report=report,
        capabilities=capabilities,
        verdict=_verdict(capabilities, execution),
        execution=execution,
    )


_COLORS = {
    Status.PASS: '\x1b[32m',
    Status.FAIL: '\x1b[31m',
    Status.WARN: '\x1b[33m',
    Status.ERROR: '\x1b[31m',
    Status.SKIP: '\x1b[2m',
}
_RESET = '\x1b[0m'


def _paint(text: str, status: Status, enabled: bool) -> str:
    if not enabled:
        return text
    return f'{_COLORS[status]}{text}{_RESET}'


def render_agent_text(report: AgentReport, *, color: bool = False) -> str:
    '''Render an agent-oriented report with actionable next steps.'''
    profile = report.profile
    protocol = report.protocol_report
    secrets = protocol.secrets
    lines = [f'{profile.name} Compatibility Report']
    if protocol.model:
        lines.append(f'Model: {truncate(redact_sensitive(protocol.model, secrets), 200)}')
    else:
        lines.append('Model: (not selected)')
    lines.append(f'Protocol: {profile.protocol}')
    lines.append('')
    for item in report.capabilities:
        label = f'{item.status.label:<5}'
        name = item.capability.name
        requirement = 'required' if item.capability.required else 'optional'
        lines.append(f'  {_paint(label, item.status, color)}  {name:<28}  {requirement}')
        if item.status != Status.PASS:
            message = redact_sensitive(item.message, secrets)
            for wrapped in textwrap.wrap(message, width=88) or ['']:
                lines.append(f'          {wrapped}')
            if item.suggestion:
                suggestion = redact_sensitive(item.suggestion, secrets)
                for wrapped in textwrap.wrap(f'Fix: {suggestion}', width=88):
                    lines.append(f'          {wrapped}')
    lines.append('')
    lines.append(f'Verdict: {report.verdict.upper()}')
    if report.execution is None:
        if protocol.results:
            lines.append(
                'Verification: inferred from protocol tests. The actual agent has not been run.'
            )
        else:
            lines.append(
                'Verification: no protocol checks were run; the actual agent has not been run.'
            )
    else:
        status = report.execution.status
        lines.append(f'Verification: actual agent command {status}.')
        if report.execution.output_preview:
            output = redact_sensitive(report.execution.output_preview, secrets)
            lines.append(f'Agent output: {truncate(output, 2000)}')
    blocked = [item for item in report.capabilities if item.status != Status.PASS]
    if blocked:
        lines.append('')
        lines.append('Actionable observations:')
        for item in blocked:
            suggestion = item.suggestion or item.message
            suggestion = redact_sensitive(suggestion, secrets)
            lines.append(f'  - {item.capability.name}: {suggestion}')
    if profile.notes:
        lines.append('')
        lines.append('Profile notes:')
        for note in profile.notes:
            lines.append(f'  - {note}')
    return '\n'.join(lines)
