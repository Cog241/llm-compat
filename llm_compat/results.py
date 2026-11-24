'''Result and summary data structures shared by checks, reports, and the CLI.'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class Status(str, Enum):
    '''Outcome of one compatibility check.'''

    PASS = 'pass'
    FAIL = 'fail'
    WARN = 'warn'
    ERROR = 'error'
    SKIP = 'skip'

    @property
    def label(self) -> str:
        return self.value.upper()

    @property
    def is_failure(self) -> bool:
        return self in {Status.FAIL, Status.ERROR}


@dataclass(frozen=True)
class ErrorInfo:
    '''Machine-readable details for a check that could not complete cleanly.'''

    type: str
    kind: str
    message: str
    status_code: Optional[int] = None
    retryable: bool = False
    hint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': self.type,
            'kind': self.kind,
            'message': self.message,
            'status_code': self.status_code,
            'retryable': self.retryable,
            'hint': self.hint,
        }


@dataclass(frozen=True)
class CheckResult:
    '''The complete outcome of one compatibility check.'''

    id: str
    name: str
    category: str
    status: Status
    message: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorInfo] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'status': self.status.value,
            'message': self.message,
            'duration_ms': round(self.duration_ms, 2),
            'details': self.details,
            'error': self.error.to_dict() if self.error else None,
        }


@dataclass(frozen=True)
class Summary:
    '''Aggregate compatibility result used by terminal and JSON reports.'''

    counts: Mapping[str, int]
    total: int
    executed: int
    score: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'verdict': self.verdict,
            'score': self.score,
            'total_checks': self.total,
            'executed_checks': self.executed,
            'counts': dict(self.counts),
        }


def summarize(results: list[CheckResult] | tuple[CheckResult, ...]) -> Summary:
    '''Count statuses and derive a pass rate and verdict.'''
    counts = {status.value: 0 for status in Status}
    for result in results:
        counts[result.status.value] += 1
    total = len(results)
    skipped = counts[Status.SKIP.value]
    executed = total - skipped
    passed = counts[Status.PASS.value]
    failed = counts[Status.FAIL.value] + counts[Status.ERROR.value]
    warnings = counts[Status.WARN.value]
    score = round((passed / executed) * 100.0, 1) if executed else 0.0
    if executed == 0:
        verdict = 'inconclusive'
    elif failed == 0 and warnings == 0 and skipped == 0 and passed == executed:
        verdict = 'compatible'
    elif passed == 0 and failed > 0:
        verdict = 'incompatible'
    else:
        verdict = 'partial'
    return Summary(
        counts=counts,
        total=total,
        executed=executed,
        score=score,
        verdict=verdict,
    )
