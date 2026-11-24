'''Run orchestration, including the explicit live-request authorization gate.'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from llm_compat.checks.base import CheckContext, CompatibilityCheck
from llm_compat.checks.registry import select_checks
from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.errors import AuthorizationRequiredError, LLMCompatError
from llm_compat.providers.base import Provider, select_chat_model
from llm_compat.providers.registry import build_provider
from llm_compat.report import CompatibilityReport
from llm_compat.results import summarize
from llm_compat.utils import utc_now


@dataclass(frozen=True)
class RunOptions:
    '''User-controlled run settings. Authorization defaults to denied.'''

    authorize: bool = False
    samples: int = 3
    max_tokens: int = 128
    only: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    all_checks: bool = False

    def settings(self) -> RunSettings:
        return RunSettings(samples=self.samples, max_tokens=self.max_tokens)


def run_compatibility(
    config: ProviderConfig,
    options: Optional[RunOptions] = None,
    *,
    provider: Optional[Provider] = None,
    checks: Optional[Sequence[CompatibilityCheck]] = None,
) -> CompatibilityReport:
    '''Run checks against one endpoint.

    Live requests require ``options.authorize=True``. Callers should obtain
    that authorization from the user before invoking this function.
    '''
    options = options or RunOptions()
    if options.authorize is not True:
        raise AuthorizationRequiredError()
    settings = options.settings()
    selected = (
        list(checks)
        if checks is not None
        else select_checks(
            options.only,
            options.skip,
            include_optional=options.all_checks,
        )
    )
    active_provider = provider or build_provider(config)

    started_at = utc_now()
    started = time.perf_counter()
    model = config.model
    model_source = 'explicit' if model else 'none'
    if model is None:
        try:
            discovered = active_provider.list_models()
        except LLMCompatError:
            discovered = []
        model = select_chat_model(discovered)
        if model is not None:
            model_source = 'auto'
    context = CheckContext(provider=active_provider, model=model, settings=settings)
    results = tuple(check.execute(context) for check in selected)
    duration_ms = (time.perf_counter() - started) * 1000.0
    finished_at = utc_now()
    return CompatibilityReport(
        provider_name=active_provider.name,
        base_url=active_provider.base_url,
        model=model,
        model_source=model_source,
        auth_configured=active_provider.auth_configured,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        results=results,
        summary=summarize(results),
        secrets=active_provider.config.sensitive_values,
    )
