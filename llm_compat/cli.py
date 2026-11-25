'''Command-line interface for LLM Compat.'''

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from llm_compat.agents import (
    AgentExecution,
    evaluate_agent,
    get_profile,
    list_profiles,
    render_agent_text,
)
from llm_compat.checks.base import CompatibilityCheck
from llm_compat.checks.registry import available_checks, select_checks
from llm_compat.compare import compare_providers, render_comparison_text
from llm_compat.config import ProviderConfig, RunSettings
from llm_compat.errors import ConfigurationError, LLMCompatError
from llm_compat.providers.registry import build_provider, get_preset, list_presets
from llm_compat.report import CompatibilityReport, render_json, render_text
from llm_compat.research import load_experiment, run_experiment
from llm_compat.results import summarize
from llm_compat.runner import RunOptions, run_compatibility
from llm_compat.utils import redact_sensitive, redact_value, safe_url, truncate, utc_now
from llm_compat.version import __version__


def _add_output_options(parser: argparse.ArgumentParser, *, color: bool = False) -> None:
    parser.add_argument(
        '--format',
        choices=('text', 'json'),
        default='text',
        help='Output format (default: text).',
    )
    parser.add_argument(
        '--output',
        metavar='PATH',
        help='Write output to a file instead of stdout.',
    )
    if color:
        parser.add_argument(
            '--no-color',
            action='store_true',
            help='Disable ANSI colors in the terminal report.',
        )


def _add_connection_options(
    parser: argparse.ArgumentParser,
    *,
    include_model: bool = True,
) -> None:
    preset_keys = [preset.key for preset in list_presets()]
    parser.add_argument(
        '--provider',
        default='openai',
        choices=[*preset_keys, 'custom'],
        help='Provider preset, or custom for any other OpenAI-compatible endpoint.',
    )
    parser.add_argument(
        '--base-url',
        metavar='URL',
        help='Override the preset base URL, or provide it with --provider custom.',
    )
    if include_model:
        parser.add_argument(
            '--model',
            metavar='MODEL',
            help='Model id to test. If omitted, a chat model is selected from discovery.',
        )
    parser.add_argument(
        '--api-key-env',
        metavar='ENV_VAR',
        help='Environment variable containing the API key. Keys are never accepted as CLI values.',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        metavar='SECONDS',
        help='Per-request timeout in seconds (default: 30).',
    )
    parser.add_argument(
        '--allow-insecure-http',
        action='store_true',
        help=(
            'Allow HTTP for a remote authenticated endpoint. Use only in an authorized '
            'development environment; credentials otherwise require HTTPS.'
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='llm-compat',
        description=(
            'Test whether an OpenAI-compatible API provider supports the features '
            'your applications and coding agents depend on.'
        ),
        epilog=(
            'Live commands require explicit authorization because requests may cost money. '
            'Run llm-compat checks to see the full test catalog.'
        ),
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser(
        'run',
        help='Run compatibility checks against an endpoint.',
        description='Run compatibility checks and produce a terminal or JSON report.',
    )
    _add_connection_options(run_parser)
    run_parser.add_argument(
        '--samples',
        type=int,
        default=3,
        metavar='N',
        help='Number of performance samples (default: 3, max: 20; p95 requires 20).',
    )
    run_parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        metavar='N',
        help='Maximum completion tokens per request (default: 128).',
    )
    run_parser.add_argument(
        '--only',
        action='append',
        metavar='ID',
        help='Run only this check id. May be repeated.',
    )
    run_parser.add_argument(
        '--skip',
        action='append',
        metavar='ID',
        help='Skip this check id. May be repeated.',
    )
    run_parser.add_argument(
        '--all',
        dest='all_checks',
        action='store_true',
        help='Include opt-in checks for advanced tool, schema, system, and reasoning features.',
    )
    run_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the plan without making network requests.',
    )
    run_parser.add_argument(
        '--yes',
        action='store_true',
        help='Authorize live requests without an interactive confirmation.',
    )
    _add_output_options(run_parser, color=True)

    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple independent provider configurations.',
        description=(
            'Run the same checks against providers configured in a JSON file. '
            'Each target has its own endpoint, model, and API-key environment variable.'
        ),
    )
    compare_parser.add_argument(
        '--config',
        required=True,
        metavar='PATH',
        help='JSON file containing the comparison targets. API key literals are rejected.',
    )
    compare_parser.add_argument(
        '--samples',
        type=int,
        default=3,
        metavar='N',
        help='Number of performance samples (default: 3, max: 20; p95 requires 20).',
    )
    compare_parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        metavar='N',
        help='Maximum completion tokens per request (default: 128).',
    )
    compare_parser.add_argument(
        '--only',
        action='append',
        metavar='ID',
        help='Run only this check id. May be repeated.',
    )
    compare_parser.add_argument(
        '--skip',
        action='append',
        metavar='ID',
        help='Skip this check id. May be repeated.',
    )
    compare_parser.add_argument(
        '--all',
        dest='all_checks',
        action='store_true',
        help='Include opt-in checks for advanced tool, schema, system, and reasoning features.',
    )
    compare_parser.add_argument(
        '--allow-insecure-http',
        action='store_true',
        help=(
            'Allow HTTP for remote authenticated targets. Use only in an authorized '
            'development environment; credentials otherwise require HTTPS.'
        ),
    )
    compare_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the comparison plan without making network requests.',
    )
    compare_parser.add_argument(
        '--yes',
        action='store_true',
        help='Authorize live requests without an interactive confirmation.',
    )
    _add_output_options(compare_parser, color=True)

    models_parser = subparsers.add_parser(
        'models',
        help='Discover models advertised by an endpoint.',
        description='Fetch and print the model list from GET /models.',
    )
    _add_connection_options(models_parser, include_model=False)
    models_parser.add_argument(
        '--yes',
        action='store_true',
        help='Authorize the live request without an interactive confirmation.',
    )
    _add_output_options(models_parser)

    checks_parser = subparsers.add_parser(
        'checks',
        help='List built-in compatibility checks.',
        description='List check ids, descriptions, and model requirements.',
    )
    _add_output_options(checks_parser)

    providers_parser = subparsers.add_parser(
        'providers',
        help='List built-in provider presets.',
        description='List connection presets and their API key environment variables.',
    )
    _add_output_options(providers_parser)

    agent_parser = subparsers.add_parser(
        'agent',
        help='Inspect and test coding-agent compatibility profiles.',
        description=(
            'Evaluate documented agent protocol requirements. Protocol results are '
            'inferred from checks; an actual agent is never launched unless an explicit '
            '--verify-command is supplied.'
        ),
    )
    agent_subparsers = agent_parser.add_subparsers(dest='agent_command', required=True)
    agent_list_parser = agent_subparsers.add_parser(
        'list',
        help='List built-in agent profiles and their required capabilities.',
    )
    _add_output_options(agent_list_parser)
    agent_test_parser = agent_subparsers.add_parser(
        'test',
        help='Run protocol checks and map them to one agent profile.',
    )
    agent_test_parser.add_argument(
        '--agent',
        required=True,
        choices=[
            *[profile.key for profile in list_profiles()],
            'roo',
            'roocode',
            'open-code',
        ],
        help='Agent profile to evaluate (aliases: roo, roocode, open-code).',
    )
    _add_connection_options(agent_test_parser)
    agent_test_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the profile plan without making network requests.',
    )
    agent_test_parser.add_argument(
        '--yes',
        action='store_true',
        help='Authorize live protocol requests and any verify command without prompting.',
    )
    agent_test_parser.add_argument(
        '--verify-command',
        metavar='COMMAND',
        help=(
            'Optional shell command that actually runs the agent after protocol tests. '
            'Requires --yes. The command and output are sanitized before reporting.'
        ),
    )
    agent_test_parser.add_argument(
        '--verify-timeout',
        type=float,
        default=300.0,
        metavar='SECONDS',
        help='Maximum time for --verify-command (default: 300).',
    )
    _add_output_options(agent_test_parser, color=True)

    research_parser = subparsers.add_parser(
        'research',
        help='Validate or run reproducible provider/model experiments.',
        description='Run repeated, budgeted trials and export sanitized research datasets.',
    )
    research_subparsers = research_parser.add_subparsers(
        dest='research_command', required=True
    )
    research_validate_parser = research_subparsers.add_parser(
        'validate',
        help='Validate an experiment config and print its plan without network requests.',
    )
    research_validate_parser.add_argument('config', metavar='CONFIG', help='Experiment JSON file.')
    _add_output_options(research_validate_parser)
    research_run_parser = research_subparsers.add_parser(
        'run',
        help='Run an authorized experiment and write sanitized outputs.',
    )
    research_run_parser.add_argument('config', metavar='CONFIG', help='Experiment JSON file.')
    research_run_parser.add_argument(
        '--output-dir',
        metavar='PATH',
        help='Override output_dir from the experiment config.',
    )
    research_run_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the experiment plan without making network requests.',
    )
    research_run_parser.add_argument(
        '--yes',
        action='store_true',
        help='Authorize live requests without an interactive confirmation.',
    )
    _add_output_options(research_run_parser)

    return parser

def _build_config(args: argparse.Namespace) -> ProviderConfig:
    preset = get_preset(args.provider)
    if args.provider == 'custom':
        if not args.base_url:
            raise ConfigurationError('--base-url is required when --provider custom is used.')
        base_url = args.base_url
    else:
        base_url = args.base_url or (preset.base_url if preset else '')
        if not base_url:
            raise ConfigurationError(f'No base URL is configured for provider {args.provider!r}.')
    api_key: Optional[str] = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                f'Environment variable {args.api_key_env} is not set or is empty.'
            )
    else:
        if preset and preset.api_key_env:
            api_key = os.environ.get(preset.api_key_env)
        if not api_key and args.provider == 'custom':
            api_key = os.environ.get('LLM_COMPAT_API_KEY')
    return ProviderConfig(
        name=args.provider,
        base_url=base_url,
        api_key=api_key,
        model=getattr(args, 'model', None),
        timeout=args.timeout,
        allow_insecure_http=bool(getattr(args, 'allow_insecure_http', False)),
    )


def _estimated_requests(
    checks: Sequence[CompatibilityCheck],
    samples: int,
    *,
    model_explicit: bool,
) -> int:
    total = 0
    model_checks = {'connectivity', 'auth', 'models'}
    if any(check.id in model_checks for check in checks) or not model_explicit:
        total += 1
    single_request = {
        'chat',
        'streaming',
        'json',
        'system',
        'tools_forced',
        'tools_parallel',
        'tools_stream',
        'json_schema',
        'reasoning',
    }
    for check in checks:
        if check.id == 'latency':
            total += samples
        elif check.id == 'tools':
            total += 2
        elif check.id in single_request:
            total += 1
    return total


def _stdin_is_interactive() -> bool:
    isatty = getattr(sys.stdin, 'isatty', None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _authorize_targets(
    args: argparse.Namespace,
    targets: Sequence[tuple[ProviderConfig, int]],
    *,
    purpose: str,
) -> bool:
    if args.yes:
        return True
    if args.format == 'json' or not _stdin_is_interactive():
        print(
            'error: live API requests require explicit authorization. '
            'Re-run with --yes after reviewing the endpoint and cost implications.',
            file=sys.stderr,
        )
        return False
    total = sum(count for _, count in targets)
    secrets = tuple(
        secret for config, _ in targets for secret in config.sensitive_values
    )
    print('', file=sys.stderr)
    print(
        'LLM Compat will send live API requests that may incur provider charges.',
        file=sys.stderr,
    )
    print(
        f'Purpose: {truncate(redact_sensitive(purpose, secrets), 500)}',
        file=sys.stderr,
    )
    print(f'Estimated requests: about {total}', file=sys.stderr)
    for config, count in targets:
        print(
            f'Target: {truncate(redact_sensitive(config.name, secrets), 80)}  '
            f'{truncate(redact_sensitive(safe_url(config.base_url), secrets), 300)}  '
            f'up to {count} request(s)',
            file=sys.stderr,
        )
        credentials = (
            'configured via environment' if config.auth_configured else 'not configured'
        )
        print(f'  Credentials: {credentials}', file=sys.stderr)
        if not config.auth_configured:
            print(
                '  Warning: no API key is configured. This is expected for local endpoints '
                'that do not require authentication.',
                file=sys.stderr,
            )
    print("Type 'yes' to continue: ", end='', file=sys.stderr, flush=True)
    try:
        answer = input()
    except (EOFError, OSError):
        answer = ''
    if answer == 'yes':
        return True
    print('Aborted: authorization was not granted.', file=sys.stderr)
    return False


def _authorize(
    args: argparse.Namespace,
    config: ProviderConfig,
    *,
    purpose: str,
    request_count: int,
) -> bool:
    return _authorize_targets(args, ((config, request_count),), purpose=purpose)


def _emit(text: str, args: argparse.Namespace) -> None:
    if args.output:
        path = Path(args.output)
        try:
            path.write_text(text.rstrip('\n') + '\n', encoding='utf-8')
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f'Could not write {path}: {exc}') from exc
        print(f'Wrote output to {path}', file=sys.stderr)
    else:
        print(text)


def _use_color(args: argparse.Namespace) -> bool:
    if args.no_color or args.output or os.environ.get('NO_COLOR') is not None:
        return False
    isatty = getattr(sys.stdout, 'isatty', None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _dry_run_payload(
    config: ProviderConfig,
    checks: Sequence[CompatibilityCheck],
    settings: RunSettings,
    request_count: int,
) -> dict[str, Any]:
    payload = {
        'schema_version': '1.0',
        'tool': {'name': 'llm-compat', 'version': __version__},
        'dry_run': True,
        'provider': {
            'name': config.name,
            'base_url': safe_url(config.base_url),
            'model': config.model,
            'model_source': 'explicit' if config.model else 'auto',
            'auth_configured': config.auth_configured,
        },
        'settings': {'samples': settings.samples, 'max_tokens': settings.max_tokens},
        'estimated_requests': request_count,
        'checks': [
            {
                'id': check.id,
                'name': check.name,
                'category': check.category,
                'description': check.description,
                'requires_model': check.requires_model,
                'default_enabled': check.default_enabled,
            }
            for check in checks
        ],
    }
    return redact_value(payload, config.sensitive_values)


def _render_dry_run(
    config: ProviderConfig,
    checks: Sequence[CompatibilityCheck],
    settings: RunSettings,
    request_count: int,
) -> str:
    secrets = config.sensitive_values
    model = (
        truncate(redact_sensitive(config.model, secrets), 200)
        if config.model
        else '(auto-select a discovered chat model)'
    )
    lines = [
        f'LLM Compat {__version__}',
        'Dry run: no network requests were made.',
        f'Provider: {truncate(redact_sensitive(config.name, secrets), 120)}  '
        f'{truncate(redact_sensitive(safe_url(config.base_url), secrets), 300)}',
        f'API key: {"configured" if config.auth_configured else "not configured"}',
        f'Model: {model}',
        '',
        f'Checks ({len(checks)})',
    ]
    for check in checks:
        suffix = ' [requires a model]' if check.requires_model else ''
        if not check.default_enabled:
            suffix += ' [opt-in]'
        lines.append(f'  {check.id:<12} {check.name}{suffix}')
    lines.extend(
        [
            '',
            f'Settings: samples={settings.samples}, max_tokens={settings.max_tokens}',
            f'Estimated live requests for a real run: about {request_count}',
        ]
    )
    return '\n'.join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    settings = RunSettings(samples=args.samples, max_tokens=args.max_tokens)
    checks = select_checks(
        tuple(args.only or ()),
        tuple(args.skip or ()),
        include_optional=bool(getattr(args, 'all_checks', False)),
    )
    config = _build_config(args)
    request_count = _estimated_requests(
        checks,
        settings.samples,
        model_explicit=config.model is not None,
    )
    if args.dry_run:
        if args.format == 'json':
            text = json.dumps(
                _dry_run_payload(config, checks, settings, request_count),
                indent=2,
                ensure_ascii=False,
            )
        else:
            text = _render_dry_run(config, checks, settings, request_count)
        _emit(text, args)
        return 0
    purpose = f'{len(checks)} compatibility check(s)'
    if not _authorize(args, config, purpose=purpose, request_count=request_count):
        return 2
    provider = build_provider(config)
    report = run_compatibility(
        config,
        RunOptions(
            authorize=True,
            samples=settings.samples,
            max_tokens=settings.max_tokens,
            all_checks=bool(getattr(args, 'all_checks', False)),
        ),
        provider=provider,
        checks=checks,
    )
    if args.format == 'json':
        text = render_json(report)
    else:
        text = render_text(report, color=_use_color(args))
    _emit(text, args)
    counts = report.summary.counts
    incomplete = counts.get('fail', 0) + counts.get('error', 0) + counts.get('skip', 0)
    if incomplete or report.summary.verdict == 'inconclusive':
        return 1
    return 0


def _load_comparison_configs(
    path: str,
    *,
    allow_insecure_http: bool,
) -> tuple[ProviderConfig, ...]:
    '''Load independent targets from JSON without accepting key literals.'''
    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f'Could not read comparison config {config_path}: {exc}'
        ) from exc
    if len(raw) > 1_000_000:
        raise ConfigurationError('Comparison config is too large (limit: 1 MB).')
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f'Invalid JSON in comparison config at line {exc.lineno}, column {exc.colno}.'
        ) from exc
    except RecursionError as exc:
        raise ConfigurationError('Comparison config JSON is nested too deeply.') from exc
    if isinstance(document, dict):
        unknown_top = sorted(set(document) - {'providers'})
        if unknown_top:
            raise ConfigurationError(
                f'Unknown comparison config field(s): {", ".join(unknown_top)}.'
            )
        entries = document.get('providers')
    else:
        entries = document
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError('Comparison config must contain a non-empty providers list.')
    configs = []
    allowed = {'name', 'provider', 'base_url', 'model', 'api_key_env', 'timeout'}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ConfigurationError(f'Comparison target {index} must be a JSON object.')
        unknown = sorted(set(entry) - allowed)
        if unknown:
            if 'api_key' in unknown:
                raise ConfigurationError(
                    f'Comparison target {index} must not contain an api_key literal; '
                    'use api_key_env instead.'
                )
            raise ConfigurationError(
                f'Unknown field(s) in comparison target {index}: {", ".join(unknown)}.'
            )
        provider_key = entry.get('provider', 'custom')
        if not isinstance(provider_key, str):
            raise ConfigurationError(f'Comparison target {index} provider must be a string.')
        preset = get_preset(provider_key) if provider_key.strip().lower() != 'custom' else None
        if provider_key.strip().lower() != 'custom' and preset is None:
            raise ConfigurationError(
                f'Unknown provider preset in comparison target {index}: {provider_key!r}.'
            )
        base_url = entry.get('base_url')
        if base_url is None and preset is not None:
            base_url = preset.base_url
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigurationError(
                f'Comparison target {index} requires base_url (or a provider preset).'
            )
        name = entry.get('name')
        if name is None:
            name = preset.key if preset is not None else f'target-{index}'
        explicit_env = 'api_key_env' in entry and entry.get('api_key_env') is not None
        key_env = entry.get('api_key_env')
        if key_env is None and preset is not None:
            key_env = preset.api_key_env
        if key_env is None and preset is None:
            key_env = 'LLM_COMPAT_API_KEY'
        if key_env is not None:
            if not isinstance(key_env, str) or not key_env.strip():
                raise ConfigurationError(
                    f'Comparison target {index} api_key_env must be a non-empty string.'
                )
            key_env = key_env.strip()
            api_key = os.environ.get(key_env)
            if explicit_env and (not api_key or not api_key.strip()):
                raise ConfigurationError(
                    f'Environment variable {key_env} is not set for comparison target {index}.'
                )
        else:
            api_key = None
        timeout = entry.get('timeout', 30.0)
        configs.append(
            ProviderConfig(
                base_url=base_url,
                name=name,
                api_key=api_key,
                model=entry.get('model'),
                timeout=timeout,
                allow_insecure_http=allow_insecure_http,
            )
        )
    names = [config.name for config in configs]
    if len(set(names)) != len(names):
        raise ConfigurationError('Comparison target names must be unique.')
    return tuple(configs)


def _comparison_dry_run_payload(
    configs: Sequence[ProviderConfig],
    checks: Sequence[CompatibilityCheck],
    settings: RunSettings,
    request_counts: Sequence[int],
) -> dict[str, Any]:
    payload = {
        'schema_version': '1.0',
        'tool': {'name': 'llm-compat', 'version': __version__},
        'dry_run': True,
        'comparison': {
            'provider_count': len(configs),
            'estimated_requests': sum(request_counts),
        },
        'settings': {'samples': settings.samples, 'max_tokens': settings.max_tokens},
        'providers': [
            {
                'name': config.name,
                'base_url': safe_url(config.base_url),
                'model': config.model,
                'model_source': 'explicit' if config.model else 'auto',
                'auth_configured': config.auth_configured,
                'estimated_requests': request_counts[index],
            }
            for index, config in enumerate(configs)
        ],
        'checks': [
            {
                'id': check.id,
                'name': check.name,
                'category': check.category,
                'description': check.description,
                'requires_model': check.requires_model,
                'default_enabled': check.default_enabled,
            }
            for check in checks
        ],
    }
    secrets = tuple(
        value
        for config in configs
        for value in config.sensitive_values
    )
    return redact_value(payload, secrets)


def _render_comparison_dry_run(
    configs: Sequence[ProviderConfig],
    checks: Sequence[CompatibilityCheck],
    settings: RunSettings,
    request_counts: Sequence[int],
) -> str:
    lines = [
        f'LLM Compat {__version__} - comparison dry run',
        'Dry run: no network requests were made.',
        f'Providers ({len(configs)})',
    ]
    secrets = tuple(
        secret for config in configs for secret in config.sensitive_values
    )
    for index, config in enumerate(configs):
        model = (
            truncate(redact_sensitive(config.model, secrets), 80)
            if config.model
            else '(auto)'
        )
        credentials = 'configured' if config.auth_configured else 'not configured'
        lines.append(
            f'  {truncate(redact_sensitive(config.name, secrets), 80)}  '
            f'{truncate(redact_sensitive(safe_url(config.base_url), secrets), 300)}  '
            f'model={model}  '
            f'credentials={credentials}  requests<={request_counts[index]}'
        )
    lines.extend(['', f'Checks ({len(checks)})'])
    for check in checks:
        suffix = ' [opt-in]' if not check.default_enabled else ''
        lines.append(f'  {check.id:<12} {check.name}{suffix}')
    lines.extend(
        [
            '',
            f'Settings: samples={settings.samples}, max_tokens={settings.max_tokens}',
            f'Estimated live requests for a real run: about {sum(request_counts)}',
        ]
    )
    return '\n'.join(lines)


def _cmd_compare(args: argparse.Namespace) -> int:
    configs = _load_comparison_configs(
        args.config,
        allow_insecure_http=bool(getattr(args, 'allow_insecure_http', False)),
    )
    settings = RunSettings(samples=args.samples, max_tokens=args.max_tokens)
    checks = select_checks(
        tuple(args.only or ()),
        tuple(args.skip or ()),
        include_optional=bool(getattr(args, 'all_checks', False)),
    )
    request_counts = [
        _estimated_requests(
            checks,
            settings.samples,
            model_explicit=config.model is not None,
        )
        for config in configs
    ]
    if args.dry_run:
        if args.format == 'json':
            text = json.dumps(
                _comparison_dry_run_payload(configs, checks, settings, request_counts),
                indent=2,
                ensure_ascii=False,
            )
        else:
            text = _render_comparison_dry_run(configs, checks, settings, request_counts)
        _emit(text, args)
        return 0
    targets = tuple(zip(configs, request_counts))
    if not _authorize_targets(
        args,
        targets,
        purpose=f'comparison of {len(configs)} provider(s)',
    ):
        return 2
    report = compare_providers(
        configs,
        RunOptions(
            authorize=True,
            samples=settings.samples,
            max_tokens=settings.max_tokens,
            all_checks=bool(getattr(args, 'all_checks', False)),
        ),
        checks=checks,
    )
    if args.format == 'json':
        text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = render_comparison_text(report, color=_use_color(args))
    _emit(text, args)
    for entry in report.entries:
        counts = entry.summary.counts
        if (
            counts.get('fail', 0)
            + counts.get('error', 0)
            + counts.get('skip', 0)
            or entry.summary.verdict == 'inconclusive'
        ):
            return 1
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    config = _build_config(args)
    if not _authorize(args, config, purpose='model discovery', request_count=1):
        return 2
    provider = build_provider(config)
    models = provider.list_models()
    if args.format == 'json':
        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'provider': {
                'name': config.name,
                'base_url': safe_url(config.base_url),
                'auth_configured': config.auth_configured,
            },
            'count': len(models),
            'models': [
                {'id': model.id, 'owned_by': model.owned_by, 'created': model.created}
                for model in models
            ],
        }
        payload = redact_value(payload, config.sensitive_values)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        secrets = config.sensitive_values
        lines = [
            f'LLM Compat {__version__}',
            f'Provider: {truncate(redact_sensitive(config.name, secrets), 120)}  '
            f'{truncate(redact_sensitive(safe_url(config.base_url), secrets), 300)}',
            f'Models: {len(models)}',
        ]
        for model in models:
            owner = (
                f'  ({truncate(redact_sensitive(model.owned_by, secrets), 80)})'
                if model.owned_by
                else ''
            )
            model_id = truncate(redact_sensitive(model.id, secrets), 200)
            lines.append(f'  {model_id}{owner}')
        text = '\n'.join(lines)
    _emit(text, args)
    return 0


def _cmd_checks(args: argparse.Namespace) -> int:
    checks = available_checks()
    if args.format == 'json':
        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'checks': [
                {
                    'id': check.id,
                    'name': check.name,
                    'category': check.category,
                    'description': check.description,
                    'requires_model': check.requires_model,
                    'default_enabled': check.default_enabled,
                }
                for check in checks
            ],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        lines = [f'Built-in checks ({len(checks)})']
        for check in checks:
            requirement = ' [requires a model]' if check.requires_model else ''
            if not check.default_enabled:
                requirement += ' [opt-in]'
            lines.append(f'  {check.id:<12} {check.name}{requirement}')
            lines.append(f'               {check.description}')
        text = '\n'.join(lines)
    _emit(text, args)
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    presets = list_presets()
    if args.format == 'json':
        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'providers': [
                {
                    'key': preset.key,
                    'name': preset.display_name,
                    'base_url': preset.base_url,
                    'api_key_env': preset.api_key_env,
                }
                for preset in presets
            ],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        lines = ['Built-in provider presets']
        for preset in presets:
            env = preset.api_key_env or 'no API key'
            lines.append(f'  {preset.key:<12} {preset.base_url:<40} {env}')
        lines.extend(
            [
                '',
                'Any other OpenAI-compatible endpoint:',
                '  llm-compat run --provider custom --base-url URL --api-key-env ENV_VAR',
            ]
        )
        text = '\n'.join(lines)
    _emit(text, args)
    return 0


def _empty_protocol_report(config: ProviderConfig) -> CompatibilityReport:
    now = utc_now()
    return CompatibilityReport(
        provider_name=config.name,
        base_url=config.base_url,
        model=config.model,
        model_source='explicit' if config.model else 'auto',
        auth_configured=config.auth_configured,
        started_at=now,
        finished_at=now,
        duration_ms=0.0,
        results=(),
        summary=summarize(()),
        secrets=config.sensitive_values,
    )


def _cmd_agent_list(args: argparse.Namespace) -> int:
    profiles = list_profiles()
    if args.format == 'json':
        payload = {
            'schema_version': '1.0',
            'tool': {'name': 'llm-compat', 'version': __version__},
            'agents': [profile.to_dict() for profile in profiles],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        lines = [f'Built-in agent profiles ({len(profiles)})']
        for profile in profiles:
            required = [
                capability.name
                for capability in profile.capabilities
                if capability.required
            ]
            optional = [
                capability.name
                for capability in profile.capabilities
                if not capability.required
            ]
            lines.append(f'  {profile.key:<10} {profile.name}')
            lines.append(f'             protocol: {profile.protocol}')
            lines.append(f'             required: {", ".join(required) or "none"}')
            if optional:
                lines.append(f'             optional: {", ".join(optional)}')
            lines.append(f'             docs: {profile.docs_url}')
        lines.extend(
            [
                '',
                'Test a profile:',
                '  llm-compat agent test --agent opencode --provider custom \\',
                '    --base-url URL --model MODEL --api-key-env ENV_VAR --yes',
            ]
        )
        text = '\n'.join(lines)
    _emit(text, args)
    return 0


def _agent_checks(profile: Any) -> list[CompatibilityCheck]:
    if not profile.check_ids:
        return []
    return select_checks(only=profile.check_ids)


def _agent_dry_run_payload(
    profile: Any,
    config: ProviderConfig,
    checks: Sequence[CompatibilityCheck],
    request_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        'schema_version': '1.0',
        'tool': {'name': 'llm-compat', 'version': __version__},
        'dry_run': True,
        'agent': profile.to_dict(),
        'provider': {
            'name': config.name,
            'base_url': safe_url(config.base_url),
            'model': config.model,
            'model_source': 'explicit' if config.model else 'auto',
            'auth_configured': config.auth_configured,
        },
        'estimated_requests': request_count,
        'authorization_required': bool(checks) or bool(args.verify_command),
        'verify_command_configured': bool(args.verify_command),
        'checks': [check.id for check in checks],
    }
    return redact_value(payload, config.sensitive_values)


def _render_agent_dry_run(
    profile: Any,
    config: ProviderConfig,
    checks: Sequence[CompatibilityCheck],
    request_count: int,
    verify_command: Optional[str],
) -> str:
    secrets = config.sensitive_values
    lines = [
        f'LLM Compat {__version__} - {profile.name} profile dry run',
        'Dry run: no network requests were made and no agent was launched.',
        f'Protocol: {profile.protocol}',
        f'Provider: {truncate(redact_sensitive(config.name, secrets), 120)}  '
        f'{truncate(redact_sensitive(safe_url(config.base_url), secrets), 300)}',
        f'Model: {truncate(redact_sensitive(config.model, secrets), 200) if config.model else "(auto)"}',
        '',
        'Capabilities',
    ]
    for capability in profile.capabilities:
        requirement = 'required' if capability.required else 'optional'
        check = capability.check_id or 'not implemented'
        lines.append(f'  {capability.name:<30} {requirement:<9} check={check}')
    lines.extend(
        [
            '',
            f'Checks selected: {", ".join(check.id for check in checks) or "none"}',
            f'Estimated live requests: {request_count}',
            (
                'Verification: inferred from protocol checks unless --verify-command is run.'
                if checks
                else 'Verification: no protocol checks are implemented for this profile; '
                'a live test without --verify-command is inconclusive. An authorized '
                '--verify-command adds execution evidence but does not verify the protocol.'
            ),
        ]
    )
    if verify_command:
        lines.append(
            'Verify command: configured; it will run only with --yes after protocol checks.'
        )
    return '\n'.join(lines)


def _run_verify_command(
    command: str,
    config: ProviderConfig,
    timeout: float,
) -> AgentExecution:
    started = time.perf_counter()
    timed_out = False
    exit_code: Optional[int]
    output = ''
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors='replace',
            timeout=timeout,
            check=False,
        )
        exit_code = completed.returncode
        output = (completed.stdout or '') + (completed.stderr or '')
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = (
            exc.stdout.decode(errors='replace')
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode(errors='replace')
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        output = (stdout or '') + (stderr or '')
    except (OSError, ValueError) as exc:
        exit_code = None
        output = str(exc)
    duration_ms = (time.perf_counter() - started) * 1000.0
    secrets = config.sensitive_values
    return AgentExecution(
        command=truncate(redact_sensitive(command, secrets), 500),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration_ms,
        output_preview=truncate(redact_sensitive(output, secrets), 2000),
    )


def _cmd_agent_test(args: argparse.Namespace) -> int:
    profile = get_profile(args.agent)
    if profile is None:
        raise ConfigurationError(f'Unknown agent profile: {args.agent!r}.')
    if args.verify_command is not None and not args.verify_command.strip():
        raise ConfigurationError('--verify-command must be a non-empty command.')
    if not math.isfinite(args.verify_timeout) or args.verify_timeout <= 0:
        raise ConfigurationError('--verify-timeout must be a finite number greater than zero.')
    checks = _agent_checks(profile)
    config = _build_config(args)
    request_count = (
        _estimated_requests(checks, 1, model_explicit=config.model is not None)
        if checks
        else 0
    )
    if args.dry_run:
        if args.format == 'json':
            text = json.dumps(
                _agent_dry_run_payload(profile, config, checks, request_count, args),
                indent=2,
                ensure_ascii=False,
            )
        else:
            text = _render_agent_dry_run(
                profile, config, checks, request_count, args.verify_command
            )
        _emit(text, args)
        return 0
    if args.verify_command and not args.yes:
        print(
            'error: --verify-command runs the actual agent and requires explicit --yes '
            'authorization.',
            file=sys.stderr,
        )
        return 2
    if checks:
        if not _authorize(
            args,
            config,
            purpose=f'{profile.name} protocol checks',
            request_count=request_count,
        ):
            return 2
        protocol_report = run_compatibility(
            config,
            RunOptions(authorize=True, samples=1, max_tokens=128),
            provider=build_provider(config),
            checks=checks,
        )
    else:
        protocol_report = _empty_protocol_report(config)
    execution = None
    if args.verify_command:
        execution = _run_verify_command(
            args.verify_command,
            config,
            args.verify_timeout,
        )
    agent_report = evaluate_agent(profile, protocol_report, execution=execution)
    if args.format == 'json':
        text = json.dumps(agent_report.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = render_agent_text(agent_report, color=_use_color(args))
    _emit(text, args)
    return 0 if agent_report.verdict == 'compatible' else 1


def _research_plan(
    config: Any,
    resolved: Sequence[Any],
) -> tuple[list[CompatibilityCheck], int]:
    checks = select_checks(only=config.checks)
    per_target = [
        _estimated_requests(
            checks,
            config.samples,
            model_explicit=target.requested_model is not None,
        )
        for target in resolved
    ]
    return checks, sum(per_target) * config.trials


def _research_plan_payload(
    config: Any,
    resolved: Sequence[Any],
    checks: Sequence[CompatibilityCheck],
    output_dir: str,
    estimated_requests: int,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    payload = {
        'schema_version': '1.0',
        'tool': {'name': 'llm-compat', 'version': __version__},
        'dry_run': dry_run,
        'experiment': config.name,
        'description': config.description,
        'config_fingerprint': config.with_output_dir(output_dir).fingerprint(resolved),
        'targets': [target.to_safe_dict() for target in resolved],
        'checks': [check.id for check in checks],
        'trials': config.trials,
        'samples': config.samples,
        'max_tokens': config.max_tokens,
        'seed': config.seed,
        'randomize_order': config.randomize_order,
        'request_budget': config.request_budget,
        'rate_limit_per_second': config.rate_limit_per_second,
        'estimated_requests': estimated_requests,
        'authorization_required': True,
        'output_dir': output_dir,
    }
    secrets = tuple(
        secret for target in resolved for secret in target.config.sensitive_values
    )
    return redact_value(payload, secrets)


def _render_research_plan(
    config: Any,
    resolved: Sequence[Any],
    checks: Sequence[CompatibilityCheck],
    output_dir: str,
    estimated_requests: int,
) -> str:
    secrets = tuple(
        secret for target in resolved for secret in target.config.sensitive_values
    )
    lines = [
        f'LLM Compat {__version__} - research plan',
        'No network requests were made.',
        f'Experiment: {truncate(redact_sensitive(config.name, secrets), 200)}',
    ]
    if config.description:
        lines.append(f'Description: {truncate(redact_sensitive(config.description, secrets), 300)}')
    lines.extend(['', f'Targets ({len(resolved)})'])
    for target in resolved:
        lines.append(
            f'  {truncate(redact_sensitive(target.target.name, secrets), 80)}  '
            f'{truncate(redact_sensitive(safe_url(target.config.base_url), secrets), 300)}  '
            f'model={truncate(redact_sensitive(target.requested_model or "(auto)", secrets), 100)}  '
            f'credentials={"configured" if target.config.auth_configured else "not configured"}'
        )
    lines.extend(
        [
            '',
            f'Checks: {", ".join(check.id for check in checks)}',
            f'Trials per target: {config.trials}  samples: {config.samples}  max_tokens: {config.max_tokens}',
            f'Order: {"randomized" if config.randomize_order else "fixed"} (seed={config.seed})',
            f'Request budget: {config.request_budget}  rate limit: '
            + (f'{config.rate_limit_per_second:g}/s' if config.rate_limit_per_second else 'none'),
            f'Estimated live requests: about {estimated_requests}',
            f'Output directory: {truncate(redact_sensitive(output_dir, secrets), 500)}',
            'Authorization: required before any live request; use --yes or type yes interactively.',
        ]
    )
    return '\n'.join(lines)


def _cmd_research_validate(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    resolved = config.resolve_targets(require_credentials=False)
    checks, estimated = _research_plan(config, resolved)
    output_dir = str(Path(config.output_dir).expanduser())
    if args.format == 'json':
        text = json.dumps(
            _research_plan_payload(
                config, resolved, checks, output_dir, estimated, dry_run=True
            ),
            indent=2,
            ensure_ascii=False,
        )
    else:
        text = _render_research_plan(config, resolved, checks, output_dir, estimated)
    _emit(text, args)
    return 0


def _render_research_result(report: Any) -> str:
    overall = report.summary.get('overall', {})
    files = [
        'raw_results.json',
        'results.csv',
        'requests.csv',
        'summary.json',
        'methodology.md',
        'environment.json',
        'experiment.json',
    ]
    secrets = report.secrets
    lines = [
        'Research experiment complete: '
        + truncate(redact_sensitive(report.config.name, secrets), 300),
        f'Output directory: {truncate(redact_sensitive(report.output_dir, secrets), 500)}',
        f'Trials: {overall.get("trials", 0)} / {overall.get("planned_trials", 0)}',
        f'Requests: {overall.get("requests_used", 0)} / {overall.get("request_budget", 0)} budget',
        f'Verdicts: {json.dumps(overall.get("verdicts", {}), sort_keys=True)}',
        f'Budget exhausted: {"yes" if overall.get("budget_exhausted") else "no"}',
        '',
        'Files',
    ]
    lines.extend(f'  {name}' for name in files)
    return '\n'.join(lines)


def _cmd_research_run(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    resolved = config.resolve_targets(require_credentials=not args.dry_run)
    checks, estimated = _research_plan(config, resolved)
    output_dir = str(Path(args.output_dir or config.output_dir).expanduser())
    if args.dry_run:
        if args.format == 'json':
            text = json.dumps(
                _research_plan_payload(
                    config, resolved, checks, output_dir, estimated, dry_run=True
                ),
                indent=2,
                ensure_ascii=False,
            )
        else:
            text = _render_research_plan(config, resolved, checks, output_dir, estimated)
        _emit(text, args)
        return 0
    targets = [
        (
            target.config,
            _estimated_requests(
                checks,
                config.samples,
                model_explicit=target.requested_model is not None,
            )
            * config.trials,
        )
        for target in resolved
    ]
    if not _authorize_targets(
        args,
        targets,
        purpose=f'research experiment {config.name!r} ({config.trials} trial(s) per target)',
    ):
        return 2
    report = run_experiment(config, authorize=True, output_dir=output_dir)
    if args.format == 'json':
        payload = redact_value(
            {
                'schema_version': '1.0',
                'tool': {'name': 'llm-compat', 'version': __version__},
                'output_dir': report.output_dir,
                'files': [
                    'raw_results.json',
                    'results.csv',
                    'requests.csv',
                    'summary.json',
                    'methodology.md',
                    'environment.json',
                    'experiment.json',
                ],
                'summary': report.summary,
            },
            report.secrets,
        )
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = _render_research_result(report)
    _emit(text, args)
    overall = report.summary.get('overall', {})
    if overall.get('budget_exhausted') or any(trial.status == 'error' for trial in report.trials):
        return 1
    return 0


def _print_error(exc: LLMCompatError) -> None:
    print(f'error: {truncate(redact_sensitive(str(exc)), 1000)}', file=sys.stderr)
    if exc.hint:
        print(
            f'hint: {truncate(redact_sensitive(exc.hint), 1000)}',
            file=sys.stderr,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    '''Run the CLI and return a process exit code.'''
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            return _cmd_run(args)
        if args.command == 'compare':
            return _cmd_compare(args)
        if args.command == 'models':
            return _cmd_models(args)
        if args.command == 'checks':
            return _cmd_checks(args)
        if args.command == 'providers':
            return _cmd_providers(args)
        if args.command == 'agent':
            if args.agent_command == 'list':
                return _cmd_agent_list(args)
            if args.agent_command == 'test':
                return _cmd_agent_test(args)
            parser.error(f'Unknown agent command: {args.agent_command}')
        if args.command == 'research':
            if args.research_command == 'validate':
                return _cmd_research_validate(args)
            if args.research_command == 'run':
                return _cmd_research_run(args)
            parser.error(f'Unknown research command: {args.research_command}')
        parser.error(f'Unknown command: {args.command}')
        return 2
    except ConfigurationError as exc:
        _print_error(exc)
        return 2
    except LLMCompatError as exc:
        _print_error(exc)
        return 1
    except KeyboardInterrupt:
        print('error: interrupted.', file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - keep CLI failures readable
        print(
            f'error: unexpected internal error ({type(exc).__name__}); '
            'report this issue if it persists.',
            file=sys.stderr,
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
