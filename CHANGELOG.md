# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-24

### Added

- Agent compatibility profiles for Aider, Cline, Codex, OpenCode, and Roo Code, with
  required and optional capability mappings, documentation links, actionable
  suggestions, and `llm-compat agent list` / `agent test`.
- Explicit `verification_method` reporting: protocol checks produce
  `inferred_protocol`, profiles with no implemented checks report `not_run`, and an
  authorized `--verify-command` records sanitized `agent_execution` evidence separately.
- Reproducible research experiments with validated JSON configs, environment-only
  credentials, deterministic randomized trials, seeds, repeated samples, request budgets,
  client-side rate limits, and strict authorization.
- Resumable experiment runs guarded by a SHA-256 config fingerprint, with no duplicate
  completed trials.
- Sanitized exports: `raw_results.json`, `results.csv`, `requests.csv`, `summary.json`,
  `methodology.md`, `environment.json`, and `experiment.json`, including request
  timestamps, latency, model ids, provider-reported usage, finish reasons, and failure
  classifications.
- Descriptive statistics with honest missing-data handling: p95 requires at least 20
  latencies, and token totals are never estimated.
- A three-provider free-tier research example and documentation for agent profiles and
  research experiments.
- GitHub issue forms, a pull-request template, and contributor guidance for profiles and
  research dimensions.

### Changed

- `llm-compat research validate` and `research run --dry-run` resolve configs without
  credentials and make no requests.
- The Codex profile reports its Responses API capabilities as unavailable/inconclusive
  rather than inferring compatibility from Chat Completions.
- Aider streaming is optional because Aider supports non-streaming operation.
- CI pins Python 3.9 to Ubuntu 22.04 and uses current Ubuntu images for newer versions.
- `examples/` and `docs/` are included in source distributions.

### Fixed

- The GitHub Actions Python 3.9 job no longer runs on an unsupported runner image.
- Research exports now redact configured secrets from methodology tables, trial and
  request CSV rows, report serialization, and CLI plan/result output, including names and
  output paths.
- Research plan fingerprints use the effective output directory, including an
  `--output-dir` override, so plan metadata and resume metadata agree.
- `agent test --agent` accepts the documented `roo`, `roocode`, and `open-code` aliases.
- `agent test --verify-command` rejects a blank command instead of silently skipping the
  requested execution check.
- `all_checks` research configs can change output directories or resume without tripping
  the mutually exclusive `checks` / `all_checks` validation.

### Compatibility

- The default eight checks, opt-in check behavior, and JSON report schema `1.0` remain
  backward compatible. Runtime dependencies remain empty.

## [0.2.0] - 2026-11-24

### Added

- Multi-turn tool calling: the `tools` check now returns a tool result and validates that
  the follow-up answer uses it.
- Opt-in checks for forced tool selection, parallel tool calls, streamed tool-call deltas,
  strict JSON Schema output, system messages, and reasoning-model request shapes.
- Failure classifications on results and errors: `unsupported`, `malformed`,
  `model_behavior`, `authentication`, `temporary`, `configuration`, and `unknown`.
  Reports include per-result `failure_kind` and summary `failure_kinds` totals.
- Performance details for time to first token, total latency, generation speed,
  end-to-end speed, median latency, and p95 latency. P95 is reported only when at least
  20 measured latency samples are available. The maximum sample count is now 20.
- Multi-provider comparison through `compare_providers()` and the `llm-compat compare`
  CLI command. Each target can use an independent endpoint, model, timeout, and API-key
  environment variable.
- Explicit HTTPS enforcement for remote authenticated endpoints, with a narrowly scoped
  `allow_insecure_http` / `--allow-insecure-http` override for authorized development
  environments.
- Regression coverage for malformed models, chat, streaming, and inconsistent HTTP
  responses, plus credential-redaction and comparison tests.
- Documentation for the optional live-testing procedure. Live testing remains
  authorization-only and is never part of the automated test suite.

### Changed

- Reports redact known credentials and common credential patterns from provider names,
  models, messages, details, errors, hints, previews, and comparison output.
- Tool-call checks validate call IDs, JSON arguments, and finish reasons more strictly.
- `RunSettings.samples` accepts 1 through 20 (previously 1 through 10).
- The default eight checks and the stable JSON report schema remain backward compatible;
  the six new checks are opt-in and enabled with `--all`.

### Removed

- The tracked macOS `.DS_Store` file.

## [0.1.0] - 2026-11-24

- Initial release: connectivity, authentication, model discovery, chat, streaming,
  latency, tool calling, and structured JSON checks; CLI, library API, stable JSON reports,
  typed errors, authorization gate, and zero runtime dependencies.

[0.3.0]: https://github.com/Cog241/llm-compat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Cog241/llm-compat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Cog241/llm-compat/releases/tag/v0.1.0