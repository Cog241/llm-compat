# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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

[0.2.0]: https://github.com/Cog241/llm-compat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Cog241/llm-compat/releases/tag/v0.1.0