# Contributing to LLM Compat

Thank you for helping make LLM compatibility testing more reliable. Contributions of
checks, provider presets and adapters, tests, documentation, and bug reports are welcome.

## Before you start

- Open an issue before a large change so the approach can be discussed first.
- Keep pull requests focused. Separate refactors from behavior changes and documentation.
- Never include real API keys, account identifiers, private endpoint URLs, or provider
  response bodies containing personal data in an issue or commit.

## Development setup

```bash
git clone https://github.com/Cog241/llm-compat.git
cd llm-compat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package supports Python 3.9 and newer and has no runtime dependencies. Please avoid
adding a runtime dependency unless it is essential; the standard-library transport is a
deliberate design choice.

## Quality gates

Run these before opening a pull request:

```bash
pytest
ruff check .
```

CI runs the same commands on Python 3.9 through 3.13, then builds the sdist and wheel and
checks their metadata. A change is not complete until the tests pass and new behavior is
covered by tests.

## Testing policy

All automated tests use `tests/mock_server.py`, a local stateful mock of an
OpenAI-compatible API. Tests must:

- Pass without network access and without an API key.
- Cover the success path plus relevant failure, warning, and error paths.
- Avoid sleeps, wall-clock thresholds, and other sources of flakiness.
- Assert on normalized results and sanitized output, not on secrets or raw headers.

Do not add live-provider tests to the default suite. If you need an optional manual live
check, require explicit authorization, keep it out of CI, and never commit the output.
Follow the optional live-testing procedure in
[README.md](README.md#optional-live-testing).

## Adding a check

1. Create a module under `llm_compat/checks/` with a class that subclasses
   `CompatibilityCheck`.
2. Set a stable lowercase `id`, a human-readable `name`, a `category`, a one-line
   `description`, and `requires_model` when the check needs a selected chat model. Set
   `default_enabled = False` for opt-in checks that may reject valid provider parameters.
3. Implement `run(context)`. Use `context.provider` for all HTTP work and
   `context.settings` for bounded user settings.
4. Return `self.result(Status.PASS|FAIL|WARN|ERROR|SKIP, message, details=..., error=...)`.
   Raise typed errors from `llm_compat/errors.py` for failures; the base class records
   timing and maps exceptions to FAIL or ERROR.
5. Attach the most specific `FailureKind` (`unsupported`, `malformed`, `model_behavior`,
   `authentication`, `temporary`, `configuration`, or `unknown`) so users can distinguish
   a missing feature from a broken response or a transient provider problem.
6. Use `context.redact` for response previews and never include secrets or credentials
   in messages. It redacts known credentials before truncating.
7. Register the check in `llm_compat/checks/registry.py`.
8. Add tests against the mock server and update the check table in `README.md`.

Check IDs are public API. Once released, do not rename or repurpose an ID; add a new
check instead. Keep a check focused on one compatibility property so failures identify
a concrete problem.

## Adding provider presets and adapters

For another OpenAI-compatible service, add a `ProviderPreset` in
`llm_compat/providers/registry.py` and a test for the preset. Presets only supply
connection defaults.

For a provider with a different wire format, subclass `Provider` in
`llm_compat/providers/base.py` and normalize responses into `ChatResponse`,
`StreamEvent`, `ModelInfo`, `Usage`, and `ToolCall`. Wire it into `build_provider`, map
provider failures to the typed errors in `llm_compat/errors.py`, and add adapter tests
against the mock server or a protocol-specific mock.

## Code style

- Follow the existing style: single-quoted strings, type hints, `from __future__ import
  annotations`, and line length up to 100 characters.
- Keep public functions documented with short docstrings.
- Prefer small, composable functions over deep class hierarchies.
- Do not introduce an option that silently changes report schema semantics. If the JSON
  schema changes incompatibly, update `schema_version` and document the migration.
- Preserve the PASS/FAIL/WARN/ERROR/SKIP meanings described in the README.

## Pull requests

A good pull request includes:

- A concise description of the problem and the chosen solution.
- Tests that fail before the change and pass after it.
- Documentation updates for user-visible behavior.
- Confirmation that `pytest` and `ruff check .` pass locally.

Please keep commits readable and avoid committing generated reports, virtual
environments, or credentials. Maintainers may ask for a smaller scope or a different
abstraction when a change affects the public check or provider contracts.

## Reporting security issues

Do not report vulnerabilities in a public issue. Follow [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contribution is licensed under the MIT License.
