# LLM Compat

**A universal API compatibility testing toolkit for AI developers.**

[![CI](https://github.com/Cog241/llm-compat/actions/workflows/ci.yml/badge.svg)](https://github.com/Cog241/llm-compat/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LLM Compat checks whether an OpenAI-compatible API provider actually supports the
features that AI applications and coding agents depend on: streaming, tool calling,
structured JSON output, model discovery, authentication, latency, and token throughput.
It produces a clear pass/fail report for humans and a stable JSON document for CI.

It is provider-agnostic, has **zero runtime dependencies**, and is designed so that
new providers and checks can be contributed without changing the engine.

## Why this exists

Many services advertise an "OpenAI-compatible" API. In practice, compatibility varies:
some reject `response_format`, stream no text deltas, return tool calls in a different
shape, advertise models that cannot chat, or fail authentication in confusing ways.
Those differences break applications and coding agents in production.

LLM Compat gives developers a repeatable answer to three questions:

1. Can I connect and authenticate?
2. Which advertised features actually work?
3. Will my existing OpenAI-style client work, and what will it cost in latency?

## Features

- Connectivity and authentication checks against any OpenAI-compatible base URL.
- Automatic model discovery through `GET /models`, with a conservative chat-model selector.
- Basic chat completion validation, including response shape and content.
- Server-Sent Events (SSE) streaming validation: text deltas, finish reason, and `[DONE]`.
- Latency and output-token throughput measurement over repeated samples.
- Tool/function calling validation, including JSON argument parsing.
- Structured JSON output validation through `response_format={"type": "json_object"}`.
- Typed errors with actionable hints and retryability, mapped into PASS/FAIL/WARN/ERROR/SKIP results.
- Human-readable terminal reports and stable JSON exports (schema version `1.0`).
- Presets for OpenAI, Groq, Together, OpenRouter, DeepSeek, and Ollama, plus arbitrary custom endpoints.
- An explicit authorization gate before any billable live request.

## Requirements

- Python 3.9 or newer.
- An OpenAI-compatible HTTP endpoint that you are authorized to test.
- No third-party Python packages at runtime. The default transport uses the standard library.

## Installation

Install from a source checkout:

```bash
git clone https://github.com/Cog241/llm-compat.git
cd llm-compat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
llm-compat --version
```

For an isolated command-line install, `pipx install .` also works from the checkout.
The package is also installable with `python -m pip install .`.

For development, install the test and lint extras:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Export your key as an environment variable. The CLI never accepts a key literal:

```bash
export OPENAI_API_KEY="your-key"
llm-compat run --provider openai --model gpt-4o-mini
```

Because live requests can incur provider charges, the command asks for explicit
confirmation before sending anything. Read the endpoint, purpose, and request estimate,
then type exactly `yes` if you want to continue.

For automation and CI, pass `--yes` after you have reviewed the configuration:

```bash
llm-compat run --provider openai --model gpt-4o-mini --yes --format json
```

If `--model` is omitted, LLM Compat lists the provider's models and selects a likely
chat model. Selection is a heuristic, not a guarantee; pass `--model` when you need a
specific model or a reproducible run.

## Custom OpenAI-compatible endpoints

Use `--provider custom` with a base URL that includes the API version prefix
(for example, `/v1`). Keys are read from the environment variable named by
`--api-key-env`:

```bash
export MY_PROVIDER_KEY="your-key"
llm-compat run \
  --provider custom \
  --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY \
  --model example-chat
```

For custom endpoints, `LLM_COMPAT_API_KEY` is used as a fallback when
`--api-key-env` is not supplied. Endpoints that do not require authentication can omit
the key entirely. Custom headers are supported by the library through
`ProviderConfig(extra_headers=...)`; the CLI intentionally does not take arbitrary
headers so that credentials do not end up in shell history.

### Ollama and other local servers

```bash
llm-compat run --provider ollama --model llama3.2
```

The Ollama preset points at `http://localhost:11434/v1` and does not require a key.
Any local OpenAI-compatible server works with `--provider custom --base-url ...`.

## CLI reference

```text
llm-compat run       Run compatibility checks and render a report
llm-compat models    Discover models advertised by an endpoint
llm-compat checks    List built-in checks (no network access)
llm-compat providers List provider presets (no network access)
```

Common options for `run` and `models`:

| Option | Purpose |
| --- | --- |
| `--provider NAME` | Preset key or `custom` (default: `openai`). |
| `--base-url URL` | Override the preset URL, or supply the URL for `custom`. |
| `--api-key-env ENV_VAR` | Read the API key from this environment variable. |
| `--model MODEL` | Test this model instead of auto-selecting one (`run` only). |
| `--timeout SECONDS` | Per-request timeout (default: 30). |
| `--yes` | Authorize live requests without the interactive prompt. |
| `--format text\|json` | Human-readable output or machine-readable JSON. |
| `--output PATH` | Write output to a file instead of stdout. |

Additional `run` options:

| Option | Purpose |
| --- | --- |
| `--samples N` | Performance samples, 1-10 (default: 3). |
| `--max-tokens N` | Maximum completion tokens per request (default: 128). |
| `--only ID` | Run only this check. May be repeated. |
| `--skip ID` | Skip this check. May be repeated. |
| `--dry-run` | Print the plan and make no network requests. |
| `--no-color` | Disable ANSI colors. `NO_COLOR` is also honored. |

Examples:

```bash
# Inspect what would run, without spending tokens.
llm-compat run --provider openai --model gpt-4o-mini --dry-run

# List models as JSON.
llm-compat models --provider groq --format json --yes

# Run only streaming and tool calling.
llm-compat run --provider openai --model gpt-4o-mini --only streaming --only tools --yes

# Write a CI artifact.
llm-compat run --provider custom --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY --yes --format json --output report.json
```

## Checks

Checks run in this order. "Model" means the check needs a chat model; when no model can
be discovered or selected, model-dependent checks are reported as SKIP rather than
making an unrelated request.

| ID | Check | What it validates | Model |
| --- | --- | --- | --- |
| `connectivity` | API connectivity | The base URL is reachable and answers HTTP. | No |
| `auth` | Authentication | The configured credentials are accepted (401/403 fail). | No |
| `models` | Model discovery | `GET /models` returns a usable OpenAI-style list. | No |
| `chat` | Chat completion | A standard non-streaming chat request returns valid text content. | Yes |
| `streaming` | Streaming compatibility | SSE chunks, incremental content, finish reason, and `[DONE]`. | Yes |
| `latency` | Latency and throughput | Repeated calls measure average latency and output tokens/sec. | Yes |
| `tools` | Tool calling | A tool request returns the expected function name, call id, and JSON arguments. | Yes |
| `json` | Structured JSON output | `response_format=json_object` returns a valid JSON object. | Yes |

Status meanings:

- **PASS**: the feature worked as expected.
- **FAIL**: the provider is reachable but does not support the advertised feature correctly.
- **WARN**: the feature partially works, or a response shape is non-standard but usable.
- **ERROR**: the check could not complete (network, rate limit, server error, or invalid configuration).
- **SKIP**: the check was not executed, usually because no chat model was selected.

The summary verdict is `compatible` when every selected check passed with no failures,
warnings, or skips; `incompatible` when at least one check failed or errored and no check
passed; otherwise `partial` if any result is a warning or skip; and `inconclusive` when no
checks executed. The compatibility score is the percentage of executed checks that
passed; skipped checks are not counted.

## Example output

Illustrative terminal output for a provider whose tool calling does not work:

```text
LLM Compat 0.1.0
Provider: custom  https://llm.example.com/v1
Model: example-chat
API key: configured
Duration: 3.14 s

Checks
  PASS   API connectivity              123 ms
          Endpoint reachable; /models returned 12 model(s).
  PASS   Authentication                118 ms
          API key accepted by the endpoint.
  PASS   Model discovery               120 ms
          Discovered 12 model(s).
  PASS   Chat completion               642 ms
          Received a valid completion in 642 ms.
  PASS   Streaming compatibility       701 ms
          Received 18 content chunk(s); first content in 190 ms; stream completed in 701 ms.
  PASS   Latency and throughput        904 ms
          3/3 sample(s); avg latency 301 ms (min 284, max 319); 23.4 output tokens/sec.
  FAIL   Tool calling                  801 ms
          The model did not return a tool call even though the request asked for one.
  WARN   Structured JSON output        655 ms
          The structured-output response was not valid JSON.

Summary
  Verdict: PARTIAL
  Compatibility score: 75.0%
  Checks: 6 passed, 1 failed, 1 warning, 0 errors, 0 skipped
```

## JSON output

`--format json` emits a stable document with no credentials; `--output PATH` writes either
format to a file. Here is an abridged example with a truncated `results` array:

```json
{
  "schema_version": "1.0",
  "tool": { "name": "llm-compat", "version": "0.1.0" },
  "provider": {
    "name": "custom",
    "base_url": "https://llm.example.com/v1",
    "model": "example-chat",
    "model_source": "explicit",
    "auth_configured": true
  },
  "timing": {
    "started_at": "2026-01-01T00:00:00.000Z",
    "finished_at": "2026-01-01T00:00:03.140Z",
    "duration_ms": 3140.0
  },
  "summary": {
    "verdict": "partial",
    "score": 75.0,
    "total_checks": 8,
    "executed_checks": 8,
    "counts": { "pass": 6, "fail": 1, "warn": 1, "error": 0, "skip": 0 }
  },
  "results": [
    {
      "id": "tools",
      "name": "Tool calling",
      "category": "tools",
      "status": "fail",
      "message": "The model did not return a tool call even though the request asked for one.",
      "duration_ms": 801.2,
      "details": {
        "finish_reason": "stop",
        "content_preview": "I do not have access to weather data.",
        "tool_calls": [],
        "usage": { "prompt_tokens": 38, "completion_tokens": 11, "total_tokens": 49 },
        "latency_ms": 801.2
      },
      "error": null
    }
  ]
}
```

The exact `results` array contains every selected check, including SKIP entries, in
order. Consumers should key off `schema_version` and treat unknown fields as
forward-compatible additions.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every selected check ran and passed or warned; or a non-check command succeeded. |
| `1` | A selected check failed, errored, or was skipped; or no check ran. |
| `2` | Usage, configuration, authorization, or output error. |
| `130` | Interrupted with Ctrl-C. |

`--format json` still uses the same exit codes, so CI can archive the report and fail
on an incompatible or incomplete run without parsing terminal text.

## Authorization, cost, and secrets

- **Live requests require authorization.** The library requires `RunOptions(authorize=True)`
  before `run_compatibility()` will make a request. The CLI requires `--yes` or an interactive
  exact `yes`; live JSON runs must pass `--yes` so output stays non-interactive.
- **Requests cost money.** A full run makes up to `5 + samples` requests: one shared
  `/models` call for connectivity, authentication, and model discovery; one each for chat,
  streaming, tools, and JSON; and `samples` calls for latency. Auto-discovery reuses that
  `/models` call when those checks are selected; otherwise it can add one request. Provider
  pricing and rate limits apply.
- **Keys come from the environment.** LLM Compat does not accept API key literals on the
  command line, does not write keys to reports, and does not emit request headers in errors.
  Use `--api-key-env` or the preset's documented variable.
- **No telemetry.** The only network traffic is to the endpoint you configure.
- **URLs are sanitized.** Base URLs with embedded credentials, query strings, or fragments
  are rejected, and `safe_url` removes them from reports and error messages. Cross-origin
  redirects are not followed, so an Authorization header is never forwarded to another host.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities and more detail.

## Architecture

LLM Compat separates transport, provider adaptation, checks, and presentation:

```text
llm_compat/
  cli.py              argparse CLI, authorization gate, output handling
  runner.py           model selection, check orchestration, report assembly
  config.py           validated provider and run settings
  http.py             injectable HttpClient protocol and urllib transport
  errors.py           typed errors, HTTP error mapping, redaction
  report.py           terminal rendering and JSON export
  results.py          statuses, check results, summaries
  providers/
    base.py           provider contract and normalized chat/stream types
    openai.py         OpenAI-compatible adapter (REST + SSE)
    registry.py       presets and provider construction
  checks/
    base.py           check contract, timing, and error mapping
    connectivity.py   connectivity and authentication
    models.py         model discovery
    chat.py           chat completion
    streaming.py      SSE streaming
    performance.py    latency and throughput
    tools.py          tool/function calling
    structured.py     structured JSON output
    registry.py       ordered check catalog and filtering
tests/
  mock_server.py      local stateful mock of an OpenAI-compatible API
```

The engine depends only on the `Provider` abstraction and the `CompatibilityCheck`
contract. Checks never perform HTTP directly; they receive a provider through a
`CheckContext`. The default `UrllibHttpClient` can be replaced by any object satisfying
the `HttpClient` protocol, which keeps tests fast and offline.

## Extending LLM Compat

### Add a provider preset

Add a `ProviderPreset` to `llm_compat/providers/registry.py` with a stable key, display
name, base URL, and API-key environment variable. Add a test for the preset and, if the
service has unusual behavior, a dedicated adapter test against the mock server. Presets
are only connection shortcuts; they do not change the checks.

### Add a check

1. Subclass `CompatibilityCheck` in `llm_compat/checks/`.
2. Set `id`, `name`, `category`, `description`, and `requires_model`.
3. Implement `run(context)` and return `self.result(...)`. Raise a typed `LLMCompatError`
   for failures; the base class handles timing and maps it to FAIL or ERROR as appropriate.
4. Register the check in `llm_compat/checks/registry.py`.
5. Add mocked success, failure, warning, and error tests in `tests/`.

Check IDs are public API once released; keep them stable, lowercase, and short.

### Add a provider adapter

Subclass `Provider` from `llm_compat/providers/base.py`, normalize responses into
`ChatResponse`, `StreamEvent`, `ModelInfo`, `Usage`, and `ToolCall`, and return it from
`build_provider`. Adapters should never print secrets and should raise the typed errors
from `llm_compat/errors.py` rather than provider-specific exceptions.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

All automated tests use a local mock HTTP server. They do not call real providers, do not
require an API key, and do not need network access. Please do not add live-API tests to
the default suite; any optional live evaluation requires explicit user authorization and
must never be part of CI.

CI runs linting and tests on Python 3.9 through 3.13, then builds and checks the sdist and
wheel. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Limitations

- LLM Compat targets the **OpenAI-compatible** REST surface. Providers with a native,
  non-compatible API (for example, Anthropic's native Messages API) need an adapter.
- Automatic model selection is a heuristic based on model ids. It cannot know your
  account's entitlements, fine-tunes, or application-specific requirements.
- The tool test validates one function-call response; it does not send a tool result or
  complete an agent loop.
- The JSON test validates `response_format=json_object`, not JSON Schema support.
- Latency and throughput are measured from one client location over a small sample. They
  are useful for compatibility and regression signals, not as a provider benchmark.
- There are no automatic retries. Transient 429/5xx responses are reported as ERROR with
  a retryable hint so callers can decide what to do.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
test expectations, and the check/provider contribution workflow.

## Security

Please report vulnerabilities privately; see [SECURITY.md](SECURITY.md). Never include
real API keys, account identifiers, or private endpoint URLs in an issue.

## License

MIT. See [LICENSE](LICENSE).
