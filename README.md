# LLM Compat

**Compatibility testing for OpenAI-compatible AI APIs.**

[![CI](https://github.com/Cog241/llm-compat/actions/workflows/ci.yml/badge.svg)](https://github.com/Cog241/llm-compat/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LLM Compat is a zero-runtime-dependency Python toolkit for checking whether an
OpenAI-compatible endpoint supports the features real applications depend on. It runs
repeatable checks, classifies failures, measures streaming performance, and emits a
stable JSON report for CI.

It is provider-agnostic and works with hosted APIs, gateways, and local servers. Use it
to answer: can I connect and authenticate, which advertised features actually work, and
what latency and throughput should I expect?

Version 0.3 adds coding-agent compatibility profiles, explicitly authorized agent
execution evidence, and reproducible research experiments with resumable, sanitized
datasets. The v0.2 features remain: opt-in tool/schema/reasoning checks, multi-provider
comparison, richer performance metrics, stricter credential redaction, and HTTPS
enforcement for remote authenticated endpoints. See [CHANGELOG.md](CHANGELOG.md).

## Requirements

- Python 3.9 or newer.
- An OpenAI-compatible endpoint you are authorized to test.
- No third-party Python packages at runtime. The default transport uses the standard library.

## Installation

Once release artifacts are published on PyPI:

```bash
python -m pip install llm-compat
llm-compat --version
```

The 0.3 source tree is also installable directly from a checkout:

```bash
git clone https://github.com/Cog241/llm-compat.git
cd llm-compat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
llm-compat --version
```

`pipx install .` and `python -m pip install .` also work from the checkout. For
development, install the test and lint extras:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Live requests can cost money, so start with a dry run:

```bash
llm-compat run --provider openai --model gpt-4o-mini --dry-run
```

Then export the key through the environment. The CLI never accepts a key literal:

```bash
export OPENAI_API_KEY="your-key"
llm-compat run --provider openai --model gpt-4o-mini
```

The command asks for explicit confirmation before sending requests. Review the endpoint,
purpose, and request estimate, then type exactly `yes`. For reviewed automation, pass
`--yes`:

```bash
llm-compat run --provider openai --model gpt-4o-mini --yes --format json
```

If `--model` is omitted, LLM Compat selects a likely chat model from `GET /models`.
Selection is a heuristic; pass `--model` when you need a specific or reproducible run.

## Custom and local endpoints

Use `--provider custom` with the API version prefix in the base URL:

```bash
export MY_PROVIDER_KEY="your-key"
llm-compat run \
  --provider custom \
  --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY \
  --model example-chat
```

For custom endpoints, `LLM_COMPAT_API_KEY` is the fallback when `--api-key-env` is not
supplied. Endpoints that need no authentication can omit the key. Custom headers are
available through `ProviderConfig(extra_headers=...)`; the CLI intentionally does not
accept arbitrary headers.

Ollama and other local servers work without a key:

```bash
llm-compat run --provider ollama --model llama3.2
```

The Ollama preset points at `http://localhost:11434/v1`. Loopback HTTP is allowed; a
remote authenticated `http://` endpoint is rejected because the credential would travel
in clear text. Use HTTPS. The only override is `--allow-insecure-http` (library:
`ProviderConfig(allow_insecure_http=True)`), intended solely for an authorized
development environment.

## What gets checked

The default run executes eight checks in this order:

| ID | Check | What it validates | Model |
| --- | --- | --- | --- |
| `connectivity` | API connectivity | The base URL is reachable and answers HTTP. | No |
| `auth` | Authentication | The configured credentials are accepted (401/403 fail). | No |
| `models` | Model discovery | `GET /models` returns a usable OpenAI-style list. | No |
| `chat` | Chat completion | A non-streaming request returns valid text content. | Yes |
| `streaming` | Streaming compatibility | SSE deltas, finish reason, and `[DONE]`. | Yes |
| `latency` | Latency and generation speed | TTFT, total latency, median, p95, and throughput. | Yes |
| `tools` | Tool calling (multi-turn) | A weather call, returned tool result, and follow-up answer. | Yes |
| `json` | Structured JSON output | `response_format=json_object` returns valid JSON. | Yes |

Six additional checks are opt-in. They are excluded by default and selected with `--all`
(or `--only ID`):

| ID | Check | What it validates | Model |
| --- | --- | --- | --- |
| `system` | System messages | A system-role instruction influences the completion. | Yes |
| `tools_forced` | Forced tool selection | `tool_choice` returns exactly the forced function. | Yes |
| `tools_parallel` | Parallel tool calls | Two distinct calls with unique IDs and valid arguments. | Yes |
| `tools_stream` | Streamed tool calls | Tool-call deltas assemble into a usable call. | Yes |
| `json_schema` | Strict JSON Schema | A strict `json_schema` request returns exactly the required fields. | Yes |
| `reasoning` | Reasoning request shape | `max_completion_tokens` without `temperature` is accepted. | Yes |

The default check set and its ordering remain backward compatible with 0.1. `--all`
runs all fourteen checks. `--only` and `--skip` may be repeated; unknown IDs are a
configuration error.

## Statuses, verdicts, and failure classes

Each result has a status:

- **PASS**: the feature worked as expected.
- **FAIL**: the endpoint is reachable but the feature is unsupported or incorrect.
- **WARN**: the feature partially works or returned a usable non-standard response.
- **ERROR**: the check could not complete (network, rate limit, server error, and so on).
- **SKIP**: the check was not executed, usually because no chat model was selected.

Results can also carry a failure classification. This separates a missing feature from
a broken response, a model that ignores instructions, bad credentials, and a transient
provider problem:

| `failure_kind` | Meaning |
| --- | --- |
| `unsupported` | The endpoint rejects a parameter/feature or lacks the route. |
| `malformed` | JSON, SSE, usage, or response structure violates the expected protocol. |
| `model_behavior` | The API is valid, but the model ignored the requested behavior. |
| `authentication` | The credential was rejected (typically HTTP 401 or 403). |
| `temporary` | Timeout, rate limit, or server-side 5xx that may succeed later. |
| `configuration` | Local configuration prevented a valid request. |
| `unknown` | No more specific classification was available. |

Reports expose `failure_kind` per result, `error.failure_kind` when an error object
exists, and aggregate `summary.failure_kinds` counts.

The summary verdict is `compatible` when every selected check passes with no warnings
or skips; `inconclusive` when no check executed; `incompatible` when at least one check
failed or errored and none passed; and `partial` otherwise, including warnings, skips, or
a mix of passes and failures. The score is the percentage of executed checks that passed;
skipped checks are excluded.

## Performance measurements

The `latency` check sends repeated streaming requests and reports:

- **Time to first token (TTFT)**: time from request start to the first content chunk.
- **Total latency**: end-to-end time for each stream that produced content.
- **Generation speed**: completion tokens divided by generation time (total minus TTFT).
- **End-to-end speed**: completion tokens divided by total latency.
- **Median latency**: always reported once at least one sample produces content.
- **P95 latency**: reported only after at least 20 measured latency samples; otherwise
  the field is `null`. Streams with incomplete metadata are timed and flagged separately.

`--samples` accepts 1 through 20 (default 3). The details include `latency_ms`,
`ttft_ms`, `generation_tokens_per_sec`, and `end_to_end_tokens_per_sec` statistics
(`avg`, `min`, `max`, `median`, `p95`, `samples`), plus `total_latency_ms`,
`median_latency_ms`, `p95_latency_ms`, and `p95_min_samples`.

Speeds use provider-reported completion-token counts; tokens are never estimated. If
usage is missing or a stream ends without complete metadata, the check warns with
`malformed` rather than inventing a number. If the provider rejects
`stream_options.include_usage`, the check retries without it and records
`stream_options_supported: false`.

These are client-side compatibility signals from one location, not a benchmark of
provider capacity.

## Comparing providers

`compare` runs the same checks against independently configured targets. Each target may
have its own base URL, model, timeout, and API-key environment variable. Keys are still
read only from the environment.

Create `targets.json`:

```json
{
  "providers": [
    {
      "name": "openai-mini",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key_env": "OPENAI_API_KEY"
    },
    {
      "name": "local-ollama",
      "provider": "ollama",
      "base_url": "http://localhost:11434/v1",
      "model": "llama3.2"
    }
  ]
}
```

Each target supports `name`, `provider`, `base_url`, `model`, `timeout`, and
`api_key_env`. A top-level JSON array of targets is also accepted. Literal `api_key`
fields are rejected; use `api_key_env`. Names must be unique for a live comparison.

```bash
# Validate the plan without network requests.
llm-compat compare --config targets.json --dry-run

# Run the default checks with explicit authorization.
llm-compat compare --config targets.json --yes

# Compare all fourteen checks and write JSON.
llm-compat compare --config targets.json --all --yes --format json --output compare.json
```

Comparison output has a `providers` array, each with the same provider, timing,
summary, and results sections as a single-provider report. Failures are printed with
their failure class. The library equivalent is `compare_providers(configs, options)`.

## Coding-agent compatibility profiles

Agent profiles map documented agent requirements onto the same protocol checks. They
answer a narrower question than an agent run: does the endpoint expose the request,
streaming, and tool shapes this agent expects? Reports use `inferred_protocol` when
checks ran, `not_run` when a profile has no implemented checks, and `agent_execution`
only when you authorize a command that actually runs the agent.

| Profile | Wire protocol | Required capabilities |
| --- | --- | --- |
| `aider` | Chat Completions | Chat completions and system messages. Streaming is optional because Aider can use `--no-stream`. |
| `cline` | Chat Completions | Chat, streaming, system messages, multi-turn tools, and streamed tool calls. |
| `codex` | Responses | Responses API, Responses streaming, and Responses function tools. No Chat Completions checks are inferred. |
| `opencode` | Chat Completions | Chat, streaming, system messages, multi-turn tools, and streamed tool calls. |
| `roo-code` | Chat Completions | Chat, streaming, system messages, multi-turn tools, and streamed tool calls. |

Optional capabilities such as strict schema output, parallel tools, and reasoning request
shapes are also reported. Use `agent list` to see the documentation basis and every
capability:

```bash
llm-compat agent list
llm-compat agent list --format json
```

Inspect a profile plan without sending requests:

```bash
llm-compat agent test --agent opencode \
  --provider custom --base-url https://llm.example.com/v1 \
  --model example-chat --dry-run
```

Run the inferred protocol tests after reviewing the plan:

```bash
llm-compat agent test --agent opencode \
  --provider custom --base-url https://llm.example.com/v1 \
  --model example-chat --api-key-env MY_PROVIDER_KEY --yes \
  --format json --output agent-report.json
```

To collect execution-level evidence, pass an explicitly authorized command that launches
the real agent. The command runs only with `--yes`; its command string and output preview
are sanitized before they appear in the report. Otherwise the report never claims the
agent itself was run:

```bash
llm-compat agent test --agent opencode \
  --provider custom --base-url https://llm.example.com/v1 \
  --model example-chat --api-key-env MY_PROVIDER_KEY --yes \
  --verify-command 'opencode run "reply with ok"' --verify-timeout 300
```

Codex currently uses the Responses API, so this release reports its capabilities as
unavailable/inconclusive rather than treating Chat Completions success as evidence. See
[docs/agent-profiles.md](docs/agent-profiles.md) for the profile contract and how to add
one.

## Reproducible research experiments

The research runner repeats trials across independent targets, records request-level
outcomes, and exports sanitized datasets. It is designed for measurement, not for
fabricating or publishing compatibility claims. Automated tests use only the local mock
server; no bundled result files are provider claims.

```bash
# Validate the config and print the resolved plan without credentials or requests.
llm-compat research validate examples/research/three_free_providers.json

# Show the same plan without sending anything.
llm-compat research run examples/research/three_free_providers.json --dry-run

# Run only after reviewing the plan and authorizing the cost.
llm-compat research run examples/research/three_free_providers.json
```

The example compares three free-tier OpenAI-compatible providers. Free-tier model ids
and limits change, so update the template from current provider documentation before
running it. Credentials are read only from environment variables; the config rejects
literal `api_key` fields.

A run writes `raw_results.json`, `results.csv`, `requests.csv`, `summary.json`,
`methodology.md`, `environment.json`, and `experiment.json` to the configured output
directory. Resume skips completed trial ids only when the config fingerprint matches, so
interrupted experiments continue without duplicate trials. Request budgets, deterministic
randomized ordering, seeds, sample sizes, and client-side rate limits are all recorded.
See [docs/research-experiments.md](docs/research-experiments.md) for the schema, output
columns, statistics, and reproducibility rules.

## CLI reference

```text
llm-compat run              Run compatibility checks and render a report
llm-compat compare          Run the same checks against multiple configured targets
llm-compat models           Discover models advertised by an endpoint
llm-compat checks           List built-in checks (no network access)
llm-compat providers        List provider presets (no network access)
llm-compat agent list       List coding-agent profiles (no network access)
llm-compat agent test       Run protocol checks mapped to an agent profile
llm-compat research validate Validate an experiment config (no network access)
llm-compat research run     Run an authorized, resumable experiment
```

Common connection and output options:

| Option | Purpose |
| --- | --- |
| `--provider NAME` | Preset key or `custom` (default: `openai`). |
| `--base-url URL` | Override the preset URL, or supply the URL for `custom`. |
| `--api-key-env ENV_VAR` | Read the API key from this environment variable. |
| `--model MODEL` | Test this model instead of auto-selecting one. |
| `--timeout SECONDS` | Per-request timeout (default: 30). |
| `--allow-insecure-http` | Permit remote authenticated HTTP only in an authorized dev environment. |
| `--yes` | Authorize live requests without the interactive prompt. |
| `--format text\|json` | Human-readable output or machine-readable JSON. |
| `--output PATH` | Write output to a file instead of stdout. |

Additional `run` and `compare` options:

| Option | Purpose |
| --- | --- |
| `--samples N` | Performance samples, 1-20 (default: 3). |
| `--max-tokens N` | Maximum completion tokens per request (default: 128). |
| `--only ID` | Run only this check. May be repeated. |
| `--skip ID` | Skip this check. May be repeated. |
| `--all` | Include the six opt-in checks. |
| `--dry-run` | Print the plan and make no network requests. |
| `--no-color` | Disable ANSI colors (`run`, `compare`, and `agent test`). `NO_COLOR` is also honored. |

Agent and research options:

| Option | Purpose |
| --- | --- |
| `agent test --agent KEY` | Select `aider`, `cline`, `codex`, `opencode`, or `roo-code`; aliases `roo`, `roocode`, and `open-code` are accepted. |
| `--verify-command CMD` | Run the actual agent after protocol checks. Requires `--yes`. |
| `--verify-timeout SECONDS` | Bound the verify command (default: 300). |
| `research validate CONFIG` | Resolve and validate a config without credentials or requests. |
| `research run CONFIG` | Run the experiment; prompts unless `--yes` is supplied. |
| `--output-dir PATH` | Override the experiment output directory. |

Examples:

```bash
# Inspect what would run, without spending tokens.
llm-compat run --provider openai --model gpt-4o-mini --dry-run

# List models as JSON.
llm-compat models --provider groq --format json --yes

# Run only streaming and the multi-turn tool check.
llm-compat run --provider openai --model gpt-4o-mini \
  --only streaming --only tools --yes

# Exercise strict schema, reasoning shape, and advanced tools.
llm-compat run --provider custom --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY --all --samples 20 --yes

# Write a CI artifact.
llm-compat run --provider custom --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY --yes --format json --output report.json

# Map protocol checks to an agent profile without running the agent.
llm-compat agent test --agent cline \
  --provider custom --base-url https://llm.example.com/v1 \
  --model example-chat --api-key-env MY_PROVIDER_KEY --yes --format json

# Validate a resumable research experiment and then authorize it.
llm-compat research validate examples/research/three_free_providers.json
llm-compat research run examples/research/three_free_providers.json --yes
```

## Example terminal output

Illustrative output for an endpoint whose multi-turn tool calling fails:

```text
LLM Compat 0.3.0
Provider: custom  https://llm.example.com/v1
Model: example-chat
API key: configured
Duration: 4.07 s

Checks
  PASS   API connectivity                121 ms
          Endpoint reachable; /models returned 12 model(s).
  PASS   Authentication                  118 ms
          API key accepted by the endpoint.
  PASS   Model discovery                 120 ms
          Discovered 12 model(s).
  PASS   Chat completion                 642 ms
          Received a valid completion in 642 ms.
  WARN   Streaming compatibility         701 ms
          Received 18 content chunk(s); first content in 190 ms;
          the stream ended without a [DONE] marker.
          Failure class: malformed
  PASS   Latency and generation speed    904 ms
          3/3 sample(s); median latency 298 ms, avg 301 ms; first token 190 ms;
          23.4 output tokens/sec.
  FAIL   Tool calling (multi-turn)       811 ms
          The model did not return a tool call even though the request asked for one.
          Failure class: model_behavior
  PASS   Structured JSON output          655 ms
          Returned valid JSON with the requested name and version fields.

Summary
  Verdict: PARTIAL
  Compatibility score: 75.0%
  Checks: 6 passed, 1 failed, 1 warning, 0 errors, 0 skipped
  Failure classes: malformed=1, model_behavior=1
```

## JSON output

`--format json` emits a stable document. Credentials are never serialized, and string
surfaces are redacted for known secrets and common key patterns. This abridged example
shows the stable 0.3 fields:

```json
{
  "schema_version": "1.0",
  "tool": { "name": "llm-compat", "version": "0.3.0" },
  "provider": {
    "name": "custom",
    "base_url": "https://llm.example.com/v1",
    "model": "example-chat",
    "model_source": "explicit",
    "auth_configured": true
  },
  "timing": {
    "started_at": "2026-01-01T00:00:00.000Z",
    "finished_at": "2026-01-01T00:00:04.072Z",
    "duration_ms": 4072.0
  },
  "summary": {
    "verdict": "partial",
    "score": 75.0,
    "total_checks": 8,
    "executed_checks": 8,
    "counts": { "pass": 6, "fail": 1, "warn": 1, "error": 0, "skip": 0 },
    "failure_kinds": { "malformed": 1, "model_behavior": 1 }
  },
  "results": [
    {
      "id": "tools",
      "name": "Tool calling (multi-turn)",
      "category": "tools",
      "status": "fail",
      "failure_kind": "model_behavior",
      "message": "The model did not return a tool call even though the request asked for one.",
      "duration_ms": 811.2,
      "details": { "finish_reason": "stop", "tool_calls": [] },
      "error": null
    }
  ]
}
```

The `results` array contains every selected check, including SKIP entries, in execution
order. Consumers should key off `schema_version` and treat unknown fields as
forward-compatible additions.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Every selected check passed or warned; an agent report is `compatible`; a research run completed; or a non-check command succeeded. |
| `1` | A selected check failed, errored, or was skipped; an agent report is `incompatible`, `partial`, or `inconclusive`; a research run was budget-limited or recorded an error; or no check ran. |
| `2` | Usage, configuration, authorization, or output error. |
| `130` | Interrupted with Ctrl-C. |

JSON output uses the same exit codes, so CI can archive the report and still fail on an
incompatible or incomplete run.

## Authorization, cost, and secrets

- **Explicit authorization.** The library requires `RunOptions(authorize=True)` before
  `run_compatibility()` or `compare_providers()` sends a request, and
  `run_experiment(..., authorize=True)` before a research run sends one. The CLI requires
  `--yes` or an interactive exact `yes`. Agent `--verify-command` runs also require
  `--yes`. Live JSON runs must pass `--yes` so they remain non-interactive. Dry runs,
  `agent list`, a Codex profile test without a verify command, and `research validate`
  make no requests.
- **Requests cost money.** A default run estimates `6 + samples` requests: one shared
  `/models` request (also used for automatic model selection), one each for chat,
  streaming, and JSON, two for the multi-turn tool check, and `samples` for latency.
  `--all` adds six requests, and a stream-options fallback can add one retry. Provider
  pricing and rate limits apply.
- **Environment-only keys.** The CLI accepts only an environment variable name via
  `--api-key-env` or the preset's variable. It does not accept key literals on the
  command line.
- **Credential redaction.** Known API keys and authorization header values are removed
  from reports, previews, exceptions, and logs. Common patterns are also redacted, such
  as `Bearer ...`, `Basic ...`, `sk-...`, `gsk_...`, `coa_...`, `xai-...`,
  `ghp_...`, `github_pat_...`, `xox...`, `AIza...`, and `api_key=...`-style pairs.
- **HTTPS by default.** TLS verification is always on. Remote authenticated endpoints
  must use HTTPS; loopback HTTP is allowed for local servers. The explicit
  `--allow-insecure-http` override is for authorized development environments only.
- **URL and redirect safety.** Base URLs with credentials, query strings, fragments,
  control characters, or invalid ports are rejected. Cross-origin redirects are not
  followed, and redirects containing user information are rejected.
- **No telemetry.** The only network traffic goes to the endpoint you configure.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities and more detail.

## Optional live testing

Live provider testing is not part of the automated suite and is never run by CI. It is
an optional, explicitly authorized procedure for a maintainer or contributor who owns
a scoped test account. Results are environment-specific, can change without notice, and
are not stored in this repository as claims of provider compatibility.

1. Confirm you are authorized to send billable requests and that the provider's terms
   permit compatibility testing.
2. Use a dedicated, minimally scoped test key. Never use a production key.
3. Put the key only in an environment variable, then inspect the plan without
   network traffic:

```bash
export LLM_COMPAT_TEST_KEY="..."
llm-compat run \
  --provider custom \
  --base-url https://llm.example.com/v1 \
  --api-key-env LLM_COMPAT_TEST_KEY \
  --model example-chat \
  --samples 1 \
  --dry-run
```

4. Start with low-cost, low-volume checks and add features deliberately:

```bash
llm-compat run \
  --provider custom \
  --base-url https://llm.example.com/v1 \
  --api-key-env LLM_COMPAT_TEST_KEY \
  --model example-chat \
  --only connectivity --only auth --only chat --only streaming \
  --samples 1 --yes --format json --output /tmp/llm-compat-live.json
```

5. Review `/tmp/llm-compat-live.json` for unexpected secrets before sharing it, then
   delete it. Never commit live reports, account identifiers, or private endpoints.
6. Report only what the report actually says. Do not turn a single environment-specific
   run into a general provider claim.

## Architecture

LLM Compat separates transport, provider adaptation, checks, and presentation:

```text
llm_compat/
  cli.py              argparse CLI, authorization gate, output handling
  runner.py           model selection, check orchestration, authorization
  config.py           validated provider and run settings
  http.py             stdlib transport, same-origin redirects, TLS verification
  providers/          OpenAI-compatible adapter and provider presets
  checks/             one focused check per compatibility property
  checks/base.py      check contract, timing, status mapping, error info
  checks/_tooling.py  shared tool definitions and validation
  compare.py          multi-provider orchestration and rendering
  agents/             agent profile registry, capability mapping, and reports
  research/           experiment config, recording, runner, and exports
  results.py          status, failure classification, summaries
  report.py           stable JSON and terminal report rendering
  utils.py            redaction, truncation, URL and time helpers
```

Provider adapters normalize HTTP responses into `ModelInfo`, `ChatResponse`,
`StreamEvent`, `Usage`, and `ToolCall`. Checks never perform their own HTTP work; they
call the provider through `CheckContext`. The default transport is injectable, which
keeps tests local and provider adapters testable without network access.

## Extending

To add a check, subclass `CompatibilityCheck` in a module under `llm_compat/checks/`,
set a stable lowercase `id` and `default_enabled`, implement `run(context)`, and register
it in `llm_compat/checks/registry.py`. Raise typed errors from `llm_compat/errors.py`
for protocol or HTTP failures; the base class records timing and maps exceptions to
FAIL or ERROR. Use `context.settings` for bounded inputs and `context.redact` for
previews.

Check IDs are public API. Do not rename or repurpose one after release; add a new check
instead. Keep each check focused on one compatibility property so the result identifies
a concrete problem.

To add an agent profile, define one `AgentProfile` with `AgentCapability` entries under
`llm_compat/agents/profiles/`, export it from that package, and add tests. To add a
research dimension, extend the config schema, record the extra condition, and export it;
do not invent missing measurements. See [docs/agent-profiles.md](docs/agent-profiles.md)
and [docs/research-experiments.md](docs/research-experiments.md).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

CI runs the tests and lint on Python 3.9 through 3.13, then builds the sdist and wheel
and checks their metadata. New behavior needs tests against the local mock server. Do
not add live-provider tests to the default suite, and do not run `ruff format` as part
of the repository checks.

Package checks:

```bash
python -m build
python -m twine check dist/*
```

## Limitations

- Checks use small, bounded samples. P95 requires 20 samples and remains a rough signal
  from one client location.
- Model discovery and automatic model selection are heuristics, not provider contracts.
- The strict schema check exercises one fixed schema; it is not a general JSON Schema
  conformance engine.
- Default chat checks send standard `temperature` and `max_tokens` parameters. A model
  that rejects them may require the opt-in `reasoning` check or provider-specific handling.
- The reasoning check validates request shape and a basic response; it does not evaluate
  reasoning quality or reasoning-token accounting.
- Tool checks cover the modern `tools`/`tool_calls` shape. Legacy `function_call` input
  and provider-specific agent APIs are out of scope.
- There are no automatic retries for transient failures. Transient 408/425/429/5xx
  responses are classified as `temporary` so callers can decide what to do. The latency
  check has one targeted fallback when a provider rejects streamed usage options.
- Comparison runs sequentially. It is a compatibility report, not a load or capacity
  benchmark.
- Agent profiles are protocol inferences unless a separate `--verify-command` execution
  is recorded. They do not prove that an agent completes real coding tasks.
- Current Codex requires the Responses API. This release does not implement Responses
  checks, so Codex is reported as inconclusive rather than mapped onto Chat Completions.
- Research experiments measure the conditions you configure. Free-tier model ids, rate
  limits, routing, and prices can change; exported data is not a permanent provider claim.
  Automatic model selection is recorded per trial but is not fixed by the config
  fingerprint, so use an explicit model when reproducibility matters.
- Live results are environment-specific and are not claimed as general provider
  compatibility in this repository.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
test expectations, and the check/provider contribution workflow.

## Security

Please report vulnerabilities privately; see [SECURITY.md](SECURITY.md). Never include
real API keys, account identifiers, or private endpoint URLs in an issue.

## License

MIT. See [LICENSE](LICENSE).