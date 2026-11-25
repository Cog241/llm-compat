# Research experiments

The `research` module runs repeatable protocol experiments across independent
provider/model targets. It records what happened, exports sanitized data, and resumes
without duplicating completed trials. It does not invent results and does not send any
request without explicit authorization.

## Authorization model

- The library entry point requires `run_experiment(config, authorize=True)`. Passing a
  truthy non-`True` value or omitting it raises `AuthorizationRequiredError`.
- `research validate` and `research run --dry-run` resolve the config without credentials
  and make no requests.
- A live CLI run prompts for exactly `yes`, or requires `--yes` for reviewed automation.
  Live JSON runs must use `--yes`.
- Automated tests use only `tests/mock_server.py`.

## Config schema

```json
{
  "schema_version": "1.0",
  "name": "Three free OpenAI-compatible providers",
  "description": "Template; no results are bundled.",
  "seed": 20260101,
  "trials": 10,
  "samples": 3,
  "max_tokens": 128,
  "randomize_order": true,
  "request_budget": 400,
  "rate_limit_per_second": 0.5,
  "checks": ["connectivity", "auth", "models", "chat", "streaming", "latency", "tools", "json"],
  "output_dir": "research-output/three-free-providers",
  "resume": true,
  "targets": [
    {
      "name": "groq-free",
      "provider": "groq",
      "model": "llama-3.1-8b-instant",
      "api_key_env": "GROQ_API_KEY",
      "timeout": 45.0
    }
  ]
}
```

| Field | Default | Rules |
| --- | --- | --- |
| `schema_version` | `1.0` | Must match the supported schema. |
| `name` | required | Non-empty string. |
| `description` | empty | String stored in exports and methodology. |
| `seed` | `0` | Non-negative integer used for deterministic ordering. |
| `trials` | `3` | Integer from 1 through 1000 per target. |
| `samples` | `3` | Latency samples per trial, 1 through 20. |
| `max_tokens` | `128` | Completion cap, 1 through 4096. |
| `randomize_order` | `true` | Shuffle the target/trial schedule with `seed`. |
| `request_budget` | `100` | Total request cap, 1 through 1,000,000. |
| `rate_limit_per_second` | `null` | Finite positive client-side request rate, or `null`. |
| `checks` | eight default checks | Known check ids; opt-in checks must be named explicitly. |
| `all_checks` | `false` | Select all fourteen checks. Cannot be combined with `checks`. |
| `output_dir` | `research-output` | Destination directory. |
| `resume` | `true` | Resume matching fingerprints; refuse non-empty output when `false`. |
| `targets` | required | Non-empty list of unique target names. |

Each target accepts:

| Field | Default | Rules |
| --- | --- | --- |
| `name` | required | Unique, non-empty target label. |
| `provider` | `custom` | Built-in preset key or `custom`. |
| `base_url` | preset value | Required for `custom`; validated like other provider URLs. |
| `model` | auto-discover | Explicit model id. Set it for reproducible runs; automatic selection is a heuristic and can change. |
| `model_env` | `null` | Environment variable holding the model id. Mutually exclusive with `model`. |
| `api_key_env` | preset default | Environment variable name. Literal `api_key` is rejected. |
| `timeout` | `30.0` | Positive per-request timeout in seconds. |
| `allow_insecure_http` | `false` | Same narrowly scoped override as the rest of the tool. |

Unknown fields are configuration errors. Credentials are never read from the JSON file,
included in fingerprints, or written to exports.

## CLI workflow

```bash
llm-compat research validate examples/research/three_free_providers.json
llm-compat research run examples/research/three_free_providers.json --dry-run
llm-compat research run examples/research/three_free_providers.json
llm-compat research run examples/research/three_free_providers.json --yes --format json
```

`validate` prints the resolved targets, selected checks, trial count, request estimate,
budget, rate limit, and output directory. It can run without credentials so contributors
can check a config before exporting keys. A live run requires credentials for every
target that names an `api_key_env`.

## Outputs

Every completed run writes these files atomically:

- `raw_results.json`: schema version, tool version, fingerprint, resolved targets,
  environment, authorization statement, summary, trials, checks, and request records.
- `results.csv`: one row per trial, with status, verdict, check counts, request count,
  token totals, and failure kinds.
- `requests.csv`: one row per request attempt, with check id, operation, sequence,
  timestamps, latency, success/completion, requested and response model ids, status code,
  failure kind, error, provider-reported tokens, finish reason, stream event count,
  tool-call count, discovered model count, and up to 200 discovered model ids.
- `summary.json`: descriptive statistics overall and per target.
- `methodology.md`: scope, authorization, design, targets, measurements, limitations, and
  reproducibility details.
- `environment.json`: tool/Python/platform/locale/timezone metadata and an empty runtime
  dependency list. No environment variable values are included. It describes the process
  that wrote the latest export; resuming on another machine updates this file but does not
  attach a separate environment snapshot to earlier trials, so resume on the original
  environment for strict reproducibility.
- `experiment.json`: the resolved config and fingerprint used for resume validation.

CSV cells beginning with `=`, `+`, `-`, `@`, tab, or carriage return are prefixed with an
apostrophe to prevent spreadsheet formula execution.

## Statistics and missing data

- Latency is wall-clock time around one provider call in milliseconds.
- p95 is emitted only when at least 20 request latencies exist. Smaller samples still
  report count, mean, median, min, and max, with `p95_ms: null` and
  `p95_eligible: false`.
- Token totals include only values the provider reported. Missing usage is never filled
  with estimates; JSON summaries expose `*_reported_requests`, and CSV cells are blank
  when no request in that trial reported the field.
- Failure counts combine classified check failures and classified request failures.
- A `budget_exhausted` summary is true only when the budget stopped unfinished work, not
  merely when a run consumed its budget exactly.

## Resume and reproducibility

The fingerprint is SHA-256 over canonical JSON containing the tool version, normalized
config, and resolved safe target metadata. It does not include credential values or the
runtime `auth_configured` flag. An `all_checks` config is exported with an empty `checks`
list plus `all_checks: true`, so the exported wrapper remains re-runnable.

- Resume loads `raw_results.json`, rejects a fingerprint mismatch, rejects unknown trial
  ids and duplicate ids, then skips trial ids that already exist.
- Request accounting and sequence numbers continue from the existing records.
- Randomized order is deterministic for the same seed and target list.
- Auto-discovered model ids are recorded per trial but are not fixed by the fingerprint.
  Set an explicit `model` or `model_env` when model drift would invalidate comparisons.
- Changing any fingerprinted condition, including `output_dir`, requires a new output
  directory or the original config. This is deliberate: silently mixing conditions would
  invalidate the dataset.
- `request_budget` is a hard cap on recorded attempts. The controller reserves before each
  request, paces requests when configured, and stops before a request that would exceed
  the budget. A process interrupted mid-trial resumes that trial and may repeat its
  in-flight requests; completed trials are never repeated.

## Interpreting and sharing

These datasets describe protocol behavior under recorded conditions. They are not model
quality rankings, capacity benchmarks, or permanent provider claims. Before sharing,
review the sanitized files for unexpected endpoint identifiers and follow
[SECURITY.md](../SECURITY.md). Never commit live output containing account or private
endpoint details.

## Testing research changes

Tests must use the local mock server, assert on deterministic fields rather than timing
thresholds, cover sanitization, and exercise resume and budget paths. The default suite
must pass without network access or an API key. See [CONTRIBUTING.md](../CONTRIBUTING.md).
