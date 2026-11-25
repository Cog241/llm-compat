# Research experiment examples

This directory contains templates for reproducible LLM Compat research runs. No experimental
results are included, and no requests are made unless you explicitly authorize them.

## Three free providers

`three_free_providers.json` compares three OpenAI-compatible endpoints with free tiers:

- Groq (`GROQ_API_KEY`)
- OpenRouter free models (`OPENROUTER_API_KEY`)
- Google AI Studio's OpenAI-compatible endpoint (`GEMINI_API_KEY`)

Free-tier availability, model ids, and rate limits change. Treat this file as a starting point,
not a permanent claim about any provider's catalog. Update the model ids from the provider's
current documentation before running.

Store credentials only in environment variables. The config rejects literal `api_key` fields:

```sh
export GROQ_API_KEY='...'
export OPENROUTER_API_KEY='...'
export GEMINI_API_KEY='...'
```

Inspect the fully resolved plan without sending anything:

```sh
llm-compat research validate examples/research/three_free_providers.json
llm-compat research run examples/research/three_free_providers.json --dry-run
```

Run the experiment after reviewing the plan. The CLI prompts for interactive confirmation;
`--yes` is available for non-interactive automation after you have reviewed the costs:

```sh
llm-compat research run examples/research/three_free_providers.json
```

Or use the helper, which validates first and then runs the authorized experiment:

```sh
examples/research/run_example.sh
```

Outputs are written to `research-output/three-free-providers/` by default:

- `raw_results.json`: sanitized per-trial and per-request records
- `results.csv`: one row per trial
- `requests.csv`: one row per recorded provider request attempt
- `summary.json`: descriptive statistics and failure counts
- `methodology.md`: scope, authorization, conditions, and limitations
- `environment.json`: non-credential runtime metadata for reproducibility
- `experiment.json`: the exact config and fingerprint needed to resume

The experiment is resumable. Re-running the same config skips completed trial ids; it never
duplicates trials or silently changes the configuration fingerprint. Use a new `output_dir` for
a different experiment.

## Interpreting results

These are protocol compatibility measurements, not model-quality benchmarks. A `compatible`
trial means the selected protocol checks passed under the recorded conditions. It does not mean
an agent's full workflow was verified. Run the agent itself with
`llm-compat agent test --verify-command ... --yes` when you need execution-level evidence.

Free providers may rate-limit, change model ids, or route to different backends. The exported
records preserve failures and conditions so readers can distinguish those effects from a
provider that does not implement a feature.
