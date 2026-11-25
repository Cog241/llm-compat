# Agent compatibility profiles

LLM Compat 0.3 maps documented coding-agent requirements onto its protocol checks. This
page explains what the profiles claim, what they do not claim, and how to maintain them.

## Inference versus execution

Every agent report carries `verification.method`:

- `inferred_protocol`: at least one protocol check ran. The agent itself was not launched.
- `not_run`: the profile has no implemented checks and no execution command ran. This
  report does not infer compatibility from an unrelated protocol.
- `agent_execution`: an explicitly authorized `--verify-command` also ran. The report
  records its exit status, timeout state, duration, and sanitized output preview.

Protocol inference is useful because it isolates endpoint behavior. It is not evidence
that an agent completes a real workflow. Use `verification.agent_execution_verified` and
the execution object before making an execution claim. Never describe a report as an
agent run when that object is `null`.

## Built-in profiles

| Profile | Protocol | Required | Optional |
| --- | --- | --- | --- |
| `aider` | Chat Completions | chat, system | streaming, tools, tools_stream, json_schema, reasoning |
| `cline` | Chat Completions | chat, streaming, system, tools, tools_stream | json_schema, tools_parallel, reasoning |
| `codex` | Responses | unavailable in 0.3 | unavailable in 0.3 |
| `opencode` | Chat Completions | chat, streaming, system, tools, tools_stream | json_schema, tools_parallel, reasoning |
| `roo-code` | Chat Completions | chat, streaming, system, tools, tools_stream | json_schema, tools_parallel, reasoning |

Aider streaming is optional because Aider can be configured for non-streaming requests.
Cline, OpenCode, and Roo Code are modeled with native tool calling as required, including
streamed tool deltas. Codex removed the legacy Chat Completions wire path; because this
release has no Responses API checks, Codex is intentionally inconclusive instead of being
tested against the wrong surface.

Each profile records a `docs_url` and notes explaining its basis. Those notes are part of
the report so readers can see where the requirement came from.

## CLI

```bash
llm-compat agent list
llm-compat agent list --format json
llm-compat agent test --agent opencode --provider custom \
  --base-url https://llm.example.com/v1 --model example-chat --dry-run
llm-compat agent test --agent opencode --provider custom \
  --base-url https://llm.example.com/v1 --model example-chat \
  --api-key-env MY_PROVIDER_KEY --yes --format json
```

`agent test` selects every check referenced by the profile, runs them once, and maps each
check result to a capability. It never enables unrelated checks. A profile with no
implemented checks, such as Codex, sends no protocol requests and reports `not_run` with
an `inconclusive` verdict unless an authorized execution command runs; a failing command
then makes the verdict `incompatible`. The `--agent` option also accepts the aliases
`roo`, `roocode`, and `open-code`.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `compatible` | At least one required capability was tested, every tested required capability passed, optional capabilities did not fail or warn, and, if supplied, the execution command passed. |
| `partial` | Required capabilities were tested but the evidence is mixed: at least one failed, warned, or skipped; or an optional capability failed or warned. |
| `incompatible` | Required capabilities failed and none passed, or the authorized agent command failed. |
| `inconclusive` | No required capability produced a pass, fail, or warning, for example because all are unavailable or skipped. |

Every non-passing capability includes an actionable suggestion. Suggestions combine the
profile's advice with the check's failure classification (authentication, temporary,
unsupported, malformed, model behavior, configuration, or unknown).

## Adding a profile

1. Read the agent's own documentation for its OpenAI-compatible provider contract. Do not
   infer requirements from another agent.
2. Create `llm_compat/agents/profiles/<name>.py` with one `AgentProfile`. Use
   `capability(key, name, check_id, required, description, suggestion)` for each
   capability. Set `check_id=None` and an `unavailable_reason` when LLM Compat has no
   corresponding check.
3. Keep required capabilities conservative. If the agent documents a supported fallback,
   such as Aider's non-streaming mode, mark the capability optional.
4. Export the profile from `llm_compat/agents/profiles/__init__.py` and append it to
   `PROFILES` in display order.
5. Add offline tests for the registry, required/optional mapping, aliases, verdicts, and
   failure suggestions. Do not add live-agent tests to the default suite.
6. Update this table and the README summary.

Check IDs are shared with the protocol checker. Add a new check first if a documented
capability needs a new wire-level probe; do not point a capability at an unrelated check.

## Limitations

- Profiles are snapshots of documented requirements and can lag agent releases.
- A protocol pass is necessary but not sufficient for a successful agent workflow.
- Model quality, context handling, file editing, and tool-result interpretation are out
  of scope for these wire-level checks.
- The verify command is supplied by the operator. LLM Compat runs it only with explicit
  authorization, sanitizes its preview, and does not treat arbitrary command output as
  proof of any specific behavior beyond exit status.
