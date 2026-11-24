# Security Policy

LLM Compat sends authenticated requests to API endpoints chosen by the user. We take the
handling of credentials, URLs, and generated reports seriously.

## Supported versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |

While the project is pre-1.0, security fixes are made on the latest release only.

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/Cog241/llm-compat/security/advisories/new).
If that is unavailable, contact the maintainers through the repository rather than
opening a public issue.

Include:

- A description of the issue and its impact.
- Steps to reproduce, ideally against a local mock endpoint.
- Affected version and Python version.
- Any suggested fix, if you have one.

Do **not** include real API keys, access tokens, personal data, or private endpoint
URLs. Redact them before sending a report.

We will acknowledge a report as soon as practical, investigate it, and coordinate a
disclosure and release. Please allow time for a fix before publishing details.

## Security design

- **Explicit authorization.** The library requires `RunOptions(authorize=True)` before a
  live run. The CLI requires `--yes` or an interactive exact `yes`. Dry runs and non-live
  commands make no requests.
- **Environment-only keys.** The CLI accepts an environment variable name via
  `--api-key-env`, never a key literal. Cross-origin redirects are rejected, so requests
  and their Authorization headers go only to the configured endpoint (same-origin
  redirects, such as a missing trailing slash, are followed).
- **No key serialization.** Reports contain an `auth_configured` boolean, not the key or
  authorization header. Error messages do not include request headers.
- **URL sanitization.** `safe_url` removes user information, query strings, and fragments
  before URLs are written to terminal output or JSON reports.
- **No telemetry.** LLM Compat has no analytics or background network activity. Traffic
  goes only to the endpoint the user configures.
- **TLS verification.** The default transport uses standard-library TLS verification and
  does not provide an option to disable certificate checks.
- **Bounded input.** Base URLs, headers, timeouts, sample counts, and token limits are
  validated before requests are sent. Base URLs with internal whitespace, control
  characters, credentials, query strings, fragments, or invalid ports are rejected.
  API-key and header values containing control characters are rejected, and header names
  must be valid HTTP tokens.

## Credential handling guidance

API keys are read from the environment, so they can still appear in process environments,
CI configuration, shell history, or crash dumps outside LLM Compat's control. Use your
platform's secret store, scope keys to the smallest necessary permissions, rotate keys
regularly, and avoid exporting keys in shared shells.

The CLI avoids passing keys as command-line arguments because arguments are commonly
visible in process listings and shell history. Prefer:

```bash
export MY_PROVIDER_KEY="..."
llm-compat run --provider custom --base-url https://llm.example.com/v1 \
  --api-key-env MY_PROVIDER_KEY --model example-chat
```

Reports and terminal output omit credentials, but they may still contain model names,
short response previews, latency data, error details, and endpoint hostnames. Review a
report before publishing it as a CI artifact.

## Out of scope

The following are outside LLM Compat's control and generally not treated as
vulnerabilities in this project:

- A provider logging, retaining, or mishandling requests sent to it.
- A compromised endpoint or local machine.
- Denial of service caused by the user intentionally setting extreme timeouts or sample
  counts within the validated limits.
- Vulnerabilities in third-party provider services or their SDKs.

## Dependency policy

LLM Compat has no runtime dependencies. Development dependencies are pinned by lower
bounds in `pyproject.toml` and installed in CI. Please report suspicious dependency
behavior as you would any other security issue.
