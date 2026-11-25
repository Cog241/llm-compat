## Summary

Describe the problem and the smallest change that resolves it.

## Tests

- [ ] `pytest` passes locally without network access or API keys.
- [ ] New behavior has offline test coverage using `tests/mock_server.py`.
- [ ] `ruff check .` passes.

## Compatibility

- [ ] Default checks, report schema semantics, and existing CLI behavior are preserved.
- [ ] Any user-visible change is documented in `README.md`, `docs/`, and `CHANGELOG.md`.
- [ ] No live provider, agent, or research requests are part of the automated suite.

## Safety

- [ ] I removed API keys, account identifiers, private endpoint URLs, and unsanitized
      provider output from this pull request.
- [ ] Generated experiment output and virtual environments are not committed.

## Related issues

Closes #
