# Contributing

ABI accepts focused, reviewable changes.

## Workflow

1. Use the Zig toolchain pinned in [.zigversion](.zigversion).
2. Review [AGENTS.md](AGENTS.md), [tasks/todo.md](tasks/todo.md), and [tasks/lessons.md](tasks/lessons.md) before non-trivial work.
3. Run the mandatory tri-CLI consensus flow for non-trivial work before implementation.
4. Prefer small commits with a clear validation trail.
5. Update documentation when public behavior, commands, or file layout changes.

## Validation

All contributors MUST run the standard validation suite before submitting a pull request. Refer to the canonical [AGENTS.md](AGENTS.md) for the latest list of build, lint, and test commands.

The primary confidence gate is:
```bash
zig build full-check
```

If a command is blocked by the local environment (e.g., macOS 26+ linker issues), record the exact failure in your PR notes and ensure the bypass mechanisms are properly utilized.

## Related Guides

- [AGENTS.md](AGENTS.md) — Canonical repo workflow contract (single source of truth for build/test commands)
- [docs/FAQ-agents.md](docs/FAQ-agents.md) — Detailed agent guidance and code style FAQ
- [CLAUDE.md](CLAUDE.md) — Quick reference wrapper for Claude Code
- [SECURITY.md](SECURITY.md) — Responsible disclosure guidance
- [docs/README.md](docs/README.md) — Documentation structure and generated outputs
