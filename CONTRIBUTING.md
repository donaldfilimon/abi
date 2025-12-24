# Contributing

Thanks for helping improve ABI. Keep changes focused, documented, and tested.

## Development Setup
```bash
git clone <repo>
cd abi
zig build
zig build test
zig fmt .
```

## Workflow
1. Create a focused branch.
2. Make changes with clear scope.
3. Run `zig build` and `zig build test`.
4. Format with `zig fmt .`.
5. Update docs for public API changes.

## Style
- 4 spaces, no tabs, lines under 100 chars.
- PascalCase types, snake_case functions/variables.
- Explicit imports only (no `usingnamespace`).
- Use `!` return types and specific error enums.
- Prefer `defer`/`errdefer` for cleanup.

## Testing
- Unit coverage lives in library tests and `tests/mod.zig`.
- New features must include tests or clear justification.
