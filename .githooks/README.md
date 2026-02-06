# Git Hooks for ABI Framework

This directory contains git hooks that enforce project conventions automatically.

## Setup

Run this once to enable the hooks:

```bash
git config core.hooksPath .githooks
```

Then make sure the hooks are executable (they should already be if cloned on
macOS/Linux, but just in case):

```bash
chmod +x .githooks/pre-commit .githooks/commit-msg
```

## Hooks

### pre-commit

Runs automatically before every commit. Performs three checks:

| Check | Behavior | Speed |
|-------|----------|-------|
| **Credential guard** | Blocks `.env`, `.pem`, `.key`, `.p12`, `.pfx`, `.jks`, `credentials.json`, and files with "secret" in the name | Instant |
| **Zig formatting** | Runs `zig fmt --check .` on staged `.zig` files | ~1-2s |
| **Stub parity warning** | If you stage a `mod.zig`, warns if the sibling `stub.zig` is not also staged | Instant |

The credential guard and formatting check are **hard failures** (commit is
blocked). The stub parity check is a **warning only** (commit proceeds).

### commit-msg

Validates that commit messages follow the project's conventional commit format:

```
<type>(<scope>): <summary>
<type>: <summary>
```

**Allowed types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `merge`

**Examples:**
```
feat(ai): add streaming response support
fix: resolve memory leak in GPU backend
docs: update CLAUDE.md with new API patterns
refactor(gpu): simplify kernel dispatch logic
test(database): add HNSW index stress tests
chore: update build.zig for Zig 0.16
merge: claude/feature-branch
```

Auto-generated merge commits (starting with "Merge") are always allowed.

## Bypassing Hooks

If you need to bypass the hooks for a specific commit (e.g., work-in-progress):

```bash
git commit --no-verify -m "chore: wip"
```

## Disabling Hooks

To revert to default git behavior:

```bash
git config --unset core.hooksPath
```

## Troubleshooting

**"zig not found"**: The pre-commit hook requires `zig` to be in your PATH.
If you use a version manager, make sure it is configured in your shell profile.

**Permission denied**: Run `chmod +x .githooks/*` to make hooks executable.

**Hook not running**: Verify your hooks path is set: `git config core.hooksPath`
should print `.githooks`.
