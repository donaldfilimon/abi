# ABI Docs

`docs/` contains both hand-maintained documentation and generated artifacts.
The important distinction is whether a file should be edited directly or changed
through the docs generator.

## Maintained docs

These files are edited directly:

| Path | Purpose |
|------|---------|
| `docs/README.md` | Docs tree map and generation workflow |
| `docs/FAQ-agents.md` | Repo workflow FAQ for contributors and code agents |
| `docs/guides/cursor_rules.md` | Cursor-specific repo rules |
| `docs/ZIG_MACOS_LINKER_RESEARCH.md` | Darwin linker failure analysis and supported workarounds |
| `docs/ABI_WDBX_ARCHITECTURE.md` | Semantic-store architecture notes and terminology |

## Generated docs

These directories are generator-owned:

| Path | Generated from |
|------|----------------|
| `docs/api/` | `zig build gendocs` |
| `docs/plans/` | `zig build gendocs` |
| `docs/_docs/` | `zig build gendocs` |
| `docs/api-app/` | `zig build gendocs` unless `--no-wasm` is used |

Do not hand-edit generated Markdown unless you are intentionally patching a
generated artifact and capturing that as a temporary exception. Canonical fixes
belong in `tools/gendocs/`.

## Workflow

### Generate docs

```bash
zig build gendocs
```

### Check docs without writing

```bash
zig build gendocs -- --check --no-wasm --untracked-md
zig build check-docs
```

### CLI wrapper

```bash
abi gendocs
abi gendocs --check
abi gendocs --check --no-wasm --untracked-md
```

## When to edit what

- Change `README.md`, guides, or policy docs directly when the workflow or public usage changes.
- Change `tools/gendocs/` templates or renderers when generated docs are stale or structurally wrong.
- Refresh the CLI registry with `zig build refresh-cli-registry` after CLI command changes.
- Keep docs and validation instructions aligned with the pinned Zig version in `.zigversion`.

## Related references

- [`README.md`](../README.md)
- [`AGENTS.md`](../AGENTS.md)
- [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- [`CLAUDE.md`](../CLAUDE.md)
