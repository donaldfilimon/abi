# ABI Framework — New Engineer Onboarding

Welcome to ABI, a Zig 0.17.x/dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

For comprehensive build commands, architecture details, and conventions, see `CLAUDE.md` (the canonical reference). This guide covers first-day setup and orientation only.

---

## 1. Environment Setup

### Prerequisites

- **macOS 14+** or **Linux** (macOS 26.4+ requires special build wrapper — see below)
- **Git**
- No system Zig install needed — the project manages its own toolchain

### First-Time Setup

```bash
git clone <repo-url> && cd abi
tools/zigly --bootstrap    # Install pinned zig, install or reuse zls, symlink to ~/.local/bin, verify
```

Ensure `~/.local/bin` is on your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"  # Add to your shell profile
```

### Verify Your Setup

```bash
# macOS 26.4+ (Darwin 25.x) — you MUST use ./build.sh, not zig build directly
./build.sh test -Dfeat-gpu=false --summary all

# Linux or older macOS
zig build test --summary all
```

> **Why `build.sh`?** On macOS 26.4+, Zig's internal LLD linker can't resolve system symbols from Apple's `.tbd` files. `build.sh` relinks the build runner with Apple's `/usr/bin/ld`. This is not optional — `zig build` will fail.

### Editor Setup

`tools/zigly --link` installs ZLS (Zig Language Server) alongside Zig. Configure your editor to use `~/.local/bin/zls`. VS Code + the Zig extension works well.

---

## 2. Orientation

### What You're Working With

Everything is exposed through `src/root.zig` as `@import("abi")`. The public surface is `abi.<domain>` (e.g., `abi.gpu`, `abi.ai`, `abi.database`). See `CLAUDE.md` § Architecture for the full module map.

### The One Pattern You Must Know

Every feature uses **mod/stub comptime gating**. When you change a feature's public API, update both `mod.zig` and `stub.zig`, then run `zig build check-parity`. See `CLAUDE.md` § The Mod/Stub Pattern for details.

### Try the CLI

```bash
./build.sh cli                          # Build CLI binary
zig-out/bin/abi                         # Smart status
zig-out/bin/abi features                # List all 60 features with [+]/[-] status
zig-out/bin/abi doctor                  # Build config report
```

---

## 3. Key References

| Resource                       | Path       | Why Read It                                                  |
| ------------------------------ | ---------- | ------------------------------------------------------------ |
| `CLAUDE.md`                    | repo root  | Canonical build commands, architecture, conventions, gotchas |
| `AGENTS.md`                    | repo root  | AI agent guidance, code style, safety rules                  |
| `docs/spec/ABBEY-SPEC.md`      | docs/spec/ | Full architecture vision (Abbey-Aviva-Abi pipeline)          |
| `tasks/lessons.md`             | tasks/     | Pitfalls others have already hit — saves you time            |
| `src/core/feature_catalog.zig` | src/core/  | Source of truth for all 60 features                          |

---

## 4. Your First Day Checklist

- [ ] Run `tools/zigly --bootstrap` and verify the build passes
- [ ] Run `zig-out/bin/abi doctor` and `zig-out/bin/abi features` to see the system state
- [ ] Read `CLAUDE.md` — especially Architecture, Import Rules, and Zig 0.17 Gotchas
- [ ] Read `src/root.zig` to understand the public API surface
- [ ] Pick one feature directory (e.g., `src/features/cache/`) and read its `mod.zig`, `stub.zig`, and `types.zig` to understand the pattern
- [ ] Run `zig build check` to see the full quality gate
- [ ] Read `docs/spec/ABBEY-SPEC.md` for the full architecture vision
- [ ] Read `tasks/lessons.md` for pitfalls others have already hit
      /
