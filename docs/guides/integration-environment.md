---
title: Integration Environment Contract
purpose: Defines what CI and local environments must provide for integration gates
last_updated: 2026-03-16
criterion: Integration Gates v1 — Unblock Criterion C
---

# Integration Environment Contract

This document specifies the environment requirements for running ABI's
integration test gates. It covers authoritative CI, local development, and
degraded-mode configurations.

For build commands and architecture details see [CLAUDE.md](../../CLAUDE.md).
For the verification gate table see [AGENTS.md](../../AGENTS.md).
For the integration gates plan see [docs/plans/integration-gates-v1.md](../plans/integration-gates-v1.md).

---

## CI Environment (Linux, authoritative)

The CI environment is the single source of truth for gate pass/fail.

### Zig toolchain

| Requirement | Value |
|-------------|-------|
| Version | Must exactly match `.zigversion` (`0.16.0-dev.2905+5d71e3051`) |
| Source | Host-built or official prebuilt tarball for `x86_64-linux` |
| PATH | `zig` binary must be first on `$PATH`; no version shims |

### System tools

| Tool | Purpose | Required |
|------|---------|----------|
| `git` | Source checkout, version detection | Yes |
| `sh` / `bash` | Script execution (`run_build.sh`, CI glue) | Yes |
| C system headers | Zig cross-compilation cache (libc stubs) | Provided by base image |

### Network access

Network is **not required** for compilation or unit tests. It is required
only for provider integration tests that contact external services (cloud
connectors, AI provider APIs, Discord bot).

### Environment variables

All API keys are **optional**. Tests that require a missing key must skip
gracefully rather than fail the gate.

| Variable | Used by | Required |
|----------|---------|----------|
| `ABI_OPENAI_API_KEY` | AI provider connector tests (OpenAI) | No |
| `ABI_ANTHROPIC_API_KEY` | AI provider connector tests (Anthropic) | No |
| `ABI_OLLAMA_HOST` | Local LLM inference tests | No |
| `ABI_OLLAMA_MODEL` | Local LLM inference tests | No |
| `ABI_HF_API_TOKEN` | Hugging Face model hub tests | No |
| `DISCORD_BOT_TOKEN` | Messaging integration tests | No |

### Test gates

These are the commands CI must execute, in order. All must pass for the
integration gate to be green.

| Gate | Command | Scope |
|------|---------|-------|
| Format | `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` | Style conformance |
| Flag validation | `zig build validate-flags` | Feature flag combo sanity |
| Unit tests | `zig build test --summary all` | Primary test suite |
| Feature tests | `zig build feature-tests --summary all` | Comptime feature coverage |
| Full check | `zig build full-check` | Pre-commit composite gate |
| CLI tests | `zig build cli-tests` | CLI command integration |

### GPU backend on CI

CI Linux runners typically lack a physical GPU. Use one of:

- `-Dgpu-backend=none` to skip GPU tests entirely
- `-Dgpu-backend=stdgpu` for the software-only fallback backend
- `-Dfeat-gpu=false` to disable the entire GPU feature module

Auto-detection on Linux selects `cuda, vulkan, opengl, stdgpu` by default
(see `build/gpu.zig`). Override explicitly in CI to avoid missing-driver
failures.

---

## Local Environment (Darwin / macOS)

### Known linker limitation

On **Darwin 25+ / macOS 26+**, the stock prebuilt Zig (`0.16.0-dev.2905`)
fails at the linker stage before `build.zig` runs. This is a host toolchain
issue, not an ABI bug.

**Resolution options (pick one):**

1. **Host-built Zig** matching `.zigversion` — full gate support
2. **`./tools/scripts/run_build.sh`** — two-pass workaround that relinks
   the build runner with Apple's `/usr/bin/ld` (see `tools/scripts/run_build.sh`)
3. **Fallback evidence** — when neither option above is available, record:
   - `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`
   - `./tools/scripts/run_build.sh typecheck --summary all`

   This is fallback evidence only, not a replacement for `zig build full-check`.

**Never** set `use_lld = true` on macOS — LLD has zero Mach-O support.

### GPU backend on macOS

Auto-detection selects backends in this order (see `build/gpu.zig`):

| Condition | Backends selected |
|-----------|-------------------|
| Metal frameworks available | `metal, vulkan, opengl, stdgpu` |
| Metal frameworks unavailable | `vulkan, opengl, stdgpu` |

Override with `-Dgpu-backend=metal`, `-Dgpu-backend=none`, etc.

### Running a subset of tests

When the full build is linker-blocked, you can still validate individual
files:

```bash
# Single-file standalone test (no build system)
zig test src/path/to/file.zig -fno-emit-bin

# Format check (always works, even when linker is broken)
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/

# Build runner workaround
./tools/scripts/run_build.sh test --summary all
./tools/scripts/run_build.sh typecheck --summary all
```

---

## Degraded Mode

Not every environment has network, API keys, or a GPU. The table below
defines what is testable in each degraded configuration.

### Without network access

| Works | Does not work |
|-------|---------------|
| All compilation | Provider connector live tests |
| All unit tests | Cloud integration tests |
| Feature gate tests | External model download |
| CLI command tests (local) | Discord bot tests |
| GPU compute (local backends) | |

### Without API keys

| Works | Does not work |
|-------|---------------|
| All compilation (stubs compile without keys) | Live calls to OpenAI, Anthropic, HuggingFace |
| All unit tests | Provider round-trip integration tests |
| Feature gate tests | |
| CLI structural tests | |

Setting API key env vars to empty strings is **not** equivalent to unsetting
them. Tests should check for non-empty values.

### Without GPU

| Option | Effect |
|--------|--------|
| `-Dfeat-gpu=false` | Entire GPU feature module replaced by stubs |
| `-Dgpu-backend=none` | GPU feature enabled but no backends compiled |
| `-Dgpu-backend=stdgpu` | Software-only fallback; no hardware required |

The `stdgpu` backend is always available on all platforms and requires no
drivers or hardware.

### Minimal testing profile

To run the narrowest useful test surface (no GPU, no cloud, no mobile):

```bash
zig build test --summary all \
  -Dfeat-gpu=false \
  -Dfeat-cloud=false \
  -Dfeat-mobile=false
```

All 27 feature flags default to `true` except `feat_mobile` (defaults to
`false`). Flags derived from a parent inherit the parent default:
`feat_explore`, `feat_llm`, `feat_vision`, and `feat_training` inherit from
`feat_ai`; `feat_reasoning` also inherits from `feat_ai`.

For the full flag list and validation rules, see `build/options.zig` and
`build/flags.zig`.

---

## Platform-specific GPU auto-detection summary

Reference: `build/gpu.zig` and `build/gpu_policy.zig`.

| Platform | Auto-detected backends |
|----------|----------------------|
| Linux | cuda, vulkan, opengl, stdgpu |
| macOS (Metal available) | metal, vulkan, opengl, stdgpu |
| macOS (no Metal) | vulkan, opengl, stdgpu |
| Windows | cuda, vulkan, opengl, stdgpu |
| Windows (safe mode) | stdgpu |
| Android | vulkan, opengles, stdgpu |
| iOS / tvOS | metal, opengles, stdgpu |
| Web (WASI/Emscripten) | webgpu, webgl2 |
| FreeBSD | vulkan, opengl, stdgpu |
| Other Unix (NetBSD, OpenBSD, etc.) | opengl, stdgpu |
| Freestanding | stdgpu |
