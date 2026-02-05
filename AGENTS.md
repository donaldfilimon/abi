# AGENTS.md

Baseline guidance for any AI agent working in the ABI Framework repository.
Read this first. `CLAUDE.md` adds deep details, `GEMINI.md` is a condensed quick reference.

---

## Prerequisites

| Requirement | Value |
|-------------|-------|
| Zig version | `0.16.0-dev.2471+e9eadee00` or later |
| Entry point | `src/abi.zig` |
| Package version | `0.4.0` |

```bash
# Verify Zig version
zig version

# If using zvm
export PATH="$HOME/.zvm/bin:$PATH"
zvm use master
```

The codebase uses Zig 0.16 APIs throughout. Earlier versions will not compile.

---

## Before Making Changes

Always run these commands first:

```bash
git status      # Check for uncommitted work
git diff --stat # Understand scope of existing changes
```

If large or unclear diffs exist, ask about their status before proceeding.
Avoid reverting unrelated changes.
Use package managers to add new dependencies.

---

## Project Structure

```
src/
├── abi.zig              # Public API module root
├── api/                 # Entry points (main.zig)
├── core/                # Framework orchestration and config
├── features/            # Feature modules (ai, gpu, database, network, web, observability)
└── services/            # Shared infrastructure (runtime, platform, shared, tests, etc.)
```

Key rules:
- Public API imports use `@import("abi")` (avoid direct file paths).
- Nested modules import dependencies via their parent `mod.zig`.
- Feature modules must keep `mod.zig` and `stub.zig` in sync.
- Integration tests live in `services/tests/`; unit tests live alongside code as `*_test.zig`.

---

## Essential Commands

| Command | Purpose |
|---------|---------|
| `zig build` | Build the project |
| `zig build test --summary all` | Run full test suite |
| `zig test src/path/to/file.zig --test-filter "pattern"` | Run focused tests |
| `zig fmt .` | Format code (required after edits) |
| `zig build lint` | Check formatting (CI uses this) |
| `zig build cli-tests` | CLI smoke tests |
| `zig build validate-flags` | Validate all feature flag combinations compile |
| `zig build full-check` | Format + tests + flag validation + CLI smoke tests |
| `zig build benchmarks` | Performance validation |
| `zig build run -- --help` | Run CLI help |

---

## Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false
zig build -Dgpu-backend=vulkan,cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI agent system |
| `-Denable-llm` | true | Local LLM inference (requires AI) |
| `-Denable-vision` | true | Vision/image processing (requires AI) |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | Web/HTTP support |
| `-Denable-profiling` | true | Metrics/tracing |
| `-Denable-mobile` | false | Mobile cross-compilation |

GPU backends: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`, `opengl`, `fpga`.

---

## Module Pattern (Critical)

| File | When Used | Returns |
|------|-----------|---------|
| `mod.zig` | Feature enabled at build time | Real implementation |
| `stub.zig` | Feature disabled | `error.<Feature>Disabled` |

Both files must export identical public signatures.

```zig
// src/features/feature/stub.zig
pub fn doOperation() !void {
    return error.FeatureDisabled;
}
```

Verify both builds compile:

```bash
zig build -Denable-<feature>=true
zig build -Denable-<feature>=false
```

---

## Zig 0.16 API Patterns

| Old (0.15) | New (0.16) |
|------------|------------|
| `std.fs.cwd()` | `std.Io.Dir.cwd()` |
| `std.time.Instant.now()` | `std.time.Timer.start()` |
| `std.time.sleep()` | `abi.shared.time.sleepMs()` / `sleepNs()` (preferred) |
| `list.init()` | `list.empty` (ArrayListUnmanaged) |
| `@tagName(x)` in print | `{t}` format specifier |

Notes:
- I/O operations must use `std.Io.Threaded.init()` and its `io` handle.
- Use `std.Io.Clock.Duration.sleep()` only when you need raw clock access.
- HTTP server init uses `&reader.interface` and `&writer.interface`.
- Reserved keywords must be escaped (example: `result.@"error"`).

Example I/O setup:

```zig
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io,
    path,
    allocator,
    .limited(10 * 1024 * 1024),
);
```

---

## Code Style

| Rule | Convention |
|------|------------|
| Indentation | 4 spaces, no tabs |
| Line length | Under 100 characters |
| Types | `PascalCase` |
| Functions/Variables | `camelCase` |
| Errors | `*Error` names and specific error sets |
| Config structs | `*Config` |
| Imports | Explicit only (no `usingnamespace`) |
| Cleanup | Prefer `defer` / `errdefer` |
| Arrays | `std.ArrayListUnmanaged` with `.empty` |

---

## Testing

| Command | Purpose |
|---------|---------|
| `zig build test --summary all` | Full test suite |
| `zig test src/services/tests/integration/mod.zig` | Integration tests |
| `zig test src/services/tests/stress/mod.zig` | Stress tests |
| `zig test file.zig --test-filter "pattern"` | Filtered tests |

Parity tests (`src/services/tests/parity/`) use `DeclSpec` to verify stub modules
match real modules -- checking declaration kind (function vs type) and sub-declarations,
not just name existence.

Skip hardware-gated tests with `error.SkipZigTest`:

```zig
test "gpu operation" {
    const gpu = initGpu() catch return error.SkipZigTest;
    defer gpu.deinit();
}
```

---

## Post-Edit Checklist

```bash
zig fmt .
zig build test --summary all
zig build lint
```

---

## References

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Comprehensive reference and deep examples |
| `GEMINI.md` | Quick reference for Gemini |
| `CONTRIBUTING.md` | Development workflow |
| `docs/README.md` | Documentation system |
| `SECURITY.md` | Security practices |
| `DEPLOYMENT_GUIDE.md` | Production deployment |
| `PLAN.md` | Development roadmap |
