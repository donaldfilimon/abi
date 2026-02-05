# CLAUDE.md

Comprehensive guidance for Claude Code working with the ABI Framework. Read
`AGENTS.md` first for baseline rules; this file adds deeper details and examples.
`GEMINI.md` is a condensed quick reference.

| | |
|---|---|
| **Version** | 0.4.0 |
| **Entry Point** | `src/abi.zig` |
| **Zig Required** | 0.16.x (`0.16.0-dev.2471+e9eadee00` or later) |
| **Version File** | `.zigversion` (pins exact Zig version) |

## Quick Start

| Command | Purpose |
|---------|---------|
| `zig build` | Build the project |
| `zig build test --summary all` | Run full test suite |
| `zig fmt .` | Format code (required after edits) |
| `zig build run -- --help` | CLI help |
| `zig test src/file.zig --test-filter "pattern"` | Focused tests |

---

## Critical Gotchas

| Category | Issue | Solution |
|----------|-------|----------|
| Zig 0.16 | File system ops | Use `std.Io.Dir.cwd()` not `std.fs.cwd()` |
| Zig 0.16 | Timer API | Use `std.time.Timer.start()` not `Instant.now()` |
| Zig 0.16 | ArrayListUnmanaged | Use `.empty` not `.init()` |
| Zig 0.16 | Format specifiers | Use `{t}` for errors/enums |
| Feature | Stub/real sync | Update `mod.zig` and `stub.zig` together |
| Build | `--test-filter` syntax | Use `zig test file.zig --test-filter` |
| Build | build.zig file checks | Use `pathExists()` helper |
| GPU | Backend conflicts | Prefer one primary backend |
| WASM | Limitations | `database`, `network`, `gpu` auto-disabled |
| Sleep | Cross-platform | Prefer `abi.shared.time.sleepMs()` / `sleepNs()` |
| HTTP | Server init | Use `&reader.interface` and `&writer.interface` |

---

## Zig 0.16 API Patterns

### I/O Backend Initialization (Critical)

```zig
const std = @import("std");

var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty, // .empty for library, init.environ for CLI
});
defer io_backend.deinit();
const io = io_backend.io();

const content = try std.Io.Dir.cwd().readFileAlloc(
    io,
    path,
    allocator,
    .limited(10 * 1024 * 1024),
);
defer allocator.free(content);
```

### Other Changes

```zig
// Error/enum formatting: use {t}
std.debug.print("Error: {t}, State: {t}", .{err, state});

// ArrayListUnmanaged
var list = std.ArrayListUnmanaged(u8).empty;

// Timing
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();

// Sleep (preferred for cross-platform)
const abi = @import("abi");
abi.shared.time.sleepMs(10);

// HTTP server init
var server: std.http.Server = .init(
    &connection_reader.interface,
    &connection_writer.interface,
);
```

---

## Architecture

Flat domain structure with unified configuration. Each feature has `mod.zig`
(real) and `stub.zig` (feature-gated placeholder).

```
src/
├── abi.zig              # Public API module root
├── api/                 # Entry points
│   └── main.zig         # CLI entrypoint fallback
├── core/                # Framework orchestration and config
│   ├── config/          # Unified config (Config + Builder + per-feature)
│   ├── framework.zig    # Lifecycle states, builder pattern
│   ├── flags.zig        # Feature flags
│   └── registry/        # Feature registry (comptime, runtime, dynamic)
├── features/            # Feature modules
│   ├── ai/              # Agents, LLM, streaming, training, vision
│   ├── gpu/             # GPU acceleration (backends, kernels, codegen)
│   ├── database/        # Vector database (WDBX with HNSW/IVF-PQ)
│   ├── network/         # Distributed compute, Raft consensus
│   ├── observability/   # Metrics, tracing, system info
│   └── web/             # Web/HTTP server support
└── services/            # Shared infrastructure
    ├── runtime/         # Task execution, concurrency, scheduling
    ├── platform/        # Platform abstraction layer
    ├── shared/          # Utilities, security, SIMD
    ├── connectors/      # API connectors (OpenAI, Ollama, Anthropic, etc.)
    ├── cloud/           # Cloud provider integrations
    ├── ha/              # High availability (failover, replication)
    ├── tasks/           # Task management system
    └── tests/           # Test suite (chaos, e2e, integration, stress)
```

Import rules:
- Public API imports use `@import("abi")`.
- Nested modules import via their parent `mod.zig`.

### Granular Stub Pattern

AI submodules have individual `stub.zig` files for fine-grained feature gating:

```
src/features/ai/
├── mod.zig, stub.zig       # Top-level AI
├── agents/mod.zig, stub.zig
├── llm/mod.zig, stub.zig
├── vision/mod.zig, stub.zig
├── streaming/mod.zig, stub.zig
├── training/mod.zig, stub.zig
├── models/mod.zig, stub.zig
├── memory/mod.zig, stub.zig
├── rag/mod.zig, stub.zig
├── embeddings/mod.zig, stub.zig
├── orchestration/mod.zig, stub.zig
├── multi_agent/mod.zig, stub.zig
├── personas/mod.zig, stub.zig
└── explore/mod.zig, stub.zig
```

When modifying any of these, update **both** `mod.zig` and `stub.zig`.

---

## Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=false
zig build -Dgpu-backend=vulkan,cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | AI agent system |
| `-Denable-llm` | true | Local LLM inference |
| `-Denable-vision` | true | Vision/image processing |
| `-Denable-gpu` | true | GPU acceleration |
| `-Denable-database` | true | Vector database |
| `-Denable-network` | true | Distributed compute |
| `-Denable-web` | true | Web/HTTP support |
| `-Denable-profiling` | true | Metrics/tracing |
| `-Denable-mobile` | false | Mobile cross-compilation |

GPU backends: `auto`, `none`, `cuda`, `vulkan`, `metal`, `stdgpu`, `webgpu`,
`opengl`, `fpga`.

---

## Common Workflows

```bash
# Build + test
zig build test --summary all

# Single file test
zig test src/services/runtime/engine/engine.zig --test-filter "pattern"

# Full check
zig build full-check

# CLI smoke tests
zig build cli-tests
```

---

## Configuration System

Unified `Config` with a builder pattern:

```zig
const abi = @import("abi");

var builder = abi.config.Builder.init(allocator);
const config = builder
    .withAiDefaults()
    .withGpuDefaults()
    .withDatabaseDefaults()
    .build();
```

---

## CLI Reference (Condensed)

Key command groups:
- `abi agent`, `abi llm`, `abi model` (AI/LLM)
- `abi db`, `abi embed` (database, embeddings)
- `abi gpu`, `abi gpu-dashboard` (GPU)
- `abi train` (training)
- `abi bench` (benchmarks)
- `abi config`, `abi system-info` (system/config)

Full details in `docs/content/cli.html`.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ABI_OPENAI_API_KEY` | OpenAI connector |
| `ABI_ANTHROPIC_API_KEY` | Claude connector |
| `ABI_OLLAMA_HOST` | Ollama host |
| `ABI_OLLAMA_MODEL` | Default Ollama model |
| `ABI_HF_API_TOKEN` | HuggingFace token |
| `ABI_MASTER_KEY` | Secrets encryption (production) |

See `README.md` for the full list.

---

## Language Bindings

Bindings live in `bindings/`. Build the C shared library first:

```bash
cd bindings/c && zig build
# Outputs: bindings/c/zig-out/lib/libabi.{dylib,so}
```

| Language | Location | Setup |
|----------|----------|-------|
| C | `bindings/c/` | `#include "abi.h"` |
| Python | `bindings/python/` | `pip install -e .` (depends on C lib) |
| Go | `bindings/go/` | `go get` (depends on C lib) |
| JavaScript | `bindings/js/` | Node.js FFI wrapper |
| Rust | `bindings/rust/` | Rust FFI crate |

Set library path: `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) to `bindings/c/zig-out/lib`.

---

## Test Infrastructure

- Unit tests live alongside code as `*_test.zig`.
- Integration tests are under `tests/` (e2e, stress, chaos, integration).
- Skip hardware-gated tests with `error.SkipZigTest`.

---

## Quick File Navigation

| Task | Key Files |
|------|-----------|
| Public API | `src/abi.zig` |
| Feature module | `src/features/<feature>/mod.zig`, `src/features/<feature>/stub.zig` |
| CLI commands | `tools/cli/commands/` |
| CLI main | `tools/cli/main.zig`, `tools/cli/mod.zig` |
| CLI utilities | `tools/cli/utils/`, `tools/cli/tui/` |
| Build config | `build.zig`, `build.zig.zon` |
| Tests entry | `src/services/tests/mod.zig` |
| Docs site | `docs/` |
| Cloud integrations | `src/services/cloud/` |
| High availability | `src/services/ha/` |
| Connectors | `src/services/connectors/` |

---

## References

- `AGENTS.md` - Baseline rules
- `GEMINI.md` - Quick reference
- `CONTRIBUTING.md` - Workflow
- `docs/README.md` - Docs system
- `SECURITY.md` - Security practices
- `DEPLOYMENT_GUIDE.md` - Production deployment
- `PLAN.md` - Roadmap
