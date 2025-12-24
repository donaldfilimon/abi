# ABI Framework

Modern Zig framework for modular AI services, vector search, and systems tooling.

## Highlights
- AI agent runtime, training pipelines, and data structures
- Vector database helpers (WDBX) with unified API
- GPU backends (optional, in progress)
- Web utilities (HTTP client/server helpers, weather helper)
- Monitoring (logging, metrics, tracing, profiling)

## Requirements
- Zig 0.15.2

## Build
```bash
zig build
zig build test
zig build -Doptimize=ReleaseFast
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

## Feature Flags
- `-Denable-ai`, `-Denable-gpu`, `-Denable-web`, `-Denable-database`
- `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`

## Quick Example
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

## Architecture Overview
- `src/abi.zig`: public API surface and curated re-exports
- `src/root.zig`: root module entrypoint
- `src/framework/`: runtime config, feature orchestration, lifecycle
- `src/features/`: vertical feature stacks (AI, GPU, database, web, monitoring)
- `src/shared/`: shared utilities (logging, observability, platform, utils)
- `src/internal/legacy/`: backward-compat implementations and deprecated modules

## Project Layout
```
abi/
├── src/                # Core library sources
│   ├── core/           # Core infrastructure
│   ├── features/       # Feature modules (AI, GPU, web, etc.)
│   ├── framework/      # Framework configuration and runtime
│   ├── shared/         # Shared utilities
│   └── internal/       # Legacy + experimental modules
│       └── legacy/     # Backward-compat implementations
├── build.zig           # Build graph + feature flags
└── build.zig.zon        # Zig package metadata
```

## CLI
If a CLI entrypoint is present at `tools/cli/main.zig`, it provides a thin
wrapper for embedded usage (help + version). This tree currently omits that
entrypoint; re-add it or update `build.zig` to skip the CLI build step.

```bash
zig build run -- --help
zig build run -- --version
```

## Tests
If a test root exists at `tests/mod.zig`, run:
```bash
zig build test
```
This tree currently omits `tests/`; add tests or update `build.zig` to skip the
test step.

## Connector Environment Variables
- `ABI_OPENAI_API_KEY`, `OPENAI_API_KEY`
- `ABI_OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `ABI_OPENAI_MODE` (`responses`, `chat`, or `completions`)
- `ABI_HF_API_TOKEN`, `HF_API_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
- `ABI_HF_BASE_URL` (default `https://api-inference.huggingface.co`)
- `ABI_LOCAL_SCHEDULER_URL`, `LOCAL_SCHEDULER_URL` (default `http://127.0.0.1:8081`)
- `ABI_LOCAL_SCHEDULER_ENDPOINT` (default `/schedule`)
- `ABI_OLLAMA_HOST`, `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
- `ABI_OLLAMA_MODEL` (default `llama3.2`)

## Contributing
See `CONTRIBUTING.md` for development workflow and style guidelines.
