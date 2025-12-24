# ABI Framework

Modern Zig framework for modular AI services, vector search, and systems tooling.

## Highlights
- AI agent runtime, training pipelines, and data structures
- Vector database helpers (WDBX) with unified API
- GPU backends (optional, in progress)
- Web utilities (HTTP client/server helpers, weather helper)
- Monitoring (logging, metrics, tracing, profiling)

## Build
```bash
zig build
zig build test
zig build -Doptimize=ReleaseFast
zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true
```

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

## Project Layout
```
abi/
├── lib/        # Core library sources
├── tools/      # CLI entrypoint
├── tests/      # Smoke tests
└── docs/       # Architecture + guides
```

## CLI
The bundled CLI is intentionally minimal (help + version) and serves as a thin  
entrypoint for embedded usage.

```bash
zig build run -- --help
zig build run -- --version
```

## Documentation
- docs/guides/GETTING_STARTED.md
- docs/ARCHITECTURE.md
- docs/PROJECT_STRUCTURE.md
- docs/OBSERVABILITY.md

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
See CONTRIBUTING.md for development workflow and style guidelines.
