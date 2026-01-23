# GEMINI.md

This file gives the Gemini agent guidance on interacting with the ABI framework. It mirrors the structure of `AGENTS.md` but is tuned for Gemini‑like prompts.

## Quick Start for Gemini

```bash
zig build                             # Build the framework
zig build test --summary all          # Run all tests (regression)
zig fmt .                             # Format after edits
zig build run -- --help               # CLI help
```

Typical Gemini usage:

```
zig build run -- llm generate "Hello" --max 60
```

The example list below shows common entry points that Gemini can trigger:

| Target | Description |
|--------|-------------|
| `hello` | Hello world demo |
| `database` | Vector database workspace |
| `agent` | Runs the built‑in agent example |
| `llm` | Generates text with the local LLM |

Gemini should refer to feature flags through `-D` options when possible; runtime toggles (`--disable-llm`) modify only the running binary.

## Feature Management
Feature flags live in `build.zig` and correlate with `build_options`:

```zig
const build_options = struct{
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    // ... add others as needed
};
```

If a new feature is added, a stub must exist to maintain API parity.

## Gotchas for Gemini
* **Stub‑Real Sync** – Changes to `src/<feature>/mod.zig` must be propagated to its `stub.zig`. The CI parity checker will report mismatches.
* **GPU Backends** – Use `-Dgpu-backend=auto` or `-Dgpu-backend=cuda,vulkan` and remember it is a comma‑separated list.
* **WASM Compatibility** – When compiling for WebAssembly, the `database`, `network`, and `gpu` modules auto‑disable.
* **Compile‑time Flags** – `-D` options modify the compiled binary. `--enable-…` only toggles runtime behaviour.

★ *Tip:* After modifications run `zig fmt .` followed by `zig build test --summary all` to verify nothing broke.

