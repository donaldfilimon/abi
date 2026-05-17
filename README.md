# ABI Framework

ABI is a **Zig 0.17.0-dev.304+9787df942** framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Quick Start
```bash
./build.sh --bootstrap  # Setup toolchain and build
./build.sh check        # Run full validation gate
```

## Current Status

- `src/features/ai/streaming/server/openai.zig`: Streaming implementation is functional.
- Documentation: `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` provide AI assistant guidance.
- Build: `./build.sh check` runs the full validation gate (tests + lint + mod/stub parity).

See [docs/index.md](docs/index.md) for architecture, onboarding, and development guides.
