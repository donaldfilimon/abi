# ABI Framework

ABI is a **Zig 0.17.0-dev.304+9787df942** framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Quick Start
```bash
./build.sh --bootstrap  # Setup toolchain and build
./build.sh check        # Run full validation gate
```

## Backlog Tracker

- `test/integration/e2e_llm_test.zig`: Implement full LLM pipeline test (currently validates error handling).
- `test/integration/e2e_database_test.zig`: Add in‑memory test helpers or remove placeholder.
- `src/features/ai/streaming/server/openai.zig`: Verify streaming implementation (currently functional).
- `src/features/ai/explore/query.zig`: Review placeholder patterns (already handled).
- CI workflow: Keep blocking task-marker comments in tracked source files.
- Documentation: Ensure `CLAUDE.md` includes Getting Started and CLI command list.

See [docs/index.md](docs/index.md) for architecture, onboarding, and development guides.
