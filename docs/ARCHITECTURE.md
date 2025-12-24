# Architecture

ABI is organized into four primary layers:

1. **Core**: foundational types, diagnostics, and utilities.
2. **Features**: AI, database, GPU, web, monitoring, connectors.
3. **Framework**: lifecycle, feature gating, and plugin orchestration.
4. **Shared**: cross-cutting helpers (logging, platform, SIMD).

## Runtime Flow
- `FrameworkOptions` define feature toggles.
- `RuntimeConfig` is derived and passed into the framework runtime.
- Features initialize on demand based on enabled flags.

## Design Principles
- Explicit exports (no `usingnamespace`).
- Build-time feature flags with safe defaults.
- Composition over inheritance; minimal hidden state.
