# ABI Production Hardening & Agent Integration Design

**Date:** 2026-04-28
**Status:** Approved (Approach 2)

## 1. Overview
This design covers the refactoring of `@codebase_investigator` and `@code-reviewer` into the ABI main branch and a comprehensive cleanup of the codebase for production readiness and Zig 0.17 perfection.

## 2. Agent Architecture (Option C: Hybrid)
We will implement a hybrid agent system that combines core Zig performance with Markdown-based profiling.

### 2.1 Core Zig Logic (`src/features/ai/agents/`)
- **`codebase_investigator.zig`**: A new core agent implementation. It will use direct filesystem access and AST parsing (via Zig standard library or `ovo` components) to understand the codebase.
- **`code_reviewer.zig`**: A core agent that evaluates code changes against the `AGENTS.md` and `GEMINI.md` conventions.
- **`mod.zig`**: Update `AgentRegistry` to include these new types.

### 2.2 Markdown Profiles (`zig-abi-plugin/agents/`)
- **`codebase-investigator.md`**: A profile that exposes the `codebase_investigator` tools to the CLI/MCP.
- **`code-reviewer.md`**: A profile that exposes the `code_reviewer` tools.
- Both will use the `AgentType` enum to dispatch to the core implementations.

## 3. Massive Codebase Cleanup
The goal is to shrink the file count and improve maintainability.

### 3.1 Namespace Retirement
- **Action**: Globally replace and remove all `abi.features.*` and `abi.services.*` usages.
- **New Pattern**: Use direct domain APIs (e.g., `abi.ai`, `abi.gpu`, `abi.database`).
- **Files Affected**: `src/root.zig`, `src/core/feature_catalog.zig`, and all feature `mod.zig` files.

### 3.2 File Consolidation
- **Action**: Consolidate `mod.zig`, `stub.zig`, and `types.zig` for internal/stable features into single files where the stub is a nested struct or handled via comptime logic within one file.
- **Targets**: Features that are not intended to be independently toggled by end-users in a production build.

### 3.3 Foundation Extraction
- **Action**: Identify redundant helper functions across features (e.g., JSON parsing, CLI formatting) and move them to `src/foundation/`.

## 4. Zig 0.17 Perfection
A global review of all files to enforce "perfection-level" idioms.

- **ArrayListUnmanaged**: Must use `.empty` for initialization.
- **BoundedArray**: Replace with `buffer: [N]T = undefined` and `len: usize = 0`.
- **Time**: All `std.time.milliTimestamp()` replaced by `foundation.time.unixMs()`.
- **Imports**: Ensure all path imports have explicit `.zig` extensions.
- **Trim**: `std.mem.trimRight` -> `std.mem.trimEnd`.

## 5. Production Gating
Remove all "dev-only" shortcuts.

- **Inference**: Disable "echo mode" unless explicitly requested via a `--dev` flag.
- **Auth**: Error if `ABI_JWT_SECRET` is missing (no default secret in production).
- **GPU**: Ensure fallback to CPU is explicit and logged as a warning, not a silent failure.

## 6. Verification Plan
- **Parity Check**: `zig build check-parity` must pass for all modified features.
- **Test Suite**: `zig build full-check` (macOS) or `zig build check` (Linux) must pass.
- **Agent Verification**: Manual verification of `abi chat` with the new profiles.
