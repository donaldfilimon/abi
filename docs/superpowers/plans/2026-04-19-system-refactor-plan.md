# System-Wide Modularization Implementation Plan

**Goal:** Decouple key subsystems (`security/password.zig`, `gpu/device.zig`, `ai/training/llm_trainer.zig`) into registry-compatible, modular components.

## Tasks

### Task 1: Refactor `src/foundation/security/password.zig`
- **Goal:** Modularize password handling.
- **Plan:** Abstract crypto providers into a registry, allowing pluggable algorithms.
- **Files:** Modify `src/foundation/security/password.zig`, `src/foundation/security/mod.zig`.

### Task 2: Refactor `src/features/gpu/device.zig`
- **Goal:** Decouple device selection.
- **Plan:** Introduce device registry for backend discovery (Metal, CUDA, Vulkan).
- **Files:** Modify `src/features/gpu/device.zig`, `src/features/gpu/mod.zig`.

### Task 3: Refactor `src/features/ai/training/llm_trainer.zig`
- **Goal:** Decouple training pipelines.
- **Plan:** Modularize optimizer, scheduler, and loss functions into a registry-driven pipeline.
- **Files:** Modify `src/features/ai/training/llm_trainer.zig`, `src/features/ai/training/mod.zig`.

---
