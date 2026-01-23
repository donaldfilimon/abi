# Vulkan Backend Consolidation Plan

**Status** – Plan created; actual consolidation work pending.

**Goal** – Collapse the four separate Vulkan backend source files
`vulkan_types.zig`, `vulkan_init.zig`, `vulkan_pipelines.zig`, and
`vulkan_buffers.zig` into a **single** public module `src/gpu/backends/vulkan.zig`.
All external code already imports `backends/vulkan.zig`, so the change is
internal‑only but reduces file‑system clutter and improves discoverability.

## Rationale

* The current split is useful during early development, but the ABI
  framework is now stable and the Vulkan backend is feature‑complete.
* Consolidation avoids the need to maintain multiple `pub const` re‑exports
  and keeps the public API surface in one place.
* Simpler navigation for contributors and for IDEs that struggle with deep
  module trees.

## Scope

* **Include** – All publicly‑exposed symbols from the four files. Private
  helpers stay inside the new `vulkan` module.
* **Exclude** – Tests, examples, or any code that imports the sub‑modules
  directly (there are none in the repository).

## Steps

1. **Create staging sections** – In `vulkan.zig` add `pub const Types = struct { … }`
   containing the contents of `vulkan_types.zig` (minus the top‑level `const std`
   import which already exists).
2. **Embed init logic** – Append the body of `vulkan_init.zig` inside a new
   `pub const Init = struct { … }` block. Adjust any references to `std` or
   `types` to use the enclosing module (`@import("std")` and `Types`).
3. **Embed pipeline logic** – Repeat for `vulkan_pipelines.zig` as
   `pub const Pipelines = struct { … }`.
4. **Embed buffer logic** – Repeat for `vulkan_buffers.zig` as
   `pub const Buffers = struct { … }`.
5. **Re‑export public symbols** – At the top of `vulkan.zig` add `pub usingnamespace
   Types;` etc., or individually forward the symbols that the rest of the code
   expects (e.g., `pub const VulkanError = Types.VulkanError;`).
6. **Update internal imports** – Replace any `@import("vulkan_init.zig")` or
   similar within the code‑base with `@import("vulkan.zig")` (currently only
   `src/gpu/device.zig` imports the consolidated module, so no changes are
   needed).
7. **Delete the four original files** – `vulkan_types.zig`, `vulkan_init.zig`,
   `vulkan_pipelines.zig`, `vulkan_buffers.zig`.
8. **Run formatter** – `zig fmt .` to keep style consistent.
9. **Run full test suite** – `zig build test --summary all` to verify
   compilation and runtime behaviour.
10. **Update documentation** – Add a short note in `README.md` under the
    "GPU Backends" section that the Vulkan backend is now a single file.

## Acceptance Criteria

* `src/gpu/backends/vulkan.zig` compiles and provides the same public API as before.
* All existing tests pass.
* No import errors throughout the repository.
* `zig fmt .` reports no style violations.
