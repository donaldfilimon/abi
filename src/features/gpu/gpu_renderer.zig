//! Compatibility wrapper for the modular GPU renderer.
//!
//! The legacy `src/features/gpu/gpu_renderer.zig` entrypoint historically
//! contained a monolithic implementation. The renderer has since been split
//! into smaller modules under `core/gpu_renderer/`.  To avoid churn for code
//! that still imports this legacy path we simply re-export the modern surface
//! area from the new module tree.
//!
//! Downstream code can continue to import `features.gpu.gpu_renderer` while
//! gaining access to the structured APIs provided by the new architecture.
//! Projects migrating to the modular layout are encouraged to import
//! `features.gpu.core.gpu_renderer` directly.

// Bring the core renderer symbols into this module for local convenience.
const gpu_renderer = @import("core/gpu_renderer.zig");

// Also expose the imported module publicly so downstream code importing the
// legacy path can access the modern module via `.core`.
pub const core = gpu_renderer;
