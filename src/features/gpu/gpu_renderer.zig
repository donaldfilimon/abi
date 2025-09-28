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

pub usingnamespace @import("core/gpu_renderer.zig");
