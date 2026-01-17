//! GPU Feature Overview
//!
//! GPU support is provided via native implementations with automatic CPU fallback.
//! All backends are production-ready with comprehensive feature support.
//!
//! Backend Status:
//!   - CUDA: Complete with tensor core support, async D2D, device queries
//!   - Vulkan: Complete with SPIR-V generation
//!   - Metal: Complete with Objective-C runtime bindings
//!   - WebGPU: Complete with async adapter/device handling
//!   - OpenGL/ES: Complete with compute shader support
//!   - std.gpu: Complete with CPU fallback
//!   - WebGL2: Correctly returns UnsupportedBackend (no compute support)
//!
//! The `mod.zig` file aggregates the public API for these backends.
//! Runtime selection is controlled by build options (see `build_options.enable-gpu`).
//!
//! Adding a new GPU backend involves:
//!   1. Implementing the required kernel entry points.
//!   2. Registering the backend in the GPU module dispatcher.
//!   3. Updating tests under `tests` if needed.

