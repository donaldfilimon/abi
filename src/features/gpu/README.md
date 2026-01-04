//! GPU Feature Overview
//!
//! GPU support is provided via fallback runtimes that simulate kernels on
//! CUDA, Vulkan, Metal, and WebGPU. The `mod.zig` file aggregates the public API
//! for these backends. Runtime selection is controlled by build options (see
//! `build_options.enable-gpu`).
//!
//! Adding a new GPU backend involves:
//!   1. Implementing the required kernel entry points.
//!   2. Registering the backend in the GPU module dispatcher.
//!   3. Updating tests under `tests` if needed.
