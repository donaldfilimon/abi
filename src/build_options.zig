//! Stub build options module for standalone execution of src/main.zig.
//! Mirrors the fields provided by the generated build_options module in the
//! regular build script. Values match the default configuration defined in
//! `build.zig`.

pub const enable_gpu = true;
pub const enable_ai = true;
pub const enable_web = true;
pub const enable_database = true;
pub const enable_network = true;
pub const enable_profiling = true;

// GPU backend toggles
pub const gpu_cuda = false;
pub const gpu_vulkan = true;
pub const gpu_stdgpu = false;
pub const gpu_metal = false;
pub const gpu_webgpu = false;
pub const gpu_opengl = false;
pub const gpu_opengles = false;
pub const gpu_webgl2 = false;

// Cache settings – not used by the CLI directly but required by the module.
pub const cache_dir = ".zig-cache";
pub const global_cache_dir: ?[]const u8 = null;

// Package metadata
pub const package_version = "0.1.0";
