//! Stub build_options module for standalone execution of any source file that
//! expects the generated build options module. Mirrors the default values from
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

pub const cache_dir = ".zig-cache";
pub const global_cache_dir: ?[]const u8 = null;

pub const package_version = "0.1.0";
