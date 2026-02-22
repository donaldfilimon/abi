//! Build options stub used for standalone CLI matrix generation via `zig run`.
//!
//! The full matrix generator imports CLI command descriptors which transitively
//! import `@import("abi")`. Outside build.zig, we provide this lightweight
//! module so `src/abi.zig` can resolve `@import("build_options")`.

pub const package_version: []const u8 = "0.4.0";

pub const enable_gpu = true;
pub const enable_ai = true;
pub const enable_explore = true;
pub const enable_llm = true;
pub const enable_vision = true;
pub const enable_web = true;
pub const enable_database = true;
pub const enable_network = true;
pub const enable_profiling = true;
pub const enable_analytics = true;
pub const enable_cloud = true;
pub const enable_training = true;
pub const enable_reasoning = true;
pub const enable_auth = true;
pub const enable_messaging = true;
pub const enable_cache = true;
pub const enable_storage = true;
pub const enable_search = true;
pub const enable_mobile = false;
pub const enable_gateway = true;
pub const enable_pages = true;
pub const enable_benchmarks = true;

pub const gpu_cuda = false;
pub const gpu_vulkan = false;
pub const gpu_stdgpu = true;
pub const gpu_metal = true;
pub const gpu_webgpu = false;
pub const gpu_opengl = true;
pub const gpu_opengles = false;
pub const gpu_gl_any = true;
pub const gpu_gl_desktop = true;
pub const gpu_gl_es = false;
pub const gpu_webgl2 = false;
pub const gpu_fpga = false;
pub const gpu_tpu = false;
