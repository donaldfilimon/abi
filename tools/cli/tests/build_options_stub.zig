//! Build options stub used for standalone CLI matrix generation via `zig run`.
//!
//! The full matrix generator imports CLI command descriptors which transitively
//! import `@import("abi")`. Outside build.zig, we provide this lightweight
//! module so `src/abi.zig` can resolve `@import("build_options")`.

pub const package_version: []const u8 = "0.4.0";

pub const feat_gpu = true;
pub const feat_ai = true;
pub const feat_explore = true;
pub const feat_llm = true;
pub const feat_vision = true;
pub const feat_web = true;
pub const feat_database = true;
pub const feat_network = true;
pub const feat_profiling = true;
pub const feat_analytics = true;
pub const feat_cloud = true;
pub const feat_training = true;
pub const feat_reasoning = true;
pub const feat_auth = true;
pub const feat_messaging = true;
pub const feat_cache = true;
pub const feat_storage = true;
pub const feat_search = true;
pub const feat_mobile = false;
pub const feat_gateway = true;
pub const feat_pages = true;
pub const feat_benchmarks = true;

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

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
