const std = @import("std");

/// Build configuration options used throughout the project.
/// These are parsed from the command‑line options and environment
/// variables. Keeping them in a dedicated module makes the
/// build logic easier to maintain, unit test, and reuse in
/// library code.
pub const BuildOptions = struct {
    enable_gpu: bool,
    enable_ai: bool,
    enable_web: bool,
    enable_database: bool,
    enable_network: bool,
    enable_profiling: bool,
    gpu_cuda: bool,
    gpu_vulkan: bool,
    gpu_metal: bool,
    gpu_webgpu: bool,
    gpu_opengl: bool,
    gpu_opengles: bool,
    gpu_webgl2: bool,
    cache_dir: []const u8,
    global_cache_dir: ?[]const u8,
};

/// Default configuration values used when an explicit flag is not
/// provided by the user.
pub const Default = struct {
    const enable_gpu = true;
    const enable_ai = true;
    const enable_web = true;
    const enable_database = true;
    const enable_network = false;
    const enable_profiling = false;
    const gpu_cuda = false;
    const gpu_vulkan = false;
    const gpu_metal = false;
    const gpu_webgpu = false;
    const gpu_opengl = false;
    const gpu_opengles = false;
    const gpu_webgl2 = false;
    const cache_dir = "${HOME}/.cache/abi";
    const global_cache_dir = null;
};

/// Reads all build flags from the provided `std.Build` instance.
/// The function respects the defaults defined above when an option is
/// omitted.
pub fn read(b: *std.Build) BuildOptions {
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse Default.enable_gpu;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI support") orelse Default.enable_ai;
    const enable_web = b.option(bool, "enable-web", "Enable web backend") orelse Default.enable_web;
    const enable_database = b.option(bool, "enable-database", "Enable database support") orelse Default.enable_database;
    const enable_network = b.option(bool, "enable-network", "Enable network support") orelse Default.enable_network;
    const enable_profiling = b.option(bool, "enable-profiling", "Enable profiling") orelse Default.enable_profiling;
    // GPU back‑end flags
    const gpu_cuda = b.option(bool, "gpu-cuda", "Enable CUDA backend") orelse Default.gpu_cuda;
    const gpu_vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan backend") orelse Default.gpu_vulkan;
    const gpu_metal = b.option(bool, "gpu-metal", "Enable Metal backend") orelse Default.gpu_metal;
    const gpu_webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU backend") orelse Default.gpu_webgpu;
    const gpu_opengl = b.option(bool, "gpu-opengl", "Enable OpenGL backend") orelse Default.gpu_opengl;
    const gpu_opengles = b.option(bool, "gpu-opengles", "Enable OpenGL ES backend") orelse Default.gpu_opengles;
    const gpu_webgl2 = b.option(bool, "gpu-webgl2", "Enable WebGL2 backend") orelse Default.gpu_webgl2;
    const cache_dir = b.option([]const u8, "cache-dir", "Cache directory") orelse Default.cache_dir;
    const global_cache_dir = b.option([]const u8, "global-cache-dir", "Global cache dir") catch null;
    return BuildOptions{
        .enable_gpu = enable_gpu,
        .enable_ai = enable_ai,
        .enable_web = enable_web,
        .enable_database = enable_database,
        .enable_network = enable_network,
        .enable_profiling = enable_profiling,
        .gpu_cuda = gpu_cuda,
        .gpu_vulkan = gpu_vulkan,
        .gpu_metal = gpu_metal,
        .gpu_webgpu = gpu_webgpu,
        .gpu_opengl = gpu_opengl,
        .gpu_opengles = gpu_opengles,
        .gpu_webgl2 = gpu_webgl2,
        .cache_dir = cache_dir,
        .global_cache_dir = global_cache_dir,
    };
}
