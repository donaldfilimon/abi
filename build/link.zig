const std = @import("std");
const gpu_mod = @import("gpu.zig");

const GpuBackend = gpu_mod.GpuBackend;

// =============================================================================
// macOS Framework Linking
// =============================================================================

/// Link macOS frameworks into a module based on target OS and GPU backend.
/// - Accelerate: always linked on macOS (BLAS/LAPACK/vDSP for CPU math)
/// - Metal + CoreML + MPS + Foundation: linked when Metal backend enabled
pub fn applyFrameworkLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_metal: bool,
) void {
    if (os_tag == .macos or os_tag == .ios) {
        // Accelerate provides BLAS, LAPACK, vDSP for CPU-side linear algebra
        mod.linkFramework("Accelerate", .{});
        mod.linkFramework("Foundation", .{});

        if (gpu_metal) {
            mod.linkFramework("Metal", .{});
            mod.linkFramework("CoreML", .{});
            mod.linkFramework("MetalPerformanceShaders", .{});
        }
    }
}

// =============================================================================
// Linux System Library Linking
// =============================================================================

/// Link Linux system libraries based on selected GPU backends.
/// - CUDA: libcuda, libcublas, libcudart
/// - Vulkan: libvulkan
/// - OpenGL: libGL
pub fn applyLinuxLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_backends: []const GpuBackend,
) void {
    if (os_tag != .linux) return;

    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("m", .{}); // libm for math

    if (hasBackend(gpu_backends, .cuda)) {
        mod.linkSystemLibrary("cuda", .{});
        mod.linkSystemLibrary("cublas", .{});
        mod.linkSystemLibrary("cudart", .{});
        mod.linkSystemLibrary("cudnn", .{});
    }
    if (hasBackend(gpu_backends, .vulkan)) {
        mod.linkSystemLibrary("vulkan", .{});
    }
    if (hasBackend(gpu_backends, .opengl)) {
        mod.linkSystemLibrary("GL", .{});
    }
}

// =============================================================================
// Windows System Library Linking
// =============================================================================

/// Link Windows system libraries based on selected GPU backends.
/// - CUDA: cuda, cublas
/// - Vulkan: vulkan-1
pub fn applyWindowsLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_backends: []const GpuBackend,
) void {
    if (os_tag != .windows) return;

    if (hasBackend(gpu_backends, .cuda)) {
        mod.linkSystemLibrary("cuda", .{});
        mod.linkSystemLibrary("cublas", .{});
    }
    if (hasBackend(gpu_backends, .vulkan)) {
        mod.linkSystemLibrary("vulkan-1", .{});
    }
}

// =============================================================================
// Unified Linker Entry Point
// =============================================================================

/// Apply all platform-specific links for a module. Call this once per artifact.
pub fn applyAllPlatformLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_metal: bool,
    gpu_backends: []const GpuBackend,
) void {
    applyFrameworkLinks(mod, os_tag, gpu_metal);
    applyLinuxLinks(mod, os_tag, gpu_backends);
    applyWindowsLinks(mod, os_tag, gpu_backends);
}

// =============================================================================
// Framework Detection
// =============================================================================

const required_metal_framework_paths = [_][]const u8{
    "/System/Library/Frameworks/Metal.framework",
    "/System/Library/Frameworks/CoreML.framework",
    "/System/Library/Frameworks/MetalPerformanceShaders.framework",
    "/System/Library/Frameworks/Foundation.framework",
};

/// Probe whether Metal frameworks can be linked on this host.  Tries
/// `xcrun --sdk macosx --show-sdk-path` first; falls back to checking
/// framework paths directly.
pub fn canLinkMetalFrameworks(io: std.Io, os_tag: std.Target.Os.Tag) bool {
    if (os_tag != .macos) return false;

    if (commandSucceeds(io, &.{ "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path" }))
        return true;

    for (required_metal_framework_paths) |path|
        if (!commandSucceeds(io, &.{ "/usr/bin/test", "-e", path })) return false;
    return true;
}

/// Abort the build when the user explicitly requested `-Dgpu-backend=metal`
/// but the required frameworks are not available.
pub fn validateMetalBackendRequest(
    b: *std.Build,
    backend_arg: ?[]const u8,
    os_tag: std.Target.Os.Tag,
    can_link_metal: bool,
) void {
    _ = b;
    if (os_tag != .macos) return;
    if (!isExplicitMetalRequested(backend_arg)) return;
    if (can_link_metal) return;

    std.debug.panic(
        "explicit gpu-backend=metal requested but Apple frameworks are unavailable. " ++
            "Install Xcode Command Line Tools or use -Dgpu-backend=auto/vulkan.",
        .{},
    );
}

// =============================================================================
// Helpers
// =============================================================================

fn hasBackend(backends: []const GpuBackend, target: GpuBackend) bool {
    for (backends) |b| {
        if (b == target) return true;
    }
    return false;
}

fn isExplicitMetalRequested(backend_arg: ?[]const u8) bool {
    const arg = backend_arg orelse return false;
    var it = std.mem.splitScalar(u8, arg, ',');
    while (it.next()) |raw| {
        const token = std.mem.trim(u8, raw, " \t");
        if (token.len == 0) continue;
        if (std.ascii.eqlIgnoreCase(token, "metal")) return true;
    }
    return false;
}

fn commandSucceeds(io: std.Io, argv: []const []const u8) bool {
    var child = std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch return false;

    const term = child.wait(io) catch return false;
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}
