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
        // libc resolves to libSystem.B.dylib on macOS/iOS, providing all
        // POSIX and C runtime symbols (_malloc, _abort, _arc4random_buf,
        // _clock_gettime, etc.).  Without this, linking fails with ~40
        // undefined symbols even though frameworks are present.
        mod.linkSystemLibrary("c", .{});

        // Accelerate provides BLAS, LAPACK, vDSP for CPU-side linear algebra
        mod.linkFramework("Accelerate", .{});
        mod.linkFramework("Foundation", .{});

        if (os_tag == .macos) {
            mod.linkFramework("AppKit", .{});
            mod.linkFramework("Cocoa", .{});
        } else if (os_tag == .ios) {
            mod.linkFramework("UIKit", .{});
        }

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
// BSD System Library Linking
// =============================================================================

/// Link BSD system libraries based on selected GPU backends.
/// FreeBSD, NetBSD, OpenBSD, and DragonFly share a similar library layout
/// to Linux but may have different package paths.
/// - Vulkan: libvulkan (primarily FreeBSD)
/// - OpenGL: libGL
pub fn applyBsdLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_backends: []const GpuBackend,
) void {
    switch (os_tag) {
        .freebsd, .netbsd, .openbsd, .dragonfly => {},
        else => return,
    }

    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("m", .{});

    if (hasBackend(gpu_backends, .vulkan)) {
        mod.linkSystemLibrary("vulkan", .{});
    }
    if (hasBackend(gpu_backends, .opengl)) {
        mod.linkSystemLibrary("GL", .{});
    }
}

// =============================================================================
// illumos System Library Linking
// =============================================================================

/// Link illumos system libraries.
/// - OpenGL: libGL (via Mesa)
pub fn applyIllumosLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_backends: []const GpuBackend,
) void {
    switch (os_tag) {
        .illumos => {},
        else => return,
    }

    mod.linkSystemLibrary("c", .{});
    mod.linkSystemLibrary("m", .{});
    mod.linkSystemLibrary("socket", .{});
    mod.linkSystemLibrary("nsl", .{});

    if (hasBackend(gpu_backends, .opengl)) {
        mod.linkSystemLibrary("GL", .{});
    }
}

// =============================================================================
// Haiku System Library Linking
// =============================================================================

/// Link Haiku system libraries.
/// - OpenGL: libGL (native Haiku OpenGL kit)
pub fn applyHaikuLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_backends: []const GpuBackend,
) void {
    if (os_tag != .haiku) return;

    if (hasBackend(gpu_backends, .opengl)) {
        mod.linkSystemLibrary("GL", .{});
    }
}

// =============================================================================
// Android System Library Linking
// =============================================================================

/// Link Android system libraries.
pub fn applyAndroidLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    abi: std.Target.Abi,
) void {
    if (os_tag == .linux and abi == .android) {
        mod.linkSystemLibrary("log", .{});
        mod.linkSystemLibrary("android", .{});
        mod.linkSystemLibrary("EGL", .{});
        mod.linkSystemLibrary("GLESv2", .{});
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
    applyBsdLinks(mod, os_tag, gpu_backends);
    applyIllumosLinks(mod, os_tag, gpu_backends);
    applyHaikuLinks(mod, os_tag, gpu_backends);
    applyAndroidLinks(mod, os_tag, mod.resolved_target.?.result.abi);
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
pub fn canLinkMetalFrameworks(b: *std.Build, os_tag: std.Target.Os.Tag) bool {
    if (os_tag != .macos) return false;

    if (commandSucceeds(b, &.{ "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path" }))
        return true;

    for (required_metal_framework_paths) |path|
        if (!dirExists(b, path)) return false;
    return true;
}

/// Abort the build when the user explicitly requested `-Dgpu-backend=metal`
/// but the required frameworks are not available.
pub fn validateMetalBackendRequest(
    backend_arg: ?[]const u8,
    os_tag: std.Target.Os.Tag,
    can_link_metal: bool,
) void {
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

fn hasBackend(backends: []const GpuBackend, target_backend: GpuBackend) bool {
    for (backends) |b| {
        if (b == target_backend) return true;
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

/// Check whether an absolute directory path exists.
fn dirExists(b: *std.Build, path: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(b.graph.io, path, .{}) catch return false;
    dir.close(b.graph.io);
    return true;
}

/// Run a command and return `true` if it exits with status 0.
fn commandSucceeds(b: *std.Build, argv: []const []const u8) bool {
    var child = std.process.spawn(b.graph.io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch return false;

    const term = child.wait(b.graph.io) catch return false;
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}
