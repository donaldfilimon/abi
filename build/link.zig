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
// Solaris/illumos System Library Linking
// =============================================================================

/// Link Solaris/illumos system libraries.
/// - OpenGL: libGL (via Mesa)
pub fn applySolarisLinks(
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
    // On macOS 26+ with version-clamped targets, Zig's framework auto-detection
    // breaks. Explicitly add SDK framework paths so linkFramework can resolve.
    if (os_tag == .macos and @import("builtin").os.tag == .macos and
        @import("builtin").os.version_range.semver.min.major >= 26)
    {
        addSdkFrameworkPaths(mod, mod.owner.graph.io);
    }
    applyLinuxLinks(mod, os_tag, gpu_backends);
    applyWindowsLinks(mod, os_tag, gpu_backends);
    applyBsdLinks(mod, os_tag, gpu_backends);
    applySolarisLinks(mod, os_tag, gpu_backends);
    applyHaikuLinks(mod, os_tag, gpu_backends);
}

// =============================================================================
// SDK Framework Search Paths (macOS 26+ workaround)
// =============================================================================

/// On macOS 26+ with version-clamped targets, Zig's automatic framework
/// search breaks because the host SDK version doesn't match the clamped
/// deployment target. This function explicitly adds the SDK's framework
/// directories so that `-framework Accelerate` etc. can resolve.
pub fn addSdkFrameworkPaths(mod: *std.Build.Module, io: std.Io) void {
    const sdk_path = detectSdkPath(io) orelse return;

    // System/Library/Frameworks is where all Apple frameworks live
    const fw_suffix = "/System/Library/Frameworks";
    const lib_suffix = "/usr/lib";

    // Build full paths — use cwd_relative which accepts absolute paths
    var fw_buf: [512]u8 = undefined;
    var lib_buf: [512]u8 = undefined;

    const fw_path = std.fmt.bufPrint(&fw_buf, "{s}{s}", .{ sdk_path, fw_suffix }) catch return;
    const lib_path = std.fmt.bufPrint(&lib_buf, "{s}{s}", .{ sdk_path, lib_suffix }) catch return;

    mod.addFrameworkPath(.{ .cwd_relative = fw_path });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
}

/// Detect the macOS SDK path via xcrun, with fallbacks.
fn detectSdkPath(io: std.Io) ?[]const u8 {
    // Try xcrun first
    const xcrun_result = runCapture(io, &.{ "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path" });
    if (xcrun_result) |path| return path;

    // Common fallback paths
    const fallbacks = [_][]const u8{
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    };
    for (fallbacks) |path| {
        if (commandSucceeds(io, &.{ "/usr/bin/test", "-d", path })) return path;
    }
    return null;
}

/// Run a command and capture its stdout (trimmed). Returns null on failure.
fn runCapture(io: std.Io, argv: []const []const u8) ?[]const u8 {
    var child = std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch return null;

    // Read stdout into a static buffer
    const Static = struct {
        var buf: [512]u8 = undefined;
    };
    var len: usize = 0;
    while (len < Static.buf.len) {
        const chunk = child.stdout.?.readSome(io, Static.buf[len..]) catch break;
        if (chunk.len == 0) break;
        len += chunk.len;
    }

    const term = child.wait(io) catch return null;
    switch (term) {
        .exited => |code| if (code != 0) return null,
        else => return null,
    }

    // Trim trailing whitespace
    var end = len;
    while (end > 0 and (Static.buf[end - 1] == '\n' or Static.buf[end - 1] == '\r' or Static.buf[end - 1] == ' ')) end -= 1;
    if (end == 0) return null;
    return Static.buf[0..end];
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
