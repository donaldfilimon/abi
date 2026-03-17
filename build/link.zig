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

    // On macOS 26+ with version-clamped targets, Zig's framework auto-detection
    // breaks. Explicitly add SDK framework paths so linkFramework can resolve.
    const is_macos_host = @import("builtin").os.tag == .macos;
    const is_blocked_darwin = is_macos_host and @import("builtin").os.version_range.semver.min.major >= 26;

    if (os_tag == .macos and is_blocked_darwin) {
        addSdkFrameworkPaths(mod, mod.owner.graph.io);
    }

    applyLinuxLinks(mod, os_tag, gpu_backends);
    applyWindowsLinks(mod, os_tag, gpu_backends);
    applyBsdLinks(mod, os_tag, gpu_backends);
    applySolarisLinks(mod, os_tag, gpu_backends);
    applyHaikuLinks(mod, os_tag, gpu_backends);
    applyAndroidLinks(mod, os_tag, mod.resolved_target.?.result.abi);
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
    const alloc = mod.owner.allocator;

    const fw_path = std.fmt.allocPrint(alloc, "{s}/System/Library/Frameworks", .{sdk_path}) catch return;
    const lib_path = std.fmt.allocPrint(alloc, "{s}/usr/lib", .{sdk_path}) catch return;

    mod.addSystemFrameworkPath(.{ .cwd_relative = fw_path });
    mod.addLibraryPath(.{ .cwd_relative = lib_path });
}

/// Detect the macOS SDK path by probing known locations.
pub fn detectSdkPath(io: std.Io) ?[]const u8 {
    const candidates = [_][]const u8{
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        "/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    };
    for (candidates) |path| {
        if (dirExists(io, path)) return path;
    }
    return null;
}

/// Check if a directory exists using Zig's Io.Dir API.
fn dirExists(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{}) catch return false;
    dir.close(io);
    return true;
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
        if (!dirExists(io, path)) return false;
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

/// On Darwin 25+ (macOS 26+), Zig's built-in Mach-O linker cannot resolve
/// system symbols.  This helper takes a compiled artifact (from addObject
/// with use_llvm=true) and produces a Run step that relinks via Apple's
/// /usr/bin/ld and executes the result.
///
/// The caller must set `artifact.use_llvm = true` before calling.
/// Returns a *Run step; the caller can append extra args or depend on .step.
///
/// Pass a pre-computed `compiler_rt` path to avoid repeated filesystem walks
/// across multiple call sites, or `null` to probe on each call.
pub fn darwinRelink(
    b: *std.Build,
    artifact: *std.Build.Step.Compile,
    output_name: []const u8,
    compiler_rt: ?[]const u8,
) *std.Build.Step.Run {
    const rt_path = compiler_rt orelse findCompilerRt(b);
    const sdk_path = detectSdkPath(b.graph.io) orelse "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";

    const relink = b.addSystemCommand(&.{ "/usr/bin/ld", "-dynamic" });
    relink.addArg("-platform_version");
    relink.addArg("macos");
    // Deployment target 15.0: the last macOS version Zig's linker supports.
    // This is intentionally the clamped deployment target (matching
    // resolveNativeTarget), NOT the live host version from sw_vers.
    relink.addArg("15.0");
    relink.addArg("15.0");
    relink.addArg("-syslibroot");
    relink.addArg(sdk_path);
    relink.addArg("-e");
    relink.addArg("_main");
    relink.addArg("-o");
    const bin = relink.addOutputFileArg(output_name);
    relink.addArtifactArg(artifact);
    relink.addArg("-lSystem");
    // Link macOS frameworks that the artifact may reference transitively.
    // These are no-ops if unused and avoid "symbol not found" at relink time.
    if (@import("builtin").os.tag == .macos) {
        relink.addArg("-framework");
        relink.addArg("IOKit");
        relink.addArg("-framework");
        relink.addArg("CoreFoundation");
    }
    if (rt_path) |path| relink.addArg(path);

    const run = std.Build.Step.Run.create(b, b.fmt("run {s}", .{output_name}));
    run.addFileArg(bin);
    run.step.dependOn(&relink.step);
    return run;
}

/// Find libcompiler_rt.a path by walking the Zig global cache.
pub fn findCompilerRt(b: *std.Build) ?[]const u8 {
    const home = b.graph.environ_map.get("HOME") orelse return null;
    const global_cache = std.fs.path.join(b.allocator, &.{ home, ".cache", "zig", "o" }) catch return null;
    defer b.allocator.free(global_cache);

    const io = b.graph.io;
    var dir = std.Io.Dir.openDirAbsolute(io, global_cache, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var walker = dir.walk(b.allocator) catch return null;
    defer walker.deinit();

    while (walker.next(io) catch null) |entry| {
        if (std.mem.eql(u8, entry.basename, "libcompiler_rt.a")) {
            return std.fs.path.join(b.allocator, &.{ global_cache, entry.path }) catch return null;
        }
    }
    return null;
}
