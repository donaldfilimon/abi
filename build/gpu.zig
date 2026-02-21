const std = @import("std");
const gpu_policy = @import("gpu_policy.zig");

/// All supported GPU compute backends.
pub const GpuBackend = enum {
    none,
    auto,
    cuda,
    vulkan,
    stdgpu,
    metal,
    webgpu,
    opengl,
    opengles,
    webgl2,
    fpga,
    tpu,

    pub fn fromString(s: []const u8) ?GpuBackend {
        return std.StaticStringMap(GpuBackend).initComptime(.{
            .{ "none", .none },     .{ "auto", .auto },     .{ "cuda", .cuda },
            .{ "vulkan", .vulkan }, .{ "stdgpu", .stdgpu }, .{ "metal", .metal },
            .{ "webgpu", .webgpu }, .{ "opengl", .opengl }, .{ "opengles", .opengles },
            .{ "webgl2", .webgl2 }, .{ "fpga", .fpga },     .{ "tpu", .tpu },
        }).get(s);
    }
};

pub const backend_option_help =
    "GPU backend(s): auto, none, cuda, vulkan, metal, webgpu, tpu, opengl, opengles, webgl2, stdgpu, fpga (comma-separated)";

const backend_count = @typeInfo(GpuBackend).@"enum".fields.len;

/// Parse the `-Dgpu-backend=` CLI argument into a list of `GpuBackend`
/// values.  When omitted, auto-detection selects backends based on the
/// target platform.
pub fn parseGpuBackends(
    b: *std.Build,
    backend_str: ?[]const u8,
    enable_gpu: bool,
    enable_web: bool,
    target_os: std.Target.Os.Tag,
    target_abi: std.Target.Abi,
    can_link_metal: bool,
) []const GpuBackend {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;
    var use_auto = false;
    const windows_safe_auto = b.option(
        bool,
        "gpu-auto-windows-safe",
        "Use conservative stdgpu-only auto backend on Windows targets",
    ) orelse false;

    if (backend_str) |str| {
        var iter = std.mem.splitScalar(u8, str, ',');
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t");
            if (trimmed.len == 0) continue;
            if (GpuBackend.fromString(trimmed)) |backend| {
                if (backend == .none) return &.{};
                if (backend == .auto) {
                    use_auto = true;
                    continue;
                }
                addBackend(backend, &buffer, &count, &seen);
            } else std.log.warn("Unknown GPU backend: '{s}'", .{trimmed});
        }
        if (use_auto) {
            appendAutoBackends(
                &buffer,
                &count,
                &seen,
                enable_gpu,
                enable_web,
                target_os,
                target_abi,
                can_link_metal,
                true,
                windows_safe_auto,
            );
        }
    } else {
        appendAutoBackends(
            &buffer,
            &count,
            &seen,
            enable_gpu,
            enable_web,
            target_os,
            target_abi,
            can_link_metal,
            true,
            windows_safe_auto,
        );
    }
    return b.allocator.dupe(GpuBackend, buffer[0..count]) catch &.{};
}

/// Append platform-appropriate backends using the gpu_policy rules.
fn appendAutoBackends(
    buffer: *[backend_count]GpuBackend,
    count: *usize,
    seen: *[backend_count]bool,
    enable_gpu: bool,
    enable_web: bool,
    target_os: std.Target.Os.Tag,
    target_abi: std.Target.Abi,
    can_link_metal: bool,
    warn_if_metal_skipped: bool,
    windows_safe_auto: bool,
) void {
    if (target_os == .windows and windows_safe_auto) {
        std.log.warn(
            "Windows auto backend compatibility mode enabled; using stdgpu only. " ++
                "Pass -Dgpu-auto-windows-safe=false for canonical policy order.",
            .{},
        );
        addBackend(.stdgpu, buffer, count, seen);
        return;
    }

    const names = gpu_policy.resolveAutoBackendNames(.{
        .platform = gpu_policy.classify(target_os, target_abi),
        .enable_gpu = enable_gpu,
        .enable_web = enable_web,
        .can_link_metal = can_link_metal,
        .warn_if_metal_skipped = warn_if_metal_skipped,
        .allow_simulated = false,
    });

    for (names.slice()) |name| {
        if (GpuBackend.fromString(name)) |backend| {
            if (backend == .auto or backend == .none) continue;
            addBackend(backend, buffer, count, seen);
        }
    }
}

fn addBackend(
    backend: GpuBackend,
    buffer: *[backend_count]GpuBackend,
    count: *usize,
    seen: *[backend_count]bool,
) void {
    const idx = @intFromEnum(backend);
    if (seen[idx]) return;
    if (count.* >= buffer.len) return;
    buffer[count.*] = backend;
    count.* += 1;
    seen[idx] = true;
}

// ── Tests ────────────────────────────────────────────────────────────────

test "default backends: macOS keeps metal when frameworks are available" {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;

    appendAutoBackends(&buffer, &count, &seen, true, false, .macos, .none, true, false, false);
    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqual(GpuBackend.metal, buffer[0]);
    try std.testing.expectEqual(GpuBackend.vulkan, buffer[1]);
    try std.testing.expectEqual(GpuBackend.opengl, buffer[2]);
    try std.testing.expectEqual(GpuBackend.stdgpu, buffer[3]);
}

test "default backends: macOS falls back when frameworks are unavailable" {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;

    appendAutoBackends(&buffer, &count, &seen, true, false, .macos, .none, false, false, false);
    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqual(GpuBackend.vulkan, buffer[0]);
    try std.testing.expectEqual(GpuBackend.opengl, buffer[1]);
    try std.testing.expectEqual(GpuBackend.stdgpu, buffer[2]);
}

test "default backends: windows follows canonical policy order when compatibility mode is off" {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;

    appendAutoBackends(&buffer, &count, &seen, true, false, .windows, .none, false, false, false);
    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqual(GpuBackend.cuda, buffer[0]);
    try std.testing.expectEqual(GpuBackend.vulkan, buffer[1]);
    try std.testing.expectEqual(GpuBackend.opengl, buffer[2]);
    try std.testing.expectEqual(GpuBackend.stdgpu, buffer[3]);
}

test "default backends: windows compatibility mode keeps stdgpu only" {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;

    appendAutoBackends(&buffer, &count, &seen, true, false, .windows, .none, false, false, true);
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(GpuBackend.stdgpu, buffer[0]);
}

test "default backends: android target uses vulkan then opengles" {
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;

    appendAutoBackends(&buffer, &count, &seen, true, false, .linux, .android, false, false, false);
    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqual(GpuBackend.vulkan, buffer[0]);
    try std.testing.expectEqual(GpuBackend.opengles, buffer[1]);
    try std.testing.expectEqual(GpuBackend.stdgpu, buffer[2]);
}
