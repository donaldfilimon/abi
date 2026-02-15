const std = @import("std");
const builtin = @import("builtin");

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

pub fn parseGpuBackends(b: *std.Build, enable_gpu: bool, enable_web: bool) []const GpuBackend {
    const backend_str = b.option(
        []const u8,
        "gpu-backend",
        "GPU backend(s): auto, none, cuda, vulkan, metal, webgpu, tpu, opengl, opengles, webgl2, stdgpu, fpga (comma-separated)",
    );

    const backend_count = @typeInfo(GpuBackend).@"enum".fields.len;
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;
    var use_auto = false;

    const addBackend = struct {
        fn call(
            backend: GpuBackend,
            buffer_ptr: *[backend_count]GpuBackend,
            count_ptr: *usize,
            seen_ptr: *[backend_count]bool,
        ) void {
            const idx = @intFromEnum(backend);
            if (seen_ptr[idx]) return;
            if (count_ptr.* >= buffer_ptr.len) return;
            buffer_ptr[count_ptr.*] = backend;
            count_ptr.* += 1;
            seen_ptr[idx] = true;
        }
    }.call;

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
            if (enable_gpu) {
                if (builtin.os.tag == .macos) {
                    addBackend(.metal, &buffer, &count, &seen);
                    addBackend(.vulkan, &buffer, &count, &seen);
                } else if (builtin.os.tag == .windows) {
                    addBackend(.stdgpu, &buffer, &count, &seen);
                } else {
                    addBackend(.vulkan, &buffer, &count, &seen);
                }
            }
            if (enable_web) {
                if (builtin.os.tag != .windows) {
                    addBackend(.webgpu, &buffer, &count, &seen);
                    addBackend(.webgl2, &buffer, &count, &seen);
                }
            }
        }
    } else {
        // Defaults: macOS uses Metal first (Apple Silicon GPU + CoreML/ANE); then Vulkan
        if (enable_gpu and builtin.os.tag == .macos) {
            addBackend(.metal, &buffer, &count, &seen);
            addBackend(.vulkan, &buffer, &count, &seen);
        } else if (enable_gpu and builtin.os.tag != .windows) {
            addBackend(.vulkan, &buffer, &count, &seen);
        }
        if (enable_gpu and builtin.os.tag == .windows) addBackend(.stdgpu, &buffer, &count, &seen);
        if (enable_web and builtin.os.tag != .windows) {
            addBackend(.webgpu, &buffer, &count, &seen);
            addBackend(.webgl2, &buffer, &count, &seen);
        }
    }
    return b.allocator.dupe(GpuBackend, buffer[0..count]) catch &.{};
}
