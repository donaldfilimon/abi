//! Demonstration entrypoint showcasing core and compute features.
const std = @import("std");
const abi = @import("abi");
const core = abi.core;
const compute = abi.compute;
const workload = abi.compute.runtime.workload;

pub fn run(allocator: std.mem.Allocator) !void {
    var framework = try abi.createDefaultFramework(allocator);
    defer framework.deinit();

    const info = core.PlatformInfo.detect();
    std.debug.print("Abbey-Aviva-Abi Demo\n", .{});
    std.debug.print("  Version: {s}\n", .{abi.version()});
    std.debug.print(
        "  OS: {t} Arch: {t} Threads: {d}\n",
        .{ info.os, info.arch, info.max_threads },
    );

    const gpu_summary = abi.gpu.summary();
    const gpu_status = if (gpu_summary.module_enabled) "enabled" else "disabled";
    std.debug.print(
        "  GPU: {s} (devices {d})\n",
        .{ gpu_status, gpu_summary.device_count },
    );

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [4]f32 = undefined;
    workload.matMul(&a, &b, 2, 2, 2, out[0..]);
    std.debug.print(
        "  MatMul: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n",
        .{ out[0], out[1], out[2], out[3] },
    );

    var engine = try compute.createDefaultEngine(allocator);
    defer engine.deinit();

    const result = try compute.runTask(&engine, u64, sampleTask, 1000);
    std.debug.print("  Compute engine result: {d}\n", .{result});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try run(gpa.allocator());
}

fn sampleTask(_: std.mem.Allocator) !u64 {
    return 123;
}
