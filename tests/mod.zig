const std = @import("std");
const abi = @import("abi");

test "abi version returns non-empty string" {
    try std.testing.expect(abi.version().len > 0);
}

test "framework init and shutdown" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    try std.testing.expect(!framework.isRunning());
}

test "compute engine init and deinit" {
    const config_mod = abi.compute.runtime.config;
    const engine_mod = abi.compute.runtime.engine;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = config_mod.EngineConfig{
        .worker_count = 2,
        .drain_mode = .drain,
        .metrics_buffer_size = 1024,
        .topology_flags = 0,
    };

    const engine = try engine_mod.Engine.init(allocator, cfg);
    try std.testing.expect(engine.workers.len == 2);
    engine.deinit();
}
