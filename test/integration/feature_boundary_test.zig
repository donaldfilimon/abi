//! Phase 3 public-surface and import-boundary coverage.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

fn assertNoDirectFeatureInternalImports(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
    comptime feature_name: []const u8,
) !void {
    const owner_prefix = "src/features/" ++ feature_name ++ "/";
    const import_prefix = "features/" ++ feature_name ++ "/";
    const mod_path = import_prefix ++ "mod.zig";
    const stub_path = import_prefix ++ "stub.zig";

    var dir = try std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;

        const full_path = try std.fs.path.join(allocator, &.{ root_path, entry.path });
        defer allocator.free(full_path);

        if (std.mem.startsWith(u8, full_path, owner_prefix)) continue;

        const contents = try std.Io.Dir.cwd().readFileAlloc(
            io,
            full_path,
            allocator,
            .limited(1024 * 1024),
        );
        defer allocator.free(contents);

        var lines = std.mem.splitScalar(u8, contents, '\n');
        var line_no: usize = 1;
        while (lines.next()) |line| : (line_no += 1) {
            if (std.mem.indexOf(u8, line, "@import(") == null) continue;
            if (std.mem.indexOf(u8, line, import_prefix) == null) continue;
            if (std.mem.indexOf(u8, line, mod_path) != null) continue;
            if (std.mem.indexOf(u8, line, stub_path) != null) continue;

            std.debug.print(
                "disallowed internal {s} import in {s}:{d}: {s}\n",
                .{ feature_name, full_path, line_no, line },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test "phase3: network public surface compiles" {
    comptime {
        _ = abi.network.Context;
        _ = abi.network.NodeRegistry;
        _ = abi.network.RaftNode;
        _ = abi.network.TcpTransport;
        _ = abi.network.MessageHeader;
        _ = abi.network.HeartbeatStateMachine;
        _ = abi.network.UnifiedMemoryManager;
    }

    try std.testing.expectEqual(build_options.feat_network, abi.network.isEnabled());
}

test "phase3: network boundary forbids direct internal imports" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try assertNoDirectFeatureInternalImports(allocator, io, "src/features", "network");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/protocols", "network");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/core/database", "network");
    try assertNoDirectFeatureInternalImports(allocator, io, "test", "network");
}

test "phase3: gpu public surface compiles" {
    comptime {
        _ = abi.gpu.Gpu;
        _ = abi.gpu.GpuConfig;
        _ = abi.gpu.Backend;
        _ = abi.gpu.Device;
        _ = abi.gpu.UnifiedBuffer;
        _ = abi.gpu.KernelBuilder;
        _ = abi.gpu.ExecutionResult;
        _ = abi.gpu.HealthStatus;
    }
}

test "phase3: gpu boundary forbids direct internal imports" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try assertNoDirectFeatureInternalImports(allocator, io, "src/features", "gpu");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/protocols", "gpu");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/core/database", "gpu");
    try assertNoDirectFeatureInternalImports(allocator, io, "test", "gpu");
}

test "phase3: compute public surface compiles" {
    comptime {
        _ = abi.compute.Context;
        _ = abi.compute.mesh;
        _ = abi.compute.mesh.ComputeNode;
        _ = abi.compute.mesh.MeshOrchestrator;
    }

    try std.testing.expectEqual(build_options.feat_compute, abi.compute.isEnabled());
}

test "phase3: compute boundary forbids direct internal imports" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try assertNoDirectFeatureInternalImports(allocator, io, "src/features", "compute");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/protocols", "compute");
    try assertNoDirectFeatureInternalImports(allocator, io, "test", "compute");
}

test "phase3: ai public surface compiles" {
    comptime {
        _ = abi.ai.Context;
        _ = abi.ai.profile.MultiProfileRouter;
        _ = abi.ai.profile.ConversationMemory;
        _ = abi.ai.memory.MemoryManager;
        _ = abi.ai.tools.ToolRegistry;
        _ = abi.ai.training.TrainingConfig;
        _ = abi.ai.database.WdbxTokenDataset;
    }

    try std.testing.expectEqual(build_options.feat_ai, abi.ai.isEnabled());
}

test "phase3: ai boundary forbids direct internal imports" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try assertNoDirectFeatureInternalImports(allocator, io, "src/features", "ai");
    try assertNoDirectFeatureInternalImports(allocator, io, "src/protocols", "ai");
    try assertNoDirectFeatureInternalImports(allocator, io, "test", "ai");
}

test {
    std.testing.refAllDecls(@This());
}
