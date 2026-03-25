//! Integration tests for the canonical `abi.database` boundary.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

fn isAllowedCoreDatabaseImport(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "src/features/database/") or
        std.mem.eql(u8, path, "src/features/ai/database/neural_store.zig");
}

fn assertNoDirectCoreDatabaseImports(
    allocator: std.mem.Allocator,
    io: std.Io,
    root_path: []const u8,
) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.basename, ".zig")) continue;

        const full_path = try std.fs.path.join(allocator, &.{ root_path, entry.path });
        defer allocator.free(full_path);

        if (isAllowedCoreDatabaseImport(full_path)) continue;

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
            if (std.mem.indexOf(u8, line, "@import(") != null and
                std.mem.indexOf(u8, line, "core/database/") != null)
            {
                std.debug.print(
                    "disallowed core/database import in {s}:{d}: {s}\n",
                    .{ full_path, line_no, line },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "database: public API surface compiles" {
    comptime {
        _ = abi.database.Store;
        _ = abi.database.Context;
        _ = abi.database.memory.BlockChain;
        _ = abi.database.storage.StorageConfig;
        _ = abi.database.storage.SectionType;
        _ = abi.database.distributed.ShardManager;
        _ = abi.database.distributed.VersionVector;
        _ = abi.database.retrieval.KMeans;
        _ = abi.database.retrieval.ProductQuantizer;
    }

    try std.testing.expectEqual(build_options.feat_database, abi.database.isEnabled());
}

test "database: Stats type has expected fields" {
    const Stats = abi.database.Stats;
    const stats: Stats = .{};
    try std.testing.expectEqual(@as(usize, 0), stats.count);
    try std.testing.expectEqual(@as(usize, 0), stats.dimension);
    try std.testing.expectEqual(@as(usize, 0), stats.memory_bytes);
}

test "database: DiagnosticsInfo defaults are healthy" {
    const DiagnosticsInfo = abi.database.DiagnosticsInfo;
    const info: DiagnosticsInfo = .{};
    try std.testing.expect(info.isHealthy());
    try std.testing.expectEqual(@as(f32, 1.0), info.index_health);
    try std.testing.expectEqual(@as(f32, 1.0), info.norm_cache_health);
    try std.testing.expect(info.pool_stats == null);
}

test "database: SearchResult type is accessible" {
    const SearchResult = abi.database.SearchResult;
    const result: SearchResult = .{};
    try std.testing.expectEqual(@as(u64, 0), result.id);
    try std.testing.expectEqual(@as(f32, 0.0), result.score);
}

test "database: DatabaseConfig type compiles" {
    const Config = abi.database.DatabaseConfig;
    const cfg: Config = .{};
    _ = cfg;
}

test "database: BatchItem type is accessible" {
    const BatchItem = abi.database.BatchItem;
    const item: BatchItem = .{};
    _ = item;
}

test "database: VectorView type has expected defaults" {
    const VectorView = abi.database.VectorView;
    const view: VectorView = .{};
    try std.testing.expectEqual(@as(u64, 0), view.id);
}

test "database: deleteVector stub returns false" {
    if (!build_options.feat_database) {
        // Stub deleteVector returns false (no-op)
        var handle: abi.database.DatabaseHandle = .{};
        const deleted = abi.database.deleteVector(&handle, 42);
        try std.testing.expect(!deleted);
    }
}

test "database: boundary forbids direct core database imports" {
    const allocator = std.testing.allocator;
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    try assertNoDirectCoreDatabaseImports(allocator, io, "src/features");
    try assertNoDirectCoreDatabaseImports(allocator, io, "src/protocols");
    try assertNoDirectCoreDatabaseImports(allocator, io, "test");
}

test {
    std.testing.refAllDecls(@This());
}
