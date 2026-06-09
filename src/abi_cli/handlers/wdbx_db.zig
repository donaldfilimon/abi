const std = @import("std");
const features = @import("../../features/mod.zig");

const wdbx = features.wdbx;

fn walPath(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.wal", .{path});
}

pub fn initDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !u8 {
    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    try wdbx.persistence.saveToPath(io, allocator, &store, path);
    std.debug.print("initialized empty WDBX snapshot at {s}\n", .{path});
    return 0;
}

pub fn verifyDb(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("verify FAILED: snapshot {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const blocks_ok = store.verifyBlocks();
    const s = store.stats();
    std.debug.print("snapshot OK: kv={d} vectors={d} blocks={d} spatial={d} chain_valid={any}\n", .{ s.kv_entries, s.vectors, s.blocks, s.spatial_records, blocks_ok });

    const wp = try walPath(allocator, path);
    defer allocator.free(wp);
    var wal_store = wdbx.wal.replay(io, allocator, wp) catch |err| switch (err) {
        error.FileNotFound => return if (blocks_ok) 0 else 1,
        else => {
            std.debug.print("WAL verify FAILED: {s}: {s}\n", .{ wp, @errorName(err) });
            return 1;
        },
    };
    defer wal_store.deinit();

    const wal_blocks = wal_store.blockCount();
    const consistent = wal_blocks == s.blocks and wal_store.verifyBlocks();
    std.debug.print("WAL replay OK: blocks={d} consistent_with_snapshot={any}\n", .{ wal_blocks, consistent });
    return if (blocks_ok and consistent) 0 else 1;
}

pub fn blockInsert(io: std.Io, allocator: std.mem.Allocator, path: []const u8, profile: []const u8, metadata: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| switch (err) {
        error.FileNotFound => wdbx.Store.init(allocator),
        else => return err,
    };
    defer store.deinit();

    _ = try store.appendBlock(profile, 0, 0, metadata);
    const last = store.lastBlock().?;

    try wdbx.persistence.saveToPath(io, allocator, &store, path);

    const wp = try walPath(allocator, path);
    defer allocator.free(wp);
    try wdbx.wal.appendBlock(io, allocator, wp, profile, 0, 0, metadata, last.timestamp_ms);

    std.debug.print("appended block: profile={s} blocks={d} hash={s}\n", .{ profile, store.blockCount(), std.fmt.bytesToHex(last.id, .lower) });
    return 0;
}

pub fn blockGet(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const last = store.lastBlock() orelse {
        std.debug.print("no blocks in {s}\n", .{path});
        return 0;
    };
    std.debug.print(
        "block: profile={s} query_id={d} response_id={d} timestamp_ms={d}\n  hash={s}\n  metadata={s}\n",
        .{ last.profile, last.query_id, last.response_id, last.timestamp_ms, std.fmt.bytesToHex(last.id, .lower), last.metadata },
    );
    return 0;
}

pub fn query(io: std.Io, allocator: std.mem.Allocator, path: []const u8) anyerror!u8 {
    var store = wdbx.persistence.loadFromPath(io, allocator, path) catch |err| {
        std.debug.print("error: {s}: {s}\n", .{ path, @errorName(err) });
        return 1;
    };
    defer store.deinit();

    const manifest = try store.exportManifest(allocator);
    defer allocator.free(manifest);
    std.debug.print("{s}\n", .{manifest});
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
