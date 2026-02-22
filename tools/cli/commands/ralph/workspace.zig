//! Ralph workspace helpers (runs, lock file, latest pointers).

const std = @import("std");
const cfg = @import("config.zig");

pub const LockError = error{
    LockHeld,
};

pub const LockInfo = struct {
    run_id: []const u8 = "",
    started_at: i64 = 0,
    stale: bool = false,
};

pub fn nowEpochSeconds() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &ts);
    return @intCast(ts.sec);
}

pub fn generateRunId(allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "run-{d}", .{nowEpochSeconds()});
}

pub fn runDirPath(allocator: std.mem.Allocator, run_id: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ cfg.RUNS_DIR, run_id });
}

pub fn ensureRunLayout(allocator: std.mem.Allocator, io: std.Io, run_id: []const u8) !void {
    cfg.ensureDir(io, cfg.WORKSPACE_DIR);
    cfg.ensureDir(io, cfg.RUNS_DIR);

    const run_dir = try runDirPath(allocator, run_id);
    defer allocator.free(run_dir);
    cfg.ensureDir(io, run_dir);

    const iter_dir = try std.fmt.allocPrint(allocator, "{s}/iterations", .{run_dir});
    defer allocator.free(iter_dir);
    cfg.ensureDir(io, iter_dir);
}

pub fn readLockInfo(allocator: std.mem.Allocator, io: std.Io, stale_after_seconds: i64) LockInfo {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.LOCK_FILE,
        allocator,
        .limited(16 * 1024),
    ) catch return .{};
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, contents, .{}) catch return .{};
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return .{},
    };

    const started_at = if (obj.get("started_at")) |v| switch (v) {
        .integer => |i| i,
        else => 0,
    } else 0;

    if (started_at == 0) return .{ .started_at = 0, .stale = false };

    const age = nowEpochSeconds() - started_at;
    return .{
        .run_id = "",
        .started_at = started_at,
        .stale = age > stale_after_seconds,
    };
}

pub fn acquireLoopLock(
    allocator: std.mem.Allocator,
    io: std.Io,
    run_id: []const u8,
    stale_after_seconds: i64,
) !void {
    if (cfg.fileExists(io, cfg.LOCK_FILE)) {
        const lock_info = readLockInfo(allocator, io, stale_after_seconds);
        if (lock_info.stale) {
            cfg.removeLockFile(io);
        } else {
            return LockError.LockHeld;
        }
    }

    const lock_json = try std.fmt.allocPrint(
        allocator,
        "{{\"run_id\":\"{s}\",\"started_at\":{d}}}",
        .{ run_id, nowEpochSeconds() },
    );
    defer allocator.free(lock_json);
    try cfg.writeFile(allocator, io, cfg.LOCK_FILE, lock_json);
}

pub fn releaseLoopLock(io: std.Io) void {
    cfg.removeLockFile(io);
}

pub fn latestReportPath(allocator: std.mem.Allocator, io: std.Io) ?[]u8 {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        cfg.LATEST_RUN_FILE,
        allocator,
        .limited(16 * 1024),
    ) catch return null;
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, contents, .{}) catch return null;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };
    const report = if (obj.get("report")) |v| switch (v) {
        .string => |s| s,
        else => return null,
    } else return null;

    return allocator.dupe(u8, report) catch null;
}

pub fn writeLatestRun(
    allocator: std.mem.Allocator,
    io: std.Io,
    run_id: []const u8,
    report_path: []const u8,
) !void {
    const json = try std.fmt.allocPrint(
        allocator,
        "{{\"run_id\":\"{s}\",\"report\":\"{s}\",\"updated_at\":{d}}}",
        .{ run_id, report_path, nowEpochSeconds() },
    );
    defer allocator.free(json);
    try cfg.writeFile(allocator, io, cfg.LATEST_RUN_FILE, json);
}
