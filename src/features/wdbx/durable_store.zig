//! Durable WDBX session shared by the long-lived MCP server and the short-lived
//! CLI AI handlers.
//!
//! A session resolves a store path from the environment, opens it (recovering
//! from the segment checkpoint + sidecar WAL via `recovery.open`), keeps the
//! in-memory `Store` WAL-backed so every supported mutation is logged per
//! mutation, and folds the WAL into a segment checkpoint on close. Centralizing
//! this here gives the MCP and CLI entry points identical persistence behavior.
//!
//! The explicit `abi wdbx <path> ...` subcommands keep their own
//! argument-driven open path (`handlers/wdbx_db.zig`) and do NOT use this
//! module: those address a user-supplied path, this one the ambient default.
//!
//! Configuration (default-ON):
//!   ABI_WDBX_PATH     override the base path (default: ~/.abi/wdbx).
//!                     Set to ":memory:" to disable persistence.
//!   ABI_WDBX_PERSIST  set to 0/false/no/off to force an in-memory store.

const std = @import("std");
const builtin = @import("builtin");
const wdbx = @import("mod.zig");
const env = @import("../../foundation/env.zig");

/// Home-dir env var name, Windows-aware (USERPROFILE vs HOME).
const HOME_VAR = if (builtin.target.os.tag == .windows) "USERPROFILE" else "HOME";

pub const PATH_ENV = "ABI_WDBX_PATH";
pub const PERSIST_ENV = "ABI_WDBX_PERSIST";
pub const MEMORY_SENTINEL = ":memory:";
pub const DEFAULT_SUBPATH = ".abi/wdbx";

/// Resolved persistence configuration. `base_path` is null for an in-memory
/// store; otherwise it is an owned path the caller must free.
pub const Config = struct {
    base_path: ?[]u8 = null,

    pub fn isPersistent(self: Config) bool {
        return self.base_path != null;
    }

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        if (self.base_path) |p| allocator.free(p);
        self.base_path = null;
    }
};

fn isFalsey(value: []const u8) bool {
    return std.mem.eql(u8, value, "0") or
        std.mem.eql(u8, value, "false") or
        std.mem.eql(u8, value, "no") or
        std.mem.eql(u8, value, "off");
}

/// Resolve the durable-store configuration from the environment. Default-ON:
/// absent env yields the `~/.abi/wdbx` default. HOME being unset degrades to an
/// in-memory store rather than writing to an unexpected location.
pub fn resolveConfig(allocator: std.mem.Allocator) !Config {
    if (env.get(PERSIST_ENV)) |v| {
        if (isFalsey(v)) return .{ .base_path = null };
    }
    if (env.get(PATH_ENV)) |p| {
        if (std.mem.eql(u8, p, MEMORY_SENTINEL)) return .{ .base_path = null };
        return .{ .base_path = try allocator.dupe(u8, p) };
    }
    if (builtin.target.os.tag != .windows) {
        if (env.get("XDG_DATA_HOME")) |xdg| {
            return .{ .base_path = try std.fmt.allocPrint(allocator, "{s}/abi/wdbx", .{xdg}) };
        }
    }
    const home = env.get(HOME_VAR) orelse return .{ .base_path = null };
    return .{ .base_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, DEFAULT_SUBPATH }) };
}

/// Create (or repair) a directory tree as owner-only (`0700`) on POSIX so a
/// newly opened durable store is not world-readable by default (TM-006 step).
/// Windows/no-mode targets keep the platform default file permissions.
fn ownerOnlyDirPermissions() std.Io.Dir.Permissions {
    if (comptime std.Io.File.Permissions.has_executable_bit) {
        return std.Io.File.Permissions.fromMode(0o700);
    }
    return .default_file;
}

fn ensureOwnerOnlyDir(io: std.Io, path: []const u8) !void {
    if (path.len == 0) return;
    const perms = ownerOnlyDirPermissions();
    _ = try std.Io.Dir.createDirPathStatus(.cwd(), io, path, perms);
    // Existing dirs keep prior mode across createDirPath; repair to 0700.
    // `path` may be relative (CLI/test) or absolute (HOME-resolved ambient path).
    //
    // Open with `iterate = true` so Linux does not use O_PATH. Zig's Dir docs
    // require iterate for setPermissions; an O_PATH fd makes fchmod return
    // EBADF and panic as a "programmer bug" (ambient `abi complete` / mcp).
    if (comptime std.Io.File.Permissions.has_executable_bit) {
        const open_opts: std.Io.Dir.OpenOptions = .{ .iterate = true };
        const dir = if (std.fs.path.isAbsolute(path))
            try std.Io.Dir.openDirAbsolute(io, path, open_opts)
        else
            try std.Io.Dir.cwd().openDir(io, path, open_opts);
        defer dir.close(io);
        try dir.setPermissions(io, perms);
    }
}

fn ensureParentDir(io: std.Io, base: []const u8) !void {
    const dir = std.fs.path.dirname(base) orelse return;
    if (dir.len == 0) return;
    // createDirPath is idempotent (succeeds when the path already exists).
    try ensureOwnerOnlyDir(io, dir);
    // Also ensure the store base directory itself when it is a directory path
    // component used as the segment root (parent of leaf file names).
    // When `base` is a file path (e.g. .../wdbx), only the parent is created.
}

pub const Session = struct {
    store: wdbx.Store,
    /// IO handle for persistence; null for an in-memory session.
    io: ?std.Io = null,
    allocator: std.mem.Allocator,
    /// Owned base path, or null for an in-memory session.
    base_path: ?[]u8 = null,
    /// Owned sidecar WAL path (borrowed by `store` via `attachWal`).
    wal_path: ?[]u8 = null,

    /// Open the ambient durable store using the environment configuration.
    pub fn open(io: std.Io, allocator: std.mem.Allocator) !Session {
        var cfg = try resolveConfig(allocator);
        if (cfg.base_path) |base| {
            defer cfg.deinit(allocator);
            return openAt(io, allocator, base);
        }
        return openInMemory(allocator);
    }

    /// Open a durable store at an explicit base path (env-independent). `base`
    /// is borrowed; the session takes its own copy.
    pub fn openAt(io: std.Io, allocator: std.mem.Allocator, base: []const u8) !Session {
        const owned_base = try allocator.dupe(u8, base);
        errdefer allocator.free(owned_base);

        try ensureParentDir(io, owned_base);

        var opened = try wdbx.recovery.open(io, allocator, owned_base);
        errdefer opened.store.deinit();

        const wp = try wdbx.recovery.walPath(allocator, owned_base);
        errdefer allocator.free(wp);

        // Ensure a WAL tagged to the current checkpoint epoch exists so every
        // mutation this session logs is a recoverable delta. No-op if recovery
        // left a valid (already-merged) WAL in place; creates a fresh one if a
        // stale WAL was discarded or none existed.
        try wdbx.wal.createWithEpoch(io, allocator, wp, opened.checkpoint_epoch);

        opened.store.attachWal(io, wp);
        return .{
            .store = opened.store,
            .io = io,
            .allocator = allocator,
            .base_path = owned_base,
            .wal_path = wp,
        };
    }

    /// An in-memory session with no persistence (no WAL, no checkpoint).
    pub fn openInMemory(allocator: std.mem.Allocator) Session {
        return .{
            .store = wdbx.Store.init(allocator),
            .io = null,
            .allocator = allocator,
            .base_path = null,
            .wal_path = null,
        };
    }

    pub fn isPersistent(self: *const Session) bool {
        return self.base_path != null;
    }

    pub fn storePtr(self: *Session) *wdbx.Store {
        return &self.store;
    }

    /// Fold the WAL into a fresh segment checkpoint (with the monolithic
    /// snapshot mirror for explicit-path tooling), then drop the now-redundant
    /// WAL. The next mutation recreates the WAL. No-op for in-memory sessions.
    pub fn checkpoint(self: *Session) !void {
        const base = self.base_path orelse return;
        const io = self.io orelse return;
        _ = try checkpointAt(io, self.allocator, base, &self.store);
    }

    pub fn deinit(self: *Session) void {
        if (self.isPersistent()) {
            self.checkpoint() catch |err|
                std.log.warn("wdbx durable checkpoint failed: {s}", .{@errorName(err)});
        }
        self.store.deinit();
        if (self.wal_path) |wp| self.allocator.free(wp);
        if (self.base_path) |bp| self.allocator.free(bp);
        self.* = undefined;
    }
};

/// Checkpoint `store` at base path `base`: flush a new immutable segment, mirror
/// the monolithic snapshot for explicit-path tooling, then fold/reset the sidecar
/// WAL — delete the now-redundant log and start a fresh one tagged to the new
/// epoch. Returns the new epoch. Shared by the durable `Session` and the
/// explicit-path `abi wdbx` CLI handlers so both keep identical checkpoint
/// semantics (delta WAL + epoch-gated recovery).
pub fn checkpointAt(io: std.Io, allocator: std.mem.Allocator, base: []const u8, store: *const wdbx.Store) !u64 {
    var segment_store = wdbx.segments.SegmentStore.init(allocator, io, base);
    const new_epoch = try segment_store.flush(store);
    try wdbx.persistence.saveToPath(io, allocator, store, base);

    const wp = try wdbx.recovery.walPath(allocator, base);
    defer allocator.free(wp);
    // A crash before the delete leaves a stale (older-epoch) WAL that recovery
    // discards; a crash after it leaves a clean delta tagged to the new epoch.
    std.Io.Dir.cwd().deleteFile(io, wp) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    try wdbx.wal.createWithEpoch(io, allocator, wp, new_epoch);
    return new_epoch;
}

const testing = std.testing;

fn deleteIfExists(path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(testing.io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => std.debug.print("failed to delete test file '{s}': {s}\n", .{ path, @errorName(err) }),
    };
}

test "durable_store: in-memory session persists nothing" {
    var session = Session.openInMemory(testing.allocator);
    defer session.deinit();
    try testing.expect(!session.isPersistent());
    _ = try session.storePtr().appendBlock("abbey", 0, 0, "{\"t\":1}");
    try testing.expectEqual(@as(usize, 1), session.storePtr().blockCount());
}

test "durable_store: persistent round-trip recovers state after close" {
    const base = "zig-out/durable-roundtrip.jsonl";
    const wp = "zig-out/durable-roundtrip.jsonl.wal";
    const manifest = "zig-out/durable-roundtrip.jsonl.manifest";
    const seg0 = "zig-out/durable-roundtrip.jsonl.seg.0.jsonl";
    defer {
        deleteIfExists(base);
        deleteIfExists(wp);
        deleteIfExists(manifest);
        deleteIfExists(seg0);
    }
    deleteIfExists(base);
    deleteIfExists(wp);
    deleteIfExists(manifest);
    deleteIfExists(seg0);

    {
        var session = try Session.openAt(testing.io, testing.allocator, base);
        defer session.deinit(); // checkpoints on close
        try testing.expect(session.isPersistent());
        try session.storePtr().store("agent:abbey", "trained");
        _ = try session.storePtr().appendBlock("abbey", 1, 2, "{\"turn\":1}");
    }

    var reopened = try Session.openAt(testing.io, testing.allocator, base);
    defer reopened.deinit();
    try testing.expectEqual(@as(usize, 1), reopened.storePtr().blockCount());
    try testing.expectEqualStrings("trained", reopened.storePtr().get("agent:abbey").?);
}

test "durable_store: vector mutations survive a crash via the WAL delta" {
    const base = "zig-out/durable-vec-crash.jsonl";
    const wp = "zig-out/durable-vec-crash.jsonl.wal";
    const manifest = "zig-out/durable-vec-crash.jsonl.manifest";
    const seg0 = "zig-out/durable-vec-crash.jsonl.seg.0.jsonl";
    defer {
        deleteIfExists(base);
        deleteIfExists(wp);
        deleteIfExists(manifest);
        deleteIfExists(seg0);
    }
    deleteIfExists(base);
    deleteIfExists(wp);
    deleteIfExists(manifest);
    deleteIfExists(seg0);

    // Session inserts two vectors, then "crashes" — freed WITHOUT checkpointing,
    // so only the WAL (no segment) holds them.
    {
        var session = try Session.openAt(testing.io, testing.allocator, base);
        _ = try session.storePtr().putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
        _ = try session.storePtr().putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
        session.store.deinit();
        if (session.wal_path) |p| testing.allocator.free(p);
        if (session.base_path) |p| testing.allocator.free(p);
        // Intentionally NOT session.checkpoint()/deinit(): simulate a crash.
    }

    // Reopen: recovery folds the WAL delta onto the (empty) checkpoint.
    var reopened = try Session.openAt(testing.io, testing.allocator, base);
    defer reopened.deinit();
    try testing.expectEqual(@as(usize, 2), reopened.storePtr().vectorCount());
}

test "durable_store: store parent directory defaults to owner-only mode on POSIX" {
    if (builtin.target.os.tag == .windows) return;
    if (!comptime std.Io.File.Permissions.has_executable_bit) return;

    const base = "zig-out/g5-owner-only/store.jsonl";
    const parent = "zig-out/g5-owner-only";
    defer {
        deleteIfExists(base);
        deleteIfExists("zig-out/g5-owner-only/store.jsonl.wal");
        deleteIfExists("zig-out/g5-owner-only/store.jsonl.manifest");
        std.Io.Dir.deleteTree(.cwd(), testing.io, parent) catch {};
    }
    std.Io.Dir.deleteTree(.cwd(), testing.io, parent) catch {};

    {
        var session = try Session.openAt(testing.io, testing.allocator, base);
        defer session.deinit();
        try testing.expect(session.isPersistent());
    }

    // Repair path opens with iterate=true (non-O_PATH); verify mode via same
    // open style so Linux hosts exercise the fchmod-capable fd path.
    const dir = try std.Io.Dir.cwd().openDir(testing.io, parent, .{ .iterate = true });
    defer dir.close(testing.io);
    const stat = try dir.stat(testing.io);
    const mode = stat.permissions.toMode() & 0o777;
    try testing.expectEqual(@as(std.posix.mode_t, 0o700), mode);
}

test "durable_store: ensureOwnerOnlyDir repairs pre-existing world-writable parent" {
    if (builtin.target.os.tag == .windows) return;
    if (!comptime std.Io.File.Permissions.has_executable_bit) return;

    const parent = "zig-out/g5-owner-only-repair";
    const base = "zig-out/g5-owner-only-repair/store.jsonl";
    defer {
        deleteIfExists(base);
        deleteIfExists("zig-out/g5-owner-only-repair/store.jsonl.wal");
        deleteIfExists("zig-out/g5-owner-only-repair/store.jsonl.manifest");
        std.Io.Dir.deleteTree(.cwd(), testing.io, parent) catch {};
    }
    std.Io.Dir.deleteTree(.cwd(), testing.io, parent) catch {};

    // Pre-create a world-writable parent so createDirPathStatus is a no-op and
    // the repair path (open + setPermissions) must do the work.
    _ = try std.Io.Dir.createDirPathStatus(.cwd(), testing.io, parent, std.Io.File.Permissions.fromMode(0o777));

    {
        var session = try Session.openAt(testing.io, testing.allocator, base);
        defer session.deinit();
        try testing.expect(session.isPersistent());
    }

    const dir = try std.Io.Dir.cwd().openDir(testing.io, parent, .{ .iterate = true });
    defer dir.close(testing.io);
    const stat = try dir.stat(testing.io);
    const mode = stat.permissions.toMode() & 0o777;
    try testing.expectEqual(@as(std.posix.mode_t, 0o700), mode);
}

test "durable_store: :memory: sentinel and falsey persist flag map to in-memory" {
    // resolveConfig reads the environment; exercise the pure classifier helpers
    // that drive it so the test does not mutate process env.
    try testing.expect(isFalsey("0"));
    try testing.expect(isFalsey("false"));
    try testing.expect(!isFalsey("1"));
    try testing.expect(std.mem.eql(u8, MEMORY_SENTINEL, ":memory:"));
}

test {
    testing.refAllDecls(@This());
}
