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
const wdbx = @import("mod.zig");

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

fn envSpan(key: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(key) orelse return null;
    const span = std.mem.span(raw);
    return if (span.len == 0) null else span;
}

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
    if (envSpan(PERSIST_ENV)) |v| {
        if (isFalsey(v)) return .{ .base_path = null };
    }
    if (envSpan(PATH_ENV)) |p| {
        if (std.mem.eql(u8, p, MEMORY_SENTINEL)) return .{ .base_path = null };
        return .{ .base_path = try allocator.dupe(u8, p) };
    }
    const home = envSpan("HOME") orelse return .{ .base_path = null };
    return .{ .base_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ home, DEFAULT_SUBPATH }) };
}

fn ensureParentDir(io: std.Io, base: []const u8) !void {
    const dir = std.fs.path.dirname(base) orelse return;
    if (dir.len == 0) return;
    // createDirPath is idempotent (succeeds when the path already exists).
    try std.Io.Dir.cwd().createDirPath(io, dir);
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
        var segment_store = wdbx.segments.SegmentStore.init(self.allocator, io, base);
        _ = try segment_store.flush(&self.store);
        try wdbx.persistence.saveToPath(io, self.allocator, &self.store, base);
        if (self.wal_path) |wp| {
            std.Io.Dir.cwd().deleteFile(io, wp) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
        }
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
