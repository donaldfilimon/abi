//! WDBX segment persistence for multiway experiment exports.
const std = @import("std");
const builtin = @import("builtin");
const persistence = @import("persistence.zig");
const recovery = @import("recovery.zig");
const types = @import("multiway_types.zig");
const export_mod = @import("multiway_export.zig");

const Config = types.Config;
const Result = types.Result;
const configHash = export_mod.configHash;
const exportHashHex = export_mod.exportHashHex;

pub const EXPORT_KEY_LATEST = "multiway:experiment:latest";

/// Persist an experiment into a WDBX segment checkpoint at `path`.
///
/// Layout: each canonical state payload is stored content-addressed under
/// `multiway:state:<hex-hash>` (duplicate payloads across experiments share
/// one entry); the full canonical export is stored under
/// `multiway:experiment:<config-hash-hex>` plus the `latest` alias; and one
/// SHA-linked conversation block records provenance (config hash, export
/// hash, counts, termination, zig version). `persistence.saveToPath` writes a
/// new segment checkpoint and manifest atomically, so an interrupted write
/// leaves the previous checkpoint intact rather than a half-written
/// experiment.
pub fn persistToWdbx(io: std.Io, allocator: std.mem.Allocator, path: []const u8, config: Config, result: *const Result, export_json: []const u8) !void {
    // recovery.open returns an empty store (source == .empty) for a fresh
    // path; real corruption errors propagate rather than being masked.
    var opened = try recovery.open(io, allocator, path);
    defer opened.store.deinit();

    for (result.states.items) |state| {
        const hex = std.fmt.bytesToHex(state.hash, .lower);
        const key = try std.fmt.allocPrint(allocator, "multiway:state:{s}", .{hex});
        defer allocator.free(key);
        try opened.store.store(key, state.payload);
    }

    const cfg_hash = try configHash(allocator, config);
    const cfg_hex = std.fmt.bytesToHex(cfg_hash, .lower);
    {
        const key = try std.fmt.allocPrint(allocator, "multiway:experiment:{s}", .{cfg_hex});
        defer allocator.free(key);
        try opened.store.store(key, export_json);
        try opened.store.store(EXPORT_KEY_LATEST, export_json);
    }

    const export_hex = exportHashHex(export_json);
    const block_meta = try std.fmt.allocPrint(
        allocator,
        "{{\"kind\":\"multiway_experiment\",\"config_hash\":\"{s}\",\"export_hash\":\"{s}\",\"states\":{d},\"events\":{d},\"termination\":\"{s}\",\"complete\":{},\"zig_version\":\"{s}\"}}",
        .{ cfg_hex, export_hex, result.states.items.len, result.events.items.len, result.termination.label(), result.complete, builtin.zig_version_string },
    );
    defer allocator.free(block_meta);
    _ = try opened.store.appendBlock("multiway", 0, 0, block_meta);

    try persistence.saveToPath(io, allocator, &opened.store, path);
}

/// Load a persisted canonical export back out of a WDBX checkpoint. Pass
/// null `config_hash_hex` for the most recent experiment. Caller owns the
/// returned bytes.
pub fn loadExportFromWdbx(io: std.Io, allocator: std.mem.Allocator, path: []const u8, config_hash_hex: ?[]const u8) ![]u8 {
    var opened = try recovery.open(io, allocator, path);
    defer opened.store.deinit();
    const key = if (config_hash_hex) |hex|
        try std.fmt.allocPrint(allocator, "multiway:experiment:{s}", .{hex})
    else
        try allocator.dupe(u8, EXPORT_KEY_LATEST);
    defer allocator.free(key);
    const value = opened.store.get(key) orelse return error.ExperimentNotFound;
    return allocator.dupe(u8, value);
}

test {
    std.testing.refAllDecls(@This());
}
