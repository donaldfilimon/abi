//! MCP Status / Diagnostics Tool Handlers
//!
//! Handlers for `abi_status`, `abi_health`, `abi_features`, `abi_version`,
//! and `hardware_status` tools.

const std = @import("std");
const build_options = @import("build_options");
const types = @import("../types.zig");

pub fn handleAbiStatus(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ABI MCP Server Status: running");
}

pub fn handleAbiHealth(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ok");
}

pub fn handleAbiFeatures(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "Enabled features:");
    if (build_options.feat_gpu) try out.appendSlice(allocator, " gpu");
    if (build_options.feat_ai) try out.appendSlice(allocator, " ai");
    if (build_options.feat_database) try out.appendSlice(allocator, " database");
    if (build_options.feat_network) try out.appendSlice(allocator, " network");
    if (build_options.feat_observability) try out.appendSlice(allocator, " observability");
    if (build_options.feat_web) try out.appendSlice(allocator, " web");
    if (build_options.feat_cloud) try out.appendSlice(allocator, " cloud");
    if (build_options.feat_auth) try out.appendSlice(allocator, " auth");
    if (build_options.feat_messaging) try out.appendSlice(allocator, " messaging");
    if (build_options.feat_cache) try out.appendSlice(allocator, " cache");
    if (build_options.feat_storage) try out.appendSlice(allocator, " storage");
    if (build_options.feat_search) try out.appendSlice(allocator, " search");
    if (build_options.feat_mobile) try out.appendSlice(allocator, " mobile");
    if (build_options.feat_gateway) try out.appendSlice(allocator, " gateway");
    if (build_options.feat_pages) try out.appendSlice(allocator, " pages");
    if (build_options.feat_benchmarks) try out.appendSlice(allocator, " benchmarks");
    if (build_options.feat_compute) try out.appendSlice(allocator, " compute");
    if (build_options.feat_documents) try out.appendSlice(allocator, " documents");
    if (build_options.feat_desktop) try out.appendSlice(allocator, " desktop");
    if (build_options.feat_lsp) try out.appendSlice(allocator, " lsp");
    if (build_options.feat_mcp) try out.appendSlice(allocator, " mcp");
}

pub fn handleAbiVersion(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    try out.appendSlice(allocator, "ABI version: ");
    try out.appendSlice(allocator, build_options.package_version);
    try out.appendSlice(allocator, "\nProtocol: ");
    try out.appendSlice(allocator, types.PROTOCOL_VERSION);
    try out.appendSlice(allocator, "\nZig: 0.16.0-dev");
}

pub fn handleHardwareStatus(
    allocator: std.mem.Allocator,
    _: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) !void {
    const discovery = @import("../../../features/ai/explore/discovery.zig");
    const caps = discovery.detectCapabilities();
    const json_str = try std.json.Stringify.valueAlloc(allocator, caps, .{});
    defer allocator.free(json_str);
    try out.appendSlice(allocator, json_str);
}

test {
    std.testing.refAllDecls(@This());
}
