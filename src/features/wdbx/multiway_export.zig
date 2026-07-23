//! Canonical JSON and Graphviz DOT export for multiway experiments.
const std = @import("std");
const builtin = @import("builtin");
const json_util = @import("../../foundation/json.zig");
const types = @import("multiway_types.zig");
const metrics_mod = @import("multiway_metrics.zig");

const Config = types.Config;
const Result = types.Result;
const CausalEdge = types.CausalEdge;
const FORMAT_VERSION = types.FORMAT_VERSION;
const hashPayload = types.hashPayload;
const Metrics = metrics_mod.Metrics;

/// Derive rigorous event→event causal edges from token lineage.
/// For each event, parents are distinct producer-events of the matched LHS
/// bytes on the source state's reconstructed lineage.
pub fn buildCausalEdges(allocator: std.mem.Allocator, config: Config, result: *const Result) ![]CausalEdge {
    const n_states = result.states.items.len;
    var lineage = try allocator.alloc([]?u32, n_states);
    errdefer {
        for (lineage) |row| allocator.free(row);
        allocator.free(lineage);
    }
    for (result.states.items, 0..) |state, i| {
        lineage[i] = try allocator.alloc(?u32, state.payload.len);
        @memset(lineage[i], null);
    }

    var edges: std.ArrayListUnmanaged(CausalEdge) = .empty;
    errdefer edges.deinit(allocator);

    for (result.events.items) |event| {
        if (event.rule >= config.rules.len) continue;
        const rule = config.rules[event.rule];
        const src_idx = event.src;
        const dst_idx = event.dst;
        if (src_idx >= n_states or dst_idx >= n_states) continue;
        const src_lin = lineage[src_idx];
        const lhs_len = rule.lhs.len;
        const rhs_len = rule.rhs.len;
        const pos: usize = event.pos;
        if (pos + lhs_len > src_lin.len) continue;

        var parent_seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer parent_seen.deinit(allocator);
        var i: usize = pos;
        while (i < pos + lhs_len) : (i += 1) {
            if (src_lin[i]) |parent| {
                const gop = try parent_seen.getOrPut(allocator, parent);
                if (!gop.found_existing) {
                    try edges.append(allocator, .{ .parent = parent, .child = event.id });
                }
            }
        }

        // First writer of a destination owns its lineage snapshot.
        if (result.states.items[dst_idx].first_event) |fe| {
            if (fe == event.id) {
                const dst_len = result.states.items[dst_idx].payload.len;
                const expected = src_lin.len - lhs_len + rhs_len;
                if (expected != dst_len) continue;
                const dst_lin = lineage[dst_idx];
                @memcpy(dst_lin[0..pos], src_lin[0..pos]);
                @memset(dst_lin[pos .. pos + rhs_len], event.id);
                if (pos + lhs_len < src_lin.len) {
                    @memcpy(dst_lin[pos + rhs_len ..], src_lin[pos + lhs_len ..]);
                }
            }
        }
    }

    for (lineage) |row| allocator.free(row);
    allocator.free(lineage);
    return try edges.toOwnedSlice(allocator);
}

fn appendFmt(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const chunk = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(chunk);
    try out.appendSlice(allocator, chunk);
}

fn appendHexHash(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, digest: [32]u8) !void {
    const hex = std.fmt.bytesToHex(digest, .lower);
    try out.appendSlice(allocator, &hex);
}

fn appendConfigJson(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, config: Config) !void {
    try out.appendSlice(allocator, "{\"initial\":[");
    for (config.initial, 0..) |payload, i| {
        if (i > 0) try out.append(allocator, ',');
        try json_util.appendJsonString(out, allocator, payload);
    }
    try out.appendSlice(allocator, "],\"rules\":[");
    for (config.rules, 0..) |rule, i| {
        if (i > 0) try out.append(allocator, ',');
        const canonical = try rule.canonicalText(allocator);
        defer allocator.free(canonical);
        try out.appendSlice(allocator, "{\"id\":");
        try appendFmt(out, allocator, "{d}", .{i});
        try out.appendSlice(allocator, ",\"rule\":");
        try json_util.appendJsonString(out, allocator, canonical);
        try appendFmt(out, allocator, ",\"weight\":{d}", .{rule.weight});
        if (rule.family) |family| {
            try out.appendSlice(allocator, ",\"family\":");
            try json_util.appendJsonString(out, allocator, family);
        }
        try out.appendSlice(allocator, ",\"hash\":\"");
        try appendHexHash(out, allocator, rule.contentHash());
        try out.appendSlice(allocator, "\"}");
    }
    try appendFmt(out, allocator, "],\"max_depth\":{d},\"max_states\":{d},\"max_events\":{d},\"max_payload\":{d},\"max_duration_ms\":{d},\"max_memory_bytes\":{d},\"traversal\":\"{s}\",\"canonicalization\":\"{s}\",\"dedup\":\"{s}\",\"seed\":{d},\"workers\":{d}}}", .{
        config.max_depth,
        config.max_states,
        config.max_events,
        config.max_payload,
        config.max_duration_ms,
        config.max_memory_bytes,
        @tagName(config.traversal),
        @tagName(config.canonicalization),
        @tagName(config.dedup),
        config.seed,
        config.workers,
    });
}

/// SHA-256 of the canonical config JSON — the experiment's identity.
pub fn configHash(allocator: std.mem.Allocator, config: Config) ![32]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);
    try appendConfigJson(&buf, allocator, config);
    return hashPayload(buf.items);
}

/// Serialize the complete experiment as canonical JSON: stable field order,
/// no whitespace, states in discovery order, events in commit order. The
/// export includes the resume cursor so a bounded run can continue later.
/// Wall-clock timing is deliberately excluded so identical runs export
/// byte-for-byte identical documents.
pub fn exportCanonicalJson(allocator: std.mem.Allocator, config: Config, result: *const Result, metrics: *const Metrics) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    try appendFmt(&out, allocator, "{{\"format\":\"{s}\",\"zig_version\":\"{s}\",\"config\":", .{ FORMAT_VERSION, builtin.zig_version_string });
    try appendConfigJson(&out, allocator, config);
    try out.appendSlice(allocator, ",\"config_hash\":\"");
    try appendHexHash(&out, allocator, try configHash(allocator, config));
    try out.appendSlice(allocator, "\",\"states\":[");
    for (result.states.items, 0..) |state, i| {
        if (i > 0) try out.append(allocator, ',');
        try out.appendSlice(allocator, "{\"payload\":");
        try json_util.appendJsonString(&out, allocator, state.payload);
        try out.appendSlice(allocator, ",\"hash\":\"");
        try appendHexHash(&out, allocator, state.hash);
        try appendFmt(&out, allocator, "\",\"depth\":{d}", .{state.depth});
        if (state.first_event) |event_id| {
            try appendFmt(&out, allocator, ",\"first_event\":{d}", .{event_id});
        }
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "],\"events\":[");
    for (result.events.items, 0..) |event, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(&out, allocator, "{{\"src\":{d},\"dst\":{d},\"rule\":{d},\"pos\":{d},\"depth\":{d},\"local\":{d}}}", .{ event.src, event.dst, event.rule, event.pos, event.depth, event.local });
    }
    try out.appendSlice(allocator, "],\"metrics\":");
    try appendMetricsJson(&out, allocator, metrics);
    try appendFmt(&out, allocator, ",\"termination\":\"{s}\",\"complete\":{}", .{ result.termination.label(), result.complete });
    const causal = try buildCausalEdges(allocator, config, result);
    defer allocator.free(causal);
    try out.appendSlice(allocator, ",\"causal_graph\":{\"status\":\"token-lineage\",\"edges\":[");
    for (causal, 0..) |edge, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(&out, allocator, "{{\"parent\":{d},\"child\":{d}}}", .{ edge.parent, edge.child });
    }
    try out.appendSlice(allocator, "]}");
    if (result.complete) {
        try out.appendSlice(allocator, ",\"resume\":null");
    } else {
        try appendFmt(&out, allocator, ",\"resume\":{{\"depth\":{d},\"cursor\":{d},\"frontier\":[", .{ result.resume_depth, result.cursor });
        for (result.frontier.items, 0..) |idx, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendFmt(&out, allocator, "{d}", .{idx});
        }
        try out.appendSlice(allocator, "],\"next_frontier\":[");
        for (result.next_frontier.items, 0..) |idx, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendFmt(&out, allocator, "{d}", .{idx});
        }
        try out.appendSlice(allocator, "]}");
    }
    try out.appendSlice(allocator, "}");
    return out.toOwnedSlice(allocator);
}

fn appendMetricsJson(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, metrics: *const Metrics) !void {
    try appendFmt(out, allocator, "{{\"unique_states\":{d},\"event_count\":{d},\"unique_transitions\":{d},\"states_per_depth\":[", .{ metrics.unique_states, metrics.event_count, metrics.unique_transitions });
    for (metrics.states_per_depth, 0..) |count, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(out, allocator, "{d}", .{count});
    }
    try out.appendSlice(allocator, "],\"events_per_depth\":[");
    for (metrics.events_per_depth, 0..) |count, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(out, allocator, "{d}", .{count});
    }
    try out.appendSlice(allocator, "],\"frontier_width_per_depth\":[");
    for (metrics.frontier_width_per_depth, 0..) |count, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(out, allocator, "{d}", .{count});
    }
    try appendFmt(out, allocator, "],\"mean_out_degree\":{d},\"max_out_degree\":{d},\"median_out_degree\":{d},\"convergent_states\":{d},\"self_loops\":{d},\"has_cycle\":{},\"weakly_connected_components\":{d},\"max_payload_bytes\":{d},\"mean_payload_bytes\":{d},\"growth_rates\":[", .{
        metrics.mean_out_degree,
        metrics.max_out_degree,
        metrics.median_out_degree,
        metrics.convergent_states,
        metrics.self_loops,
        metrics.has_cycle,
        metrics.weakly_connected_components,
        metrics.max_payload_bytes,
        metrics.mean_payload_bytes,
    });
    for (metrics.growth_rates, 0..) |rate, i| {
        if (i > 0) try out.append(allocator, ',');
        try appendFmt(out, allocator, "{d}", .{rate});
    }
    try appendFmt(out, allocator, "],\"termination\":\"{s}\",\"exhaustive\":{}}}", .{ metrics.termination.label(), metrics.exhaustive });
}

/// SHA-256 hex digest of an export document.
pub fn exportHashHex(export_bytes: []const u8) [64]u8 {
    return std.fmt.bytesToHex(hashPayload(export_bytes), .lower);
}

fn appendDotEscaped(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, text: []const u8) !void {
    for (text) |c| {
        switch (c) {
            '"', '\\' => {
                try out.append(allocator, '\\');
                try out.append(allocator, c);
            },
            '\n' => try out.appendSlice(allocator, "\\n"),
            else => try out.append(allocator, c),
        }
    }
}

/// Graphviz DOT rendering of the state-transition multigraph. Every event is
/// emitted as its own edge (multiplicity preserved), labeled with its rule id
/// and match offset.
pub fn exportDot(allocator: std.mem.Allocator, config: Config, result: *const Result) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "digraph multiway {\n  rankdir=LR;\n  node [shape=box,fontname=\"monospace\"];\n");
    for (result.states.items) |state| {
        try appendFmt(&out, allocator, "  s{d} [label=\"", .{state.seq});
        try appendDotEscaped(&out, allocator, state.payload);
        try appendFmt(&out, allocator, "\\nd{d}\"];\n", .{state.depth});
    }
    for (result.events.items) |event| {
        const rule = config.rules[event.rule];
        try appendFmt(&out, allocator, "  s{d} -> s{d} [label=\"", .{ event.src, event.dst });
        try appendDotEscaped(&out, allocator, rule.lhs);
        // ASCII arrow: a quoted DOT label is plain text, so "->" renders as-is
        // (Graphviz does not interpret backslash-x byte escapes here).
        try out.appendSlice(allocator, "->");
        try appendDotEscaped(&out, allocator, rule.rhs);
        try appendFmt(&out, allocator, "@{d}\"];\n", .{event.pos});
    }
    try out.appendSlice(allocator, "}\n");
    return out.toOwnedSlice(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
