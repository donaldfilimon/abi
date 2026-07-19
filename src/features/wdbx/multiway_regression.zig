//! Reference regression experiment for the multiway simulator.
//!
//! Contains a deliberately independent, minimal breadth-first rewriting
//! implementation (string-keyed maps, positional scanning — no code shared
//! with the production engine in multiway.zig) and cross-validates both
//! implementations against externally verified expected values for:
//!
//!   initial "A"; rules A->AB, A->BA, BB->A; depth 5; overlapping matches.
//!
//! Expected values (unique states per depth 1,2,3,5,8,12; 31 states; 62
//! unique transitions; mean out-degree 2.0; max out-degree 5; one weakly
//! connected component; acyclic) were additionally verified with an external
//! Python oracle before being encoded here. Assertions compare canonical
//! content (payload sets, transition sets), never implementation-specific
//! discovery identifiers.

const std = @import("std");
const multiway = @import("multiway.zig");

/// Independent minimal reference: breadth-first exact string rewriting with
/// canonical dedup by payload equality. Uses per-position scanning (not
/// indexOfPos) so match discovery is implemented differently from the
/// production engine.
const Reference = struct {
    allocator: std.mem.Allocator,
    /// payload -> discovery depth; keys owned.
    depth_of: std.StringHashMapUnmanaged(u32) = .empty,
    /// "src\x00dst" -> {}; keys owned. Payloads in the reference experiment
    /// never contain NUL, so the separator is unambiguous.
    transitions: std.StringHashMapUnmanaged(void) = .empty,
    event_count: usize = 0,
    states_per_depth: [16]u32 = @splat(0),

    fn deinit(self: *Reference) void {
        var key_it = self.depth_of.keyIterator();
        while (key_it.next()) |key| self.allocator.free(key.*);
        self.depth_of.deinit(self.allocator);
        var edge_it = self.transitions.keyIterator();
        while (edge_it.next()) |key| self.allocator.free(key.*);
        self.transitions.deinit(self.allocator);
    }

    fn matchesAt(haystack: []const u8, pos: usize, needle: []const u8) bool {
        if (pos + needle.len > haystack.len) return false;
        return std.mem.eql(u8, haystack[pos .. pos + needle.len], needle);
    }

    fn evolve(self: *Reference, initial: []const u8, rules: []const [2][]const u8, max_depth: u32) !void {
        var frontier: std.ArrayListUnmanaged([]const u8) = .empty;
        defer frontier.deinit(self.allocator);

        const initial_owned = try self.allocator.dupe(u8, initial);
        try self.depth_of.put(self.allocator, initial_owned, 0);
        try frontier.append(self.allocator, initial_owned);
        self.states_per_depth[0] = 1;

        var depth: u32 = 0;
        while (depth < max_depth and frontier.items.len > 0) : (depth += 1) {
            var next: std.ArrayListUnmanaged([]const u8) = .empty;
            defer next.deinit(self.allocator);
            for (frontier.items) |src| {
                for (rules) |rule| {
                    const lhs = rule[0];
                    const rhs = rule[1];
                    var pos: usize = 0;
                    while (pos + lhs.len <= src.len) : (pos += 1) {
                        if (!matchesAt(src, pos, lhs)) continue;
                        self.event_count += 1;
                        const dst = try std.mem.concat(self.allocator, u8, &.{ src[0..pos], rhs, src[pos + lhs.len ..] });
                        defer self.allocator.free(dst);

                        const edge_key = try std.mem.concat(self.allocator, u8, &.{ src, "\x00", dst });
                        const edge_gop = try self.transitions.getOrPut(self.allocator, edge_key);
                        if (edge_gop.found_existing) {
                            self.allocator.free(edge_key);
                        }

                        if (!self.depth_of.contains(dst)) {
                            const dst_owned = try self.allocator.dupe(u8, dst);
                            try self.depth_of.put(self.allocator, dst_owned, depth + 1);
                            try next.append(self.allocator, dst_owned);
                            self.states_per_depth[depth + 1] += 1;
                        }
                    }
                }
            }
            frontier.clearRetainingCapacity();
            try frontier.appendSlice(self.allocator, next.items);
        }
    }
};

fn sortedPayloads(allocator: std.mem.Allocator, payloads: [][]const u8) void {
    _ = allocator;
    std.mem.sort([]const u8, payloads, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lessThan);
}

const REFERENCE_RULES = [_]multiway.Rule{
    .{ .lhs = "A", .rhs = "AB" },
    .{ .lhs = "A", .rhs = "BA" },
    .{ .lhs = "BB", .rhs = "A" },
};

pub const REFERENCE_CONFIG = multiway.Config{
    .initial = &.{"A"},
    .rules = &REFERENCE_RULES,
    .max_depth = 5,
    .max_states = 500,
    .max_events = 5000,
    .max_payload = 64,
};

test "reference regression: production engine matches externally verified values" {
    const allocator = std.testing.allocator;
    var result = try multiway.run(allocator, REFERENCE_CONFIG, null);
    defer result.deinit();
    var metrics = try multiway.computeMetrics(allocator, &result);
    defer metrics.deinit();

    try std.testing.expectEqual(multiway.Termination.max_depth, result.termination);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 8, 12 }, metrics.states_per_depth);
    try std.testing.expectEqual(@as(u32, 31), metrics.unique_states);
    try std.testing.expectEqual(@as(u32, 62), metrics.unique_transitions);
    try std.testing.expectEqual(@as(u32, 66), metrics.event_count);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), metrics.mean_out_degree, 1e-12);
    try std.testing.expectEqual(@as(u32, 5), metrics.max_out_degree);
    try std.testing.expectEqual(@as(u32, 1), metrics.weakly_connected_components);
    try std.testing.expect(!metrics.has_cycle);
    try std.testing.expectEqual(@as(u32, 0), metrics.self_loops);
    try std.testing.expectEqual(@as(u32, 20), metrics.convergent_states);
    try std.testing.expect(!metrics.exhaustive);
}

test "reference regression: independent implementation agrees with production engine" {
    const allocator = std.testing.allocator;

    var reference = Reference{ .allocator = allocator };
    defer reference.deinit();
    try reference.evolve("A", &.{ .{ "A", "AB" }, .{ "A", "BA" }, .{ "BB", "A" } }, 5);

    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 5, 8, 12 }, reference.states_per_depth[0..6]);
    try std.testing.expectEqual(@as(usize, 31), reference.depth_of.count());
    try std.testing.expectEqual(@as(usize, 62), reference.transitions.count());
    try std.testing.expectEqual(@as(usize, 66), reference.event_count);

    var result = try multiway.run(allocator, REFERENCE_CONFIG, null);
    defer result.deinit();

    // Compare canonical state content (sorted payloads), never discovery ids.
    var production_payloads = try allocator.alloc([]const u8, result.states.items.len);
    defer allocator.free(production_payloads);
    for (result.states.items, 0..) |state, i| production_payloads[i] = state.payload;
    sortedPayloads(allocator, production_payloads);

    var reference_payloads = try allocator.alloc([]const u8, reference.depth_of.count());
    defer allocator.free(reference_payloads);
    {
        var i: usize = 0;
        var it = reference.depth_of.keyIterator();
        while (it.next()) |key| : (i += 1) reference_payloads[i] = key.*;
        sortedPayloads(allocator, reference_payloads);
    }

    try std.testing.expectEqual(reference_payloads.len, production_payloads.len);
    for (production_payloads, reference_payloads) |prod, ref| {
        try std.testing.expectEqualStrings(ref, prod);
    }

    // Discovery depths must agree state-by-state (content-keyed lookup).
    for (result.states.items) |state| {
        const ref_depth = reference.depth_of.get(state.payload) orelse return error.MissingReferenceState;
        try std.testing.expectEqual(ref_depth, state.depth);
    }

    // Transition sets must agree, compared by payload content.
    var transition_checks: usize = 0;
    for (result.events.items) |event| {
        const src = result.states.items[event.src].payload;
        const dst = result.states.items[event.dst].payload;
        const key = try std.mem.concat(allocator, u8, &.{ src, "\x00", dst });
        defer allocator.free(key);
        try std.testing.expect(reference.transitions.contains(key));
        transition_checks += 1;
    }
    try std.testing.expect(transition_checks == result.events.items.len);
}

test "reference regression: canonical export is stable across repeated runs" {
    const allocator = std.testing.allocator;
    var hashes: [2][64]u8 = undefined;
    for (&hashes) |*slot| {
        var result = try multiway.run(allocator, REFERENCE_CONFIG, null);
        defer result.deinit();
        var metrics = try multiway.computeMetrics(allocator, &result);
        defer metrics.deinit();
        const export_json = try multiway.exportCanonicalJson(allocator, REFERENCE_CONFIG, &result, &metrics);
        defer allocator.free(export_json);
        slot.* = multiway.exportHashHex(export_json);
    }
    try std.testing.expectEqualStrings(&hashes[0], &hashes[1]);
}

test {
    std.testing.refAllDecls(@This());
}
