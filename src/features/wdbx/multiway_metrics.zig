//! Derived structural metrics for multiway experiment results.
const std = @import("std");
const types = @import("multiway_types.zig");

const Result = types.Result;
const Termination = types.Termination;

/// Derived structural metrics. Denominators: `mean_out_degree` is
/// unique transitions / unique states (terminal states count with degree 0);
/// `median_out_degree` is the median of per-state distinct-destination
/// out-degrees over ALL unique states; `mean_payload` averages over all
/// unique states; `growth_rates[d]` is states discovered at depth d+1 divided
/// by states discovered at depth d (only defined while the earlier depth is
/// non-empty). No metric here is a "computational complexity" claim.
pub const Metrics = struct {
    allocator: std.mem.Allocator,
    unique_states: u32,
    event_count: u32,
    unique_transitions: u32,
    states_per_depth: []u32,
    events_per_depth: []u32,
    frontier_width_per_depth: []u32,
    mean_out_degree: f64,
    max_out_degree: u32,
    median_out_degree: f64,
    convergent_states: u32,
    self_loops: u32,
    has_cycle: bool,
    weakly_connected_components: u32,
    max_payload_bytes: u32,
    mean_payload_bytes: f64,
    growth_rates: []f64,
    termination: Termination,
    exhaustive: bool,

    pub fn deinit(self: *Metrics) void {
        self.allocator.free(self.states_per_depth);
        self.allocator.free(self.events_per_depth);
        self.allocator.free(self.frontier_width_per_depth);
        self.allocator.free(self.growth_rates);
    }
};

fn transitionKey(src: u32, dst: u32) u64 {
    return (@as(u64, src) << 32) | @as(u64, dst);
}

pub fn computeMetrics(allocator: std.mem.Allocator, result: *const Result) !Metrics {
    const n_states = result.states.items.len;

    var transitions: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer transitions.deinit(allocator);
    for (result.events.items) |event| {
        try transitions.put(allocator, transitionKey(event.src, event.dst), {});
    }

    const out_degree = try allocator.alloc(u32, n_states);
    defer allocator.free(out_degree);
    @memset(out_degree, 0);
    const in_degree = try allocator.alloc(u32, n_states);
    defer allocator.free(in_degree);
    @memset(in_degree, 0);
    var self_loops: u32 = 0;
    {
        var it = transitions.keyIterator();
        while (it.next()) |key| {
            const src: u32 = @intCast(key.* >> 32);
            const dst: u32 = @intCast(key.* & 0xffff_ffff);
            out_degree[src] += 1;
            in_degree[dst] += 1;
            if (src == dst) self_loops += 1;
        }
    }

    var max_out: u32 = 0;
    var convergent: u32 = 0;
    for (out_degree) |deg| max_out = @max(max_out, deg);
    for (in_degree) |deg| {
        if (deg > 1) convergent += 1;
    }

    const median = blk: {
        if (n_states == 0) break :blk 0.0;
        const sorted = try allocator.dupe(u32, out_degree);
        defer allocator.free(sorted);
        std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
        const mid = n_states / 2;
        if (n_states % 2 == 1) break :blk @as(f64, @floatFromInt(sorted[mid]));
        break :blk (@as(f64, @floatFromInt(sorted[mid - 1])) + @as(f64, @floatFromInt(sorted[mid]))) / 2.0;
    };

    // Weakly connected components via union-find over unique transitions.
    const parent = try allocator.alloc(u32, n_states);
    defer allocator.free(parent);
    for (parent, 0..) |*p, i| p.* = @intCast(i);
    const find = struct {
        fn find(par: []u32, x0: u32) u32 {
            var x = x0;
            while (par[x] != x) {
                par[x] = par[par[x]];
                x = par[x];
            }
            return x;
        }
    }.find;
    {
        var it = transitions.keyIterator();
        while (it.next()) |key| {
            const src: u32 = @intCast(key.* >> 32);
            const dst: u32 = @intCast(key.* & 0xffff_ffff);
            const rs = find(parent, src);
            const rd = find(parent, dst);
            if (rs != rd) parent[rs] = rd;
        }
    }
    var components: u32 = 0;
    for (0..n_states) |i| {
        if (find(parent, @intCast(i)) == @as(u32, @intCast(i))) components += 1;
    }

    // Cycle detection: iterative 3-color DFS over the unique-transition graph.
    const has_cycle = blk: {
        if (n_states == 0) break :blk false;
        var adj_heads = try allocator.alloc(std.ArrayListUnmanaged(u32), n_states);
        defer {
            for (adj_heads) |*list| list.deinit(allocator);
            allocator.free(adj_heads);
        }
        for (adj_heads) |*list| list.* = .empty;
        var it = transitions.keyIterator();
        while (it.next()) |key| {
            const src: u32 = @intCast(key.* >> 32);
            const dst: u32 = @intCast(key.* & 0xffff_ffff);
            try adj_heads[src].append(allocator, dst);
        }
        const colors = try allocator.alloc(u8, n_states);
        defer allocator.free(colors);
        @memset(colors, 0); // 0 white, 1 gray, 2 black
        const Frame = struct { node: u32, next_child: u32 };
        var stack: std.ArrayListUnmanaged(Frame) = .empty;
        defer stack.deinit(allocator);
        for (0..n_states) |root| {
            if (colors[root] != 0) continue;
            try stack.append(allocator, .{ .node = @intCast(root), .next_child = 0 });
            colors[root] = 1;
            while (stack.items.len > 0) {
                const frame = &stack.items[stack.items.len - 1];
                const children = adj_heads[frame.node].items;
                if (frame.next_child < children.len) {
                    const child = children[frame.next_child];
                    frame.next_child += 1;
                    if (colors[child] == 1) break :blk true;
                    if (colors[child] == 0) {
                        colors[child] = 1;
                        try stack.append(allocator, .{ .node = child, .next_child = 0 });
                    }
                } else {
                    colors[frame.node] = 2;
                    _ = stack.pop();
                }
            }
        }
        break :blk false;
    };

    var max_payload: u32 = 0;
    var total_payload: u64 = 0;
    for (result.states.items) |state| {
        max_payload = @max(max_payload, @as(u32, @intCast(state.payload.len)));
        total_payload += state.payload.len;
    }

    const depth_len = result.states_per_depth.items.len;
    var growth: std.ArrayListUnmanaged(f64) = .empty;
    errdefer growth.deinit(allocator);
    var d: usize = 1;
    while (d < depth_len) : (d += 1) {
        const prev = result.states_per_depth.items[d - 1];
        if (prev == 0) break;
        try growth.append(allocator, @as(f64, @floatFromInt(result.states_per_depth.items[d])) / @as(f64, @floatFromInt(prev)));
    }

    return .{
        .allocator = allocator,
        .unique_states = @intCast(n_states),
        .event_count = @intCast(result.events.items.len),
        .unique_transitions = transitions.count(),
        .states_per_depth = try allocator.dupe(u32, result.states_per_depth.items),
        .events_per_depth = try allocator.dupe(u32, result.events_per_depth.items),
        // Breadth-first traversal expands exactly the states first discovered
        // at each depth, so frontier width per depth equals discoveries.
        .frontier_width_per_depth = try allocator.dupe(u32, result.states_per_depth.items),
        .mean_out_degree = if (n_states == 0) 0.0 else @as(f64, @floatFromInt(transitions.count())) / @as(f64, @floatFromInt(n_states)),
        .max_out_degree = max_out,
        .median_out_degree = median,
        .convergent_states = convergent,
        .self_loops = self_loops,
        .has_cycle = has_cycle,
        .weakly_connected_components = components,
        .max_payload_bytes = max_payload,
        .mean_payload_bytes = if (n_states == 0) 0.0 else @as(f64, @floatFromInt(total_payload)) / @as(f64, @floatFromInt(n_states)),
        .growth_rates = try growth.toOwnedSlice(allocator),
        .termination = result.termination,
        .exhaustive = result.complete,
    };
}

test {
    std.testing.refAllDecls(@This());
}
