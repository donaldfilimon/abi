//! Bounded multiway (Wolfram-style) string-rewriting simulator.
//!
//! Simulates finite, explicitly bounded slices of computational rule space:
//! exact string rewriting with overlapping matches, breadth-first multiway
//! evolution, canonical state deduplication by content hash, and full event
//! multiplicity. This module does NOT simulate "the complete ruliad" and makes
//! no claims about fundamental physics; structural resemblance of a rule's
//! output to physical behavior is not evidence that the rule describes the
//! physical universe. See docs/spec/wdbx-multiway.mdx for scope boundaries.
//!
//! Determinism: with the same initial states, rules, configuration, and seed,
//! `run`/`resume` produce byte-for-byte identical canonical exports. The
//! canonical export deliberately excludes wall-clock timing; runtime metrics
//! live only in the human summary.
//!
//! Causal note: without symbol-level lineage tracking, causal edges between
//! events cannot be derived rigorously for string rewriting; this module
//! exports state-transition and event graphs only and marks causal analysis
//! as limited.

const std = @import("std");
const builtin = @import("builtin");
const json_util = @import("../../foundation/json.zig");
const foundation_time = @import("../../foundation/time.zig");
const persistence = @import("persistence.zig");
const recovery = @import("recovery.zig");

pub const FORMAT_VERSION = "abi-multiway-v1";

/// Hard ceiling on rules per experiment; keeps candidate scans bounded.
pub const MAX_RULES = 256;

// ---------------------------------------------------------------------------
// Core data model
// ---------------------------------------------------------------------------

/// An exact string-rewriting rule `lhs -> rhs`. The rule identifier is its
/// index in `Config.rules` (stable because the rule list is part of the
/// hashed configuration). `canonicalText` and `contentHash` provide the
/// stable serialized form and content address.
pub const Rule = struct {
    lhs: []const u8,
    rhs: []const u8,
    weight: f64 = 1.0,
    family: ?[]const u8 = null,

    pub fn canonicalText(self: Rule, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{s}->{s}", .{ self.lhs, self.rhs });
    }

    pub fn contentHash(self: Rule) [32]u8 {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(self.lhs);
        hasher.update("->");
        hasher.update(self.rhs);
        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        return digest;
    }
};

pub const Traversal = enum { breadth_first };
pub const Canonicalization = enum { exact_string };
pub const DedupPolicy = enum { by_canonical_hash };

/// Complete, serializable, hashable experiment configuration. All bounds are
/// hard limits: the engine terminates (with a recorded reason and valid
/// partial results) rather than exceed any of them.
pub const Config = struct {
    initial: []const []const u8,
    rules: []const Rule,
    max_depth: u32 = 5,
    max_states: u32 = 10_000,
    max_events: u32 = 100_000,
    /// Maximum state payload length in bytes, enforced BEFORE allocating any
    /// replacement result.
    max_payload: u32 = 4096,
    /// Wall-clock budget in milliseconds; 0 disables the deadline.
    max_duration_ms: u64 = 0,
    /// Approximate engine memory budget in bytes (payload bytes + fixed
    /// per-state/per-event overhead); 0 disables the budget.
    max_memory_bytes: u64 = 0,
    traversal: Traversal = .breadth_first,
    canonicalization: Canonicalization = .exact_string,
    dedup: DedupPolicy = .by_canonical_hash,
    /// Recorded for reproducibility; the exact-string BFS engine is fully
    /// deterministic and consumes no randomness yet.
    seed: u64 = 0,
    /// Recorded for reproducibility. The engine currently always expands
    /// single-threaded so exported identifiers can never depend on scheduler
    /// order; values > 1 are accepted and recorded but do not spawn threads.
    workers: u32 = 1,
};

pub const ConfigError = error{
    NoInitialStates,
    NoRules,
    TooManyRules,
    EmptyLhs,
    InitialPayloadTooLarge,
    ZeroBound,
};

/// Validate a configuration; returns an actionable error for CLI diagnostics.
pub fn validateConfig(config: Config) ConfigError!void {
    if (config.initial.len == 0) return ConfigError.NoInitialStates;
    if (config.rules.len == 0) return ConfigError.NoRules;
    if (config.rules.len > MAX_RULES) return ConfigError.TooManyRules;
    for (config.rules) |rule| {
        // Empty LHS would match everywhere unboundedly; rejected until an
        // explicitly bounded insertion mode exists.
        if (rule.lhs.len == 0) return ConfigError.EmptyLhs;
    }
    if (config.max_depth == 0 or config.max_states == 0 or config.max_events == 0 or config.max_payload == 0) {
        return ConfigError.ZeroBound;
    }
    for (config.initial) |payload| {
        if (payload.len > config.max_payload) return ConfigError.InitialPayloadTooLarge;
    }
}

/// Why evolution stopped. Only `frontier_exhausted` marks a complete
/// exploration of the bounded domain.
pub const Termination = enum {
    frontier_exhausted,
    max_depth,
    max_states,
    max_events,
    payload_limit,
    deadline,
    cancelled,
    allocation_failure,
    invalid_rule,
    invariant_failure,

    pub fn label(self: Termination) []const u8 {
        return @tagName(self);
    }
};

/// A canonical state. Identity is the SHA-256 of the canonical payload —
/// never the memory address, discovery order, or branch path. `seq` is the
/// deterministic creation sequence number (also the index into the state
/// list). In breadth-first traversal the discovery depth IS the minimum
/// known depth.
pub const State = struct {
    payload: []u8,
    hash: [32]u8,
    depth: u32,
    seq: u32,
    /// Event id that first produced this state; null for initial states.
    first_event: ?u32,
};

/// One rule application. Multiple events may connect the same source and
/// destination when they represent genuinely different applications
/// (different rule and/or match offset); state dedup never erases them.
pub const Event = struct {
    /// Stable event identifier == deterministic commit index.
    id: u32,
    src: u32,
    dst: u32,
    rule: u32,
    pos: u32,
    depth: u32,
    /// Deterministic local index within the source state's expansion.
    local: u32,
};

// ---------------------------------------------------------------------------
// Result and engine
// ---------------------------------------------------------------------------

pub const Result = struct {
    allocator: std.mem.Allocator,
    states: std.ArrayListUnmanaged(State) = .empty,
    events: std.ArrayListUnmanaged(Event) = .empty,
    states_per_depth: std.ArrayListUnmanaged(u32) = .empty,
    events_per_depth: std.ArrayListUnmanaged(u32) = .empty,
    termination: Termination = .invariant_failure,
    /// True only when the frontier emptied with no resource bound tripped.
    complete: bool = false,
    /// Resume cursor: depth currently being expanded, remaining frontier
    /// (state indices at `resume_depth`), accumulated next frontier, and the
    /// index into `frontier` of the next unexpanded state.
    resume_depth: u32 = 0,
    frontier: std.ArrayListUnmanaged(u32) = .empty,
    next_frontier: std.ArrayListUnmanaged(u32) = .empty,
    cursor: u32 = 0,
    /// Wall-clock spent inside the engine; excluded from canonical exports.
    elapsed_ns: u64 = 0,

    pub fn deinit(self: *Result) void {
        for (self.states.items) |state| self.allocator.free(state.payload);
        self.states.deinit(self.allocator);
        self.events.deinit(self.allocator);
        self.states_per_depth.deinit(self.allocator);
        self.events_per_depth.deinit(self.allocator);
        self.frontier.deinit(self.allocator);
        self.next_frontier.deinit(self.allocator);
    }

    pub fn findState(self: *const Result, hash: [32]u8) ?u32 {
        for (self.states.items) |state| {
            if (std.mem.eql(u8, &state.hash, &hash)) return state.seq;
        }
        return null;
    }
};

fn hashPayload(payload: []const u8) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
    return digest;
}

const Candidate = struct {
    rule: u32,
    pos: u32,
    payload: []u8,
    hash: [32]u8,
};

const STATE_OVERHEAD_BYTES: u64 = @sizeOf(State) + 48; // hash-map slot estimate
const EVENT_OVERHEAD_BYTES: u64 = @sizeOf(Event);

const Engine = struct {
    allocator: std.mem.Allocator,
    config: Config,
    result: Result,
    index_by_hash: std.AutoHashMapUnmanaged([32]u8, u32) = .empty,
    approx_bytes: u64 = 0,
    cancel: ?*const std.atomic.Value(bool),
    deadline_ns: ?i64,

    fn deinitIndex(self: *Engine) void {
        self.index_by_hash.deinit(self.allocator);
    }

    fn bumpDepthCounters(self: *Engine, depth: u32) !void {
        while (self.result.states_per_depth.items.len <= depth) {
            try self.result.states_per_depth.append(self.allocator, 0);
        }
        while (self.result.events_per_depth.items.len <= depth) {
            try self.result.events_per_depth.append(self.allocator, 0);
        }
    }

    /// Commit initial states (deduplicated) at depth 0.
    fn seed(self: *Engine) !?Termination {
        try self.bumpDepthCounters(0);
        for (self.config.initial) |payload| {
            if (payload.len > self.config.max_payload) return .payload_limit;
            const digest = hashPayload(payload);
            const gop = try self.index_by_hash.getOrPut(self.allocator, digest);
            if (gop.found_existing) continue;
            if (self.result.states.items.len >= self.config.max_states) {
                self.index_by_hash.removeByPtr(gop.key_ptr);
                return .max_states;
            }
            const owned = try self.allocator.dupe(u8, payload);
            errdefer self.allocator.free(owned);
            const seq: u32 = @intCast(self.result.states.items.len);
            gop.value_ptr.* = seq;
            try self.result.states.append(self.allocator, .{
                .payload = owned,
                .hash = digest,
                .depth = 0,
                .seq = seq,
                .first_event = null,
            });
            try self.result.frontier.append(self.allocator, seq);
            self.result.states_per_depth.items[0] += 1;
            self.approx_bytes += owned.len + STATE_OVERHEAD_BYTES;
        }
        return null;
    }

    /// Generate every valid rule application for one source state in
    /// deterministic order (rule index, then match offset, overlapping
    /// matches included). Payload bound is enforced before allocating any
    /// replacement buffer. Caller owns the returned candidates.
    fn generateCandidates(self: *Engine, src_payload: []const u8, out: *std.ArrayListUnmanaged(Candidate)) !?Termination {
        for (self.config.rules, 0..) |rule, rule_idx| {
            var start: usize = 0;
            while (std.mem.indexOfPos(u8, src_payload, start, rule.lhs)) |pos| {
                const dst_len = src_payload.len - rule.lhs.len + rule.rhs.len;
                if (dst_len > self.config.max_payload) return .payload_limit;
                const dst = try self.allocator.alloc(u8, dst_len);
                @memcpy(dst[0..pos], src_payload[0..pos]);
                @memcpy(dst[pos .. pos + rule.rhs.len], rule.rhs);
                @memcpy(dst[pos + rule.rhs.len ..], src_payload[pos + rule.lhs.len ..]);
                try out.append(self.allocator, .{
                    .rule = @intCast(rule_idx),
                    .pos = @intCast(pos),
                    .payload = dst,
                    .hash = hashPayload(dst),
                });
                start = pos + 1; // overlapping matches
            }
        }
        return null;
    }

    fn freeCandidates(self: *Engine, candidates: *std.ArrayListUnmanaged(Candidate)) void {
        for (candidates.items) |cand| self.allocator.free(cand.payload);
        candidates.deinit(self.allocator);
    }

    /// Expand one source state. All of the state's events commit atomically
    /// or none do, so hard caps are never exceeded and the resume cursor can
    /// stay state-granular. Returns a termination reason when a bound trips.
    fn expandOne(self: *Engine, src_idx: u32, child_depth: u32) !?Termination {
        var candidates: std.ArrayListUnmanaged(Candidate) = .empty;
        defer self.freeCandidates(&candidates);

        const src_payload = self.result.states.items[src_idx].payload;
        if (try self.generateCandidates(src_payload, &candidates)) |term| return term;

        // Pre-commit cap checks over the whole batch.
        if (self.result.events.items.len + candidates.items.len > self.config.max_events) {
            return .max_events;
        }
        var new_unique: usize = 0;
        var payload_bytes: u64 = 0;
        for (candidates.items, 0..) |cand, i| {
            if (self.index_by_hash.contains(cand.hash)) continue;
            var dup_in_batch = false;
            for (candidates.items[0..i]) |prev| {
                if (std.mem.eql(u8, &prev.hash, &cand.hash)) {
                    dup_in_batch = true;
                    break;
                }
            }
            if (!dup_in_batch) {
                new_unique += 1;
                payload_bytes += cand.payload.len;
            }
        }
        if (self.result.states.items.len + new_unique > self.config.max_states) {
            return .max_states;
        }
        if (self.config.max_memory_bytes != 0) {
            const projected = self.approx_bytes +
                payload_bytes +
                @as(u64, @intCast(new_unique)) * STATE_OVERHEAD_BYTES +
                @as(u64, @intCast(candidates.items.len)) * EVENT_OVERHEAD_BYTES;
            if (projected > self.config.max_memory_bytes) return .allocation_failure;
        }

        // Commit phase.
        try self.bumpDepthCounters(child_depth);
        for (candidates.items, 0..) |cand, local| {
            const event_id: u32 = @intCast(self.result.events.items.len);
            const gop = try self.index_by_hash.getOrPut(self.allocator, cand.hash);
            var dst_idx: u32 = undefined;
            if (gop.found_existing) {
                dst_idx = gop.value_ptr.*;
            } else {
                const owned = try self.allocator.dupe(u8, cand.payload);
                errdefer self.allocator.free(owned);
                dst_idx = @intCast(self.result.states.items.len);
                gop.value_ptr.* = dst_idx;
                try self.result.states.append(self.allocator, .{
                    .payload = owned,
                    .hash = cand.hash,
                    .depth = child_depth,
                    .seq = dst_idx,
                    .first_event = event_id,
                });
                try self.result.next_frontier.append(self.allocator, dst_idx);
                self.result.states_per_depth.items[child_depth] += 1;
                self.approx_bytes += owned.len + STATE_OVERHEAD_BYTES;
            }
            try self.result.events.append(self.allocator, .{
                .id = event_id,
                .src = src_idx,
                .dst = dst_idx,
                .rule = cand.rule,
                .pos = cand.pos,
                .depth = child_depth,
                .local = @intCast(local),
            });
            self.result.events_per_depth.items[child_depth] += 1;
            self.approx_bytes += EVENT_OVERHEAD_BYTES;
        }
        return null;
    }

    fn evolve(self: *Engine) !void {
        const start_ns = foundation_time.monotonicNs();
        defer self.result.elapsed_ns += @intCast(@max(foundation_time.monotonicNs() - start_ns, 0));

        outer: while (true) {
            if (self.result.frontier.items.len == 0) {
                self.result.termination = .frontier_exhausted;
                self.result.complete = true;
                return;
            }
            if (self.result.resume_depth >= self.config.max_depth) {
                self.result.termination = .max_depth;
                return;
            }
            const child_depth = self.result.resume_depth + 1;
            while (self.result.cursor < self.result.frontier.items.len) {
                if (self.cancel) |flag| {
                    if (flag.load(.acquire)) {
                        self.result.termination = .cancelled;
                        return;
                    }
                }
                if (self.deadline_ns) |deadline| {
                    if (foundation_time.monotonicNs() >= deadline) {
                        self.result.termination = .deadline;
                        return;
                    }
                }
                const src_idx = self.result.frontier.items[self.result.cursor];
                const term = self.expandOne(src_idx, child_depth) catch |err| switch (err) {
                    error.OutOfMemory => {
                        self.result.termination = .allocation_failure;
                        return;
                    },
                };
                if (term) |reason| {
                    self.result.termination = reason;
                    return;
                }
                self.result.cursor += 1;
            }
            // Depth fully expanded: rotate frontiers.
            self.result.frontier.clearRetainingCapacity();
            try self.result.frontier.appendSlice(self.allocator, self.result.next_frontier.items);
            self.result.next_frontier.clearRetainingCapacity();
            self.result.cursor = 0;
            self.result.resume_depth += 1;
            continue :outer;
        }
    }
};

/// Run a bounded multiway evolution from `config`. Always returns a valid
/// (possibly partial) `Result` with a recorded termination reason; hard caps
/// (`max_states`, `max_events`, `max_depth`, `max_payload`) are never
/// exceeded. `cancel`, when supplied, is polled once per state expansion.
pub fn run(allocator: std.mem.Allocator, config: Config, cancel: ?*const std.atomic.Value(bool)) !Result {
    var engine = Engine{
        .allocator = allocator,
        .config = config,
        .result = .{ .allocator = allocator },
        .cancel = cancel,
        .deadline_ns = if (config.max_duration_ms == 0) null else foundation_time.monotonicNs() + @as(i64, @intCast(config.max_duration_ms * std.time.ns_per_ms)),
    };
    defer engine.deinitIndex();
    errdefer engine.result.deinit();

    validateConfig(config) catch {
        engine.result.termination = .invalid_rule;
        return engine.result;
    };
    if (try engine.seed()) |term| {
        engine.result.termination = term;
        return engine.result;
    }
    try engine.evolve();
    return engine.result;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Canonical JSON export / import (resume)
// ---------------------------------------------------------------------------

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
    // Causal analysis is limited for exact string rewriting: no symbol-level
    // lineage is tracked, so no causal edges are exported.
    try out.appendSlice(allocator, ",\"causal_graph\":\"limited:no-token-lineage\"");
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

pub const ResumeError = error{
    UnsupportedFormat,
    MalformedExport,
    AlreadyComplete,
    ConfigMismatch,
    OutOfMemory,
};

/// Resume a bounded experiment from a canonical export produced by
/// `exportCanonicalJson`. `config` supplies the new effective limits and must
/// contain the same initial states and rules (same canonical text and order)
/// as the original run; only the bounds may differ. A resumed deterministic
/// run commits identifiers in exactly the order an uninterrupted run under
/// the same effective limits would, so the final exports match byte-for-byte.
pub fn resume_(allocator: std.mem.Allocator, export_json: []const u8, config: Config, cancel: ?*const std.atomic.Value(bool)) !Result {
    validateConfig(config) catch return ResumeError.ConfigMismatch;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, export_json, .{}) catch |err| switch (err) {
        error.OutOfMemory => return ResumeError.OutOfMemory,
        else => return ResumeError.MalformedExport,
    };
    defer parsed.deinit();
    const root = objectOrNull(parsed.value) orelse return ResumeError.MalformedExport;

    const format = stringField(root, "format") orelse return ResumeError.MalformedExport;
    if (!std.mem.eql(u8, format, FORMAT_VERSION)) return ResumeError.UnsupportedFormat;

    // Rule/initial compatibility: the persisted config must match the new one
    // on identity fields (canonical rules + initial states).
    const cfg_obj = objectOrNull(root.get("config") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
    {
        const initial = arrayOrNull(cfg_obj.get("initial") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
        if (initial.items.len != config.initial.len) return ResumeError.ConfigMismatch;
        for (initial.items, 0..) |item, i| {
            const text = stringOrNull(item) orelse return ResumeError.MalformedExport;
            if (!std.mem.eql(u8, text, config.initial[i])) return ResumeError.ConfigMismatch;
        }
        const rules = arrayOrNull(cfg_obj.get("rules") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
        if (rules.items.len != config.rules.len) return ResumeError.ConfigMismatch;
        for (rules.items, 0..) |item, i| {
            const rule_obj = objectOrNull(item) orelse return ResumeError.MalformedExport;
            const text = stringField(rule_obj, "rule") orelse return ResumeError.MalformedExport;
            const expected = try config.rules[i].canonicalText(allocator);
            defer allocator.free(expected);
            if (!std.mem.eql(u8, text, expected)) return ResumeError.ConfigMismatch;
        }
    }

    const resume_value = root.get("resume") orelse return ResumeError.MalformedExport;
    if (resume_value == .null) return ResumeError.AlreadyComplete;
    const resume_obj = objectOrNull(resume_value) orelse return ResumeError.MalformedExport;

    var engine = Engine{
        .allocator = allocator,
        .config = config,
        .result = .{ .allocator = allocator },
        .cancel = cancel,
        .deadline_ns = if (config.max_duration_ms == 0) null else foundation_time.monotonicNs() + @as(i64, @intCast(config.max_duration_ms * std.time.ns_per_ms)),
    };
    defer engine.deinitIndex();
    errdefer engine.result.deinit();

    // Rebuild states.
    const states = arrayOrNull(root.get("states") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
    for (states.items, 0..) |item, i| {
        const state_obj = objectOrNull(item) orelse return ResumeError.MalformedExport;
        const payload = stringField(state_obj, "payload") orelse return ResumeError.MalformedExport;
        const depth = intField(state_obj, "depth") orelse return ResumeError.MalformedExport;
        const first_event: ?u32 = if (intField(state_obj, "first_event")) |v| @intCast(v) else null;
        const owned = try allocator.dupe(u8, payload);
        errdefer allocator.free(owned);
        const digest = hashPayload(owned);
        const seq: u32 = @intCast(i);
        try engine.index_by_hash.put(allocator, digest, seq);
        try engine.result.states.append(allocator, .{
            .payload = owned,
            .hash = digest,
            .depth = @intCast(depth),
            .seq = seq,
            .first_event = first_event,
        });
        engine.approx_bytes += owned.len + STATE_OVERHEAD_BYTES;
    }

    // Rebuild events.
    const events = arrayOrNull(root.get("events") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
    for (events.items, 0..) |item, i| {
        const event_obj = objectOrNull(item) orelse return ResumeError.MalformedExport;
        try engine.result.events.append(allocator, .{
            .id = @intCast(i),
            .src = @intCast(intField(event_obj, "src") orelse return ResumeError.MalformedExport),
            .dst = @intCast(intField(event_obj, "dst") orelse return ResumeError.MalformedExport),
            .rule = @intCast(intField(event_obj, "rule") orelse return ResumeError.MalformedExport),
            .pos = @intCast(intField(event_obj, "pos") orelse return ResumeError.MalformedExport),
            .depth = @intCast(intField(event_obj, "depth") orelse return ResumeError.MalformedExport),
            .local = @intCast(intField(event_obj, "local") orelse return ResumeError.MalformedExport),
        });
        engine.approx_bytes += EVENT_OVERHEAD_BYTES;
    }

    // Rebuild per-depth counters from the states/events themselves.
    for (engine.result.states.items) |state| {
        try engine.bumpDepthCounters(state.depth);
        engine.result.states_per_depth.items[state.depth] += 1;
    }
    for (engine.result.events.items) |event| {
        try engine.bumpDepthCounters(event.depth);
        engine.result.events_per_depth.items[event.depth] += 1;
    }

    // Rebuild the resume cursor.
    engine.result.resume_depth = @intCast(intField(resume_obj, "depth") orelse return ResumeError.MalformedExport);
    engine.result.cursor = @intCast(intField(resume_obj, "cursor") orelse return ResumeError.MalformedExport);
    const frontier = arrayOrNull(resume_obj.get("frontier") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
    for (frontier.items) |item| {
        const idx = intOrNull(item) orelse return ResumeError.MalformedExport;
        if (idx >= engine.result.states.items.len) return ResumeError.MalformedExport;
        try engine.result.frontier.append(allocator, @intCast(idx));
    }
    const next_frontier = arrayOrNull(resume_obj.get("next_frontier") orelse return ResumeError.MalformedExport) orelse return ResumeError.MalformedExport;
    for (next_frontier.items) |item| {
        const idx = intOrNull(item) orelse return ResumeError.MalformedExport;
        if (idx >= engine.result.states.items.len) return ResumeError.MalformedExport;
        try engine.result.next_frontier.append(allocator, @intCast(idx));
    }
    if (engine.result.cursor > engine.result.frontier.items.len) return ResumeError.MalformedExport;

    try engine.evolve();
    return engine.result;
}

fn objectOrNull(value: std.json.Value) ?std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => null,
    };
}

fn arrayOrNull(value: std.json.Value) ?std.json.Array {
    return switch (value) {
        .array => |arr| arr,
        else => null,
    };
}

fn stringOrNull(value: std.json.Value) ?[]const u8 {
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

fn intOrNull(value: std.json.Value) ?u64 {
    return switch (value) {
        .integer => |n| if (n < 0) null else @intCast(n),
        else => null,
    };
}

fn stringField(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    return stringOrNull(obj.get(key) orelse return null);
}

fn intField(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    return intOrNull(obj.get(key) orelse return null);
}

// ---------------------------------------------------------------------------
// DOT export
// ---------------------------------------------------------------------------

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
        try out.appendSlice(allocator, "\\xe2\\x86\\x92");
        try appendDotEscaped(&out, allocator, rule.rhs);
        try appendFmt(&out, allocator, "@{d}\"];\n", .{event.pos});
    }
    try out.appendSlice(allocator, "}\n");
    return out.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// WDBX persistence
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rule parsing (shared by CLI and config files)
// ---------------------------------------------------------------------------

pub const ParseRuleError = error{ MissingArrow, EmptyLhs, OutOfMemory };

/// Parse `LHS->RHS` (whitespace around either side is trimmed; the RHS may be
/// empty for deletion rules). Both sides are duplicated into `allocator`.
pub fn parseRule(allocator: std.mem.Allocator, text: []const u8) ParseRuleError!Rule {
    const arrow = std.mem.indexOf(u8, text, "->") orelse return ParseRuleError.MissingArrow;
    const lhs = std.mem.trim(u8, text[0..arrow], " \t");
    const rhs = std.mem.trim(u8, text[arrow + 2 ..], " \t");
    if (lhs.len == 0) return ParseRuleError.EmptyLhs;
    const owned_lhs = try allocator.dupe(u8, lhs);
    errdefer allocator.free(owned_lhs);
    const owned_rhs = try allocator.dupe(u8, rhs);
    return .{ .lhs = owned_lhs, .rhs = owned_rhs };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing_alloc = std.testing.allocator;

fn testRules(comptime specs: []const []const u8) [specs.len]Rule {
    var rules: [specs.len]Rule = undefined;
    inline for (specs, 0..) |spec, i| {
        const arrow = comptime std.mem.indexOf(u8, spec, "->").?;
        rules[i] = .{ .lhs = spec[0..arrow], .rhs = spec[arrow + 2 ..] };
    }
    return rules;
}

test "rule parsing accepts trimmed forms and rejects malformed input" {
    const rule = try parseRule(testing_alloc, " A -> AB ");
    defer {
        testing_alloc.free(rule.lhs);
        testing_alloc.free(rule.rhs);
    }
    try std.testing.expectEqualStrings("A", rule.lhs);
    try std.testing.expectEqualStrings("AB", rule.rhs);

    const deletion = try parseRule(testing_alloc, "BB->");
    defer {
        testing_alloc.free(deletion.lhs);
        testing_alloc.free(deletion.rhs);
    }
    try std.testing.expectEqualStrings("", deletion.rhs);

    try std.testing.expectError(ParseRuleError.MissingArrow, parseRule(testing_alloc, "AB"));
    try std.testing.expectError(ParseRuleError.EmptyLhs, parseRule(testing_alloc, "->B"));
    try std.testing.expectError(ParseRuleError.EmptyLhs, parseRule(testing_alloc, "  ->B"));
}

test "rule content hash is stable and canonical" {
    const a = Rule{ .lhs = "A", .rhs = "AB" };
    const b = Rule{ .lhs = "A", .rhs = "AB" };
    const c = Rule{ .lhs = "AA", .rhs = "B" };
    try std.testing.expect(std.mem.eql(u8, &a.contentHash(), &b.contentHash()));
    try std.testing.expect(!std.mem.eql(u8, &a.contentHash(), &c.contentHash()));
    const canonical = try a.canonicalText(testing_alloc);
    defer testing_alloc.free(canonical);
    try std.testing.expectEqualStrings("A->AB", canonical);
}

test "state hashing is content-based" {
    const h1 = hashPayload("ABA");
    const h2 = hashPayload("ABA");
    const h3 = hashPayload("AAB");
    try std.testing.expect(std.mem.eql(u8, &h1, &h2));
    try std.testing.expect(!std.mem.eql(u8, &h1, &h3));
}

test "overlapping matches produce one event per occurrence" {
    // "AAA" with rule AA->B has two overlapping matches (pos 0 and 1).
    const rules = testRules(&.{"AA->B"});
    var result = try run(testing_alloc, .{
        .initial = &.{"AAA"},
        .rules = &rules,
        .max_depth = 1,
        .max_states = 100,
        .max_events = 100,
        .max_payload = 16,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.events.items.len);
    try std.testing.expectEqual(@as(u32, 0), result.events.items[0].pos);
    try std.testing.expectEqual(@as(u32, 1), result.events.items[1].pos);
    // BA and AB are distinct canonical states.
    try std.testing.expectEqual(@as(usize, 3), result.states.items.len);
    try std.testing.expectEqualStrings("BA", result.states.items[1].payload);
    try std.testing.expectEqualStrings("AB", result.states.items[2].payload);
}

test "replacement application preserves prefix and suffix" {
    const rules = testRules(&.{"BC->X"});
    var result = try run(testing_alloc, .{
        .initial = &.{"ABCD"},
        .rules = &rules,
        .max_depth = 1,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 16,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.states.items.len);
    try std.testing.expectEqualStrings("AXD", result.states.items[1].payload);
}

test "event multiplicity: duplicate rules and convergent applications are distinct events" {
    // Two identical rules: state dedup must not erase the two distinct events.
    const rules = testRules(&.{ "A->B", "A->B" });
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 1,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.states.items.len);
    try std.testing.expectEqual(@as(usize, 2), result.events.items.len);
    try std.testing.expectEqual(@as(u32, 0), result.events.items[0].rule);
    try std.testing.expectEqual(@as(u32, 1), result.events.items[1].rule);
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    try std.testing.expectEqual(@as(u32, 2), metrics.event_count);
    try std.testing.expectEqual(@as(u32, 1), metrics.unique_transitions);
}

test "self-loop rule is recorded and detected" {
    const rules = testRules(&.{"A->A"});
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 3,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    // A -> A rewrites to the same canonical state: 1 state, self-loop events,
    // and the frontier drains (no NEW states) after depth 1.
    try std.testing.expectEqual(@as(usize, 1), result.states.items.len);
    try std.testing.expect(result.events.items.len >= 1);
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    try std.testing.expectEqual(@as(u32, 1), metrics.self_loops);
    try std.testing.expect(metrics.has_cycle);
    try std.testing.expectEqual(Termination.frontier_exhausted, result.termination);
    try std.testing.expect(result.complete);
}

test "two-cycle system detects a cycle and terminates by dedup" {
    const rules = testRules(&.{ "A->B", "B->A" });
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 10,
        .max_states = 10,
        .max_events = 100,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.states.items.len);
    try std.testing.expect(result.complete);
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    try std.testing.expect(metrics.has_cycle);
    try std.testing.expectEqual(@as(u32, 1), metrics.weakly_connected_components);
}

test "shrinking rules can reach the empty state" {
    const rules = testRules(&.{"A->"});
    var result = try run(testing_alloc, .{
        .initial = &.{"AA"},
        .rules = &rules,
        .max_depth = 5,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    try std.testing.expect(result.complete);
    // AA -> A -> "" (empty payload is a valid canonical state).
    try std.testing.expectEqual(@as(usize, 3), result.states.items.len);
    try std.testing.expectEqualStrings("", result.states.items[2].payload);
}

test "convergent branches deduplicate the state but keep both transitions" {
    // AB and BA both rewrite to CC via two different rules.
    const rules = testRules(&.{ "A->C", "B->C" });
    var result = try run(testing_alloc, .{
        .initial = &.{"AB"},
        .rules = &rules,
        .max_depth = 3,
        .max_states = 20,
        .max_events = 20,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    // States: AB, CB, AC, CC — CC discovered twice, stored once.
    try std.testing.expectEqual(@as(u32, 4), metrics.unique_states);
    try std.testing.expectEqual(@as(u32, 1), metrics.convergent_states);
    try std.testing.expect(result.complete);
}

test "max depth bound terminates with unexpanded frontier" {
    const rules = testRules(&.{"A->AA"});
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 2,
        .max_states = 100,
        .max_events = 100,
        .max_payload = 64,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.max_depth, result.termination);
    try std.testing.expect(!result.complete);
    try std.testing.expect(result.frontier.items.len > 0);
    for (result.states.items) |state| try std.testing.expect(state.depth <= 2);
}

test "max states bound is never exceeded" {
    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 10,
        .max_states = 7,
        .max_events = 10_000,
        .max_payload = 64,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.max_states, result.termination);
    try std.testing.expect(!result.complete);
    try std.testing.expect(result.states.items.len <= 7);
}

test "max events bound is never exceeded" {
    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 10,
        .max_states = 10_000,
        .max_events = 9,
        .max_payload = 64,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.max_events, result.termination);
    try std.testing.expect(!result.complete);
    try std.testing.expect(result.events.items.len <= 9);
}

test "payload bound terminates before allocating an oversized replacement" {
    const rules = testRules(&.{"A->AAAA"});
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 10,
        .max_states = 100,
        .max_events = 100,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.payload_limit, result.termination);
    try std.testing.expect(!result.complete);
    for (result.states.items) |state| try std.testing.expect(state.payload.len <= 8);
}

test "cancellation flag terminates the run" {
    const rules = testRules(&.{"A->AB"});
    var flag = std.atomic.Value(bool).init(true);
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 10,
        .max_states = 100,
        .max_events = 100,
        .max_payload = 64,
    }, &flag);
    defer result.deinit();
    try std.testing.expectEqual(Termination.cancelled, result.termination);
    try std.testing.expect(!result.complete);
}

test "memory budget terminates with allocation_failure reason" {
    const rules = testRules(&.{"A->AA"});
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 20,
        .max_states = 100_000,
        .max_events = 100_000,
        .max_payload = 4096,
        .max_memory_bytes = 512,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.allocation_failure, result.termination);
    try std.testing.expect(!result.complete);
}

test "invalid rule reports invalid_rule termination" {
    const bad_rules = [_]Rule{.{ .lhs = "", .rhs = "X" }};
    var result = try run(testing_alloc, .{
        .initial = &.{"A"},
        .rules = &bad_rules,
        .max_depth = 2,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 8,
    }, null);
    defer result.deinit();
    try std.testing.expectEqual(Termination.invalid_rule, result.termination);
    try std.testing.expectEqual(@as(usize, 0), result.states.items.len);
}

test "config validation rejects impossible bounds" {
    const rules = testRules(&.{"A->B"});
    try std.testing.expectError(ConfigError.NoInitialStates, validateConfig(.{ .initial = &.{}, .rules = &rules }));
    try std.testing.expectError(ConfigError.NoRules, validateConfig(.{ .initial = &.{"A"}, .rules = &.{} }));
    try std.testing.expectError(ConfigError.ZeroBound, validateConfig(.{ .initial = &.{"A"}, .rules = &rules, .max_depth = 0 }));
    try std.testing.expectError(ConfigError.InitialPayloadTooLarge, validateConfig(.{ .initial = &.{"AAAA"}, .rules = &rules, .max_payload = 2 }));
}

test "deterministic export: repeated runs are byte-for-byte identical" {
    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    const config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 4,
        .max_states = 500,
        .max_events = 5000,
        .max_payload = 64,
    };
    var first: ?[]u8 = null;
    defer if (first) |bytes| testing_alloc.free(bytes);
    var round: usize = 0;
    while (round < 3) : (round += 1) {
        var result = try run(testing_alloc, config, null);
        defer result.deinit();
        var metrics = try computeMetrics(testing_alloc, &result);
        defer metrics.deinit();
        const export_json = try exportCanonicalJson(testing_alloc, config, &result, &metrics);
        if (first) |bytes| {
            defer testing_alloc.free(export_json);
            try std.testing.expectEqualStrings(bytes, export_json);
        } else {
            first = export_json;
        }
    }
}

test "resume equivalence: interrupted + resumed run matches uninterrupted run" {
    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    const full_config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 5,
        .max_states = 500,
        .max_events = 5000,
        .max_payload = 64,
    };

    // Uninterrupted run.
    var direct = try run(testing_alloc, full_config, null);
    defer direct.deinit();
    var direct_metrics = try computeMetrics(testing_alloc, &direct);
    defer direct_metrics.deinit();
    const direct_export = try exportCanonicalJson(testing_alloc, full_config, &direct, &direct_metrics);
    defer testing_alloc.free(direct_export);

    // Interrupted at depth 3, then resumed under the full limits.
    var shallow_config = full_config;
    shallow_config.max_depth = 3;
    var partial = try run(testing_alloc, shallow_config, null);
    defer partial.deinit();
    try std.testing.expectEqual(Termination.max_depth, partial.termination);
    var partial_metrics = try computeMetrics(testing_alloc, &partial);
    defer partial_metrics.deinit();
    const partial_export = try exportCanonicalJson(testing_alloc, shallow_config, &partial, &partial_metrics);
    defer testing_alloc.free(partial_export);

    var resumed = try resume_(testing_alloc, partial_export, full_config, null);
    defer resumed.deinit();
    var resumed_metrics = try computeMetrics(testing_alloc, &resumed);
    defer resumed_metrics.deinit();
    const resumed_export = try exportCanonicalJson(testing_alloc, full_config, &resumed, &resumed_metrics);
    defer testing_alloc.free(resumed_export);

    try std.testing.expectEqualStrings(direct_export, resumed_export);
}

test "resume equivalence holds for mid-depth event-cap interruption" {
    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    const full_config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 4,
        .max_states = 500,
        .max_events = 5000,
        .max_payload = 64,
    };
    var capped_config = full_config;
    capped_config.max_events = 11; // trips mid-depth

    var direct = try run(testing_alloc, full_config, null);
    defer direct.deinit();
    var direct_metrics = try computeMetrics(testing_alloc, &direct);
    defer direct_metrics.deinit();
    const direct_export = try exportCanonicalJson(testing_alloc, full_config, &direct, &direct_metrics);
    defer testing_alloc.free(direct_export);

    var partial = try run(testing_alloc, capped_config, null);
    defer partial.deinit();
    try std.testing.expectEqual(Termination.max_events, partial.termination);
    var partial_metrics = try computeMetrics(testing_alloc, &partial);
    defer partial_metrics.deinit();
    const partial_export = try exportCanonicalJson(testing_alloc, capped_config, &partial, &partial_metrics);
    defer testing_alloc.free(partial_export);

    var resumed = try resume_(testing_alloc, partial_export, full_config, null);
    defer resumed.deinit();
    var resumed_metrics = try computeMetrics(testing_alloc, &resumed);
    defer resumed_metrics.deinit();
    const resumed_export = try exportCanonicalJson(testing_alloc, full_config, &resumed, &resumed_metrics);
    defer testing_alloc.free(resumed_export);

    try std.testing.expectEqualStrings(direct_export, resumed_export);
}

test "resume rejects mismatched rules and completed exports" {
    const rules = testRules(&.{"A->AB"});
    const config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 2,
        .max_states = 50,
        .max_events = 50,
        .max_payload = 32,
    };
    var result = try run(testing_alloc, config, null);
    defer result.deinit();
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    const export_json = try exportCanonicalJson(testing_alloc, config, &result, &metrics);
    defer testing_alloc.free(export_json);

    const other_rules = testRules(&.{"A->BB"});
    var other_config = config;
    other_config.rules = &other_rules;
    try std.testing.expectError(ResumeError.ConfigMismatch, resume_(testing_alloc, export_json, other_config, null));
    try std.testing.expectError(ResumeError.MalformedExport, resume_(testing_alloc, "{\"format\":\"abi-multiway-v1\"}", config, null));
    try std.testing.expectError(ResumeError.UnsupportedFormat, resume_(testing_alloc, "{\"format\":\"nope\"}", config, null));

    // A frontier-exhausted export cannot be resumed.
    const cyc_rules = testRules(&.{ "A->B", "B->A" });
    const cyc_config = Config{
        .initial = &.{"A"},
        .rules = &cyc_rules,
        .max_depth = 10,
        .max_states = 10,
        .max_events = 100,
        .max_payload = 8,
    };
    var cyc = try run(testing_alloc, cyc_config, null);
    defer cyc.deinit();
    try std.testing.expect(cyc.complete);
    var cyc_metrics = try computeMetrics(testing_alloc, &cyc);
    defer cyc_metrics.deinit();
    const cyc_export = try exportCanonicalJson(testing_alloc, cyc_config, &cyc, &cyc_metrics);
    defer testing_alloc.free(cyc_export);
    try std.testing.expectError(ResumeError.AlreadyComplete, resume_(testing_alloc, cyc_export, cyc_config, null));
}

test "DOT export renders states and per-event edges" {
    const rules = testRules(&.{ "A->B", "A->B" });
    const config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 1,
        .max_states = 10,
        .max_events = 10,
        .max_payload = 8,
    };
    var result = try run(testing_alloc, config, null);
    defer result.deinit();
    const dot = try exportDot(testing_alloc, config, &result);
    defer testing_alloc.free(dot);
    try std.testing.expect(std.mem.indexOf(u8, dot, "digraph multiway") != null);
    try std.testing.expect(std.mem.indexOf(u8, dot, "s0 [label=\"A\\nd0\"]") != null);
    // Two events between the same pair — multiplicity visible in DOT.
    const first = std.mem.indexOf(u8, dot, "s0 -> s1").?;
    try std.testing.expect(std.mem.indexOfPos(u8, dot, first + 1, "s0 -> s1") != null);
}

test "config hash is stable and sensitive to rule changes" {
    const rules_a = testRules(&.{"A->AB"});
    const rules_b = testRules(&.{"A->BA"});
    const base = Config{ .initial = &.{"A"}, .rules = &rules_a };
    var changed = base;
    changed.rules = &rules_b;
    const h1 = try configHash(testing_alloc, base);
    const h2 = try configHash(testing_alloc, base);
    const h3 = try configHash(testing_alloc, changed);
    try std.testing.expect(std.mem.eql(u8, &h1, &h2));
    try std.testing.expect(!std.mem.eql(u8, &h1, &h3));
}

test "bounded random property sweep: caps hold and exports stay deterministic" {
    // Seeded generative sweep over small random rule systems. Verifies for
    // every system: caps never exceeded, depth monotonicity, dedup soundness
    // (unique hashes), and export determinism across a re-run.
    var prng = std.Random.DefaultPrng.init(0x2026_0ab1);
    const random = prng.random();
    const alphabet = "AB";
    var case_idx: usize = 0;
    while (case_idx < 24) : (case_idx += 1) {
        var rule_storage: [3]Rule = undefined;
        var text_storage: [3][8]u8 = undefined;
        const n_rules = random.intRangeAtMost(usize, 1, 3);
        for (0..n_rules) |r| {
            const lhs_len = random.intRangeAtMost(usize, 1, 2);
            const rhs_len = random.intRangeAtMost(usize, 0, 3);
            for (0..lhs_len) |i| text_storage[r][i] = alphabet[random.intRangeAtMost(usize, 0, 1)];
            for (0..rhs_len) |i| text_storage[r][4 + i] = alphabet[random.intRangeAtMost(usize, 0, 1)];
            rule_storage[r] = .{ .lhs = text_storage[r][0..lhs_len], .rhs = text_storage[r][4 .. 4 + rhs_len] };
        }
        const config = Config{
            .initial = &.{"AB"},
            .rules = rule_storage[0..n_rules],
            .max_depth = 4,
            .max_states = 64,
            .max_events = 256,
            .max_payload = 24,
        };
        var result = try run(testing_alloc, config, null);
        defer result.deinit();
        try std.testing.expect(result.states.items.len <= config.max_states);
        try std.testing.expect(result.events.items.len <= config.max_events);
        var seen: std.AutoHashMapUnmanaged([32]u8, void) = .empty;
        defer seen.deinit(testing_alloc);
        for (result.states.items) |state| {
            try std.testing.expect(state.payload.len <= config.max_payload);
            try std.testing.expect(state.depth <= config.max_depth);
            const gop = try seen.getOrPut(testing_alloc, state.hash);
            try std.testing.expect(!gop.found_existing);
        }
        var metrics = try computeMetrics(testing_alloc, &result);
        defer metrics.deinit();
        const export_a = try exportCanonicalJson(testing_alloc, config, &result, &metrics);
        defer testing_alloc.free(export_a);

        var result_b = try run(testing_alloc, config, null);
        defer result_b.deinit();
        var metrics_b = try computeMetrics(testing_alloc, &result_b);
        defer metrics_b.deinit();
        const export_b = try exportCanonicalJson(testing_alloc, config, &result_b, &metrics_b);
        defer testing_alloc.free(export_b);
        try std.testing.expectEqualStrings(export_a, export_b);
    }
}

test "wdbx persistence round-trip preserves the canonical export" {
    const test_helpers = @import("../../foundation/test_helpers.zig");
    const path = "zig-out/multiway-persist-test.jsonl";
    const cleanup = struct {
        fn cleanup(p: []const u8) void {
            var buf: [256]u8 = undefined;
            test_helpers.deleteTestFileIfExists(p);
            if (std.fmt.bufPrint(&buf, "{s}.wal", .{p})) |wp| test_helpers.deleteTestFileIfExists(wp) else |_| {}
            if (std.fmt.bufPrint(&buf, "{s}.manifest", .{p})) |mp| test_helpers.deleteTestFileIfExists(mp) else |_| {}
            var epoch: u64 = 0;
            while (epoch < 8) : (epoch += 1) {
                if (std.fmt.bufPrint(&buf, "{s}.seg.{d}.jsonl", .{ p, epoch })) |sp| test_helpers.deleteTestFileIfExists(sp) else |_| {}
            }
        }
    }.cleanup;
    cleanup(path);
    defer cleanup(path);

    const rules = testRules(&.{ "A->AB", "A->BA", "BB->A" });
    const config = Config{
        .initial = &.{"A"},
        .rules = &rules,
        .max_depth = 3,
        .max_states = 500,
        .max_events = 5000,
        .max_payload = 64,
    };
    var result = try run(testing_alloc, config, null);
    defer result.deinit();
    var metrics = try computeMetrics(testing_alloc, &result);
    defer metrics.deinit();
    const export_json = try exportCanonicalJson(testing_alloc, config, &result, &metrics);
    defer testing_alloc.free(export_json);

    try persistToWdbx(std.testing.io, testing_alloc, path, config, &result, export_json);

    // Round-trip by latest alias and by config hash.
    const loaded = try loadExportFromWdbx(std.testing.io, testing_alloc, path, null);
    defer testing_alloc.free(loaded);
    try std.testing.expectEqualStrings(export_json, loaded);

    const cfg_hash = try configHash(testing_alloc, config);
    const cfg_hex = std.fmt.bytesToHex(cfg_hash, .lower);
    const by_hash = try loadExportFromWdbx(std.testing.io, testing_alloc, path, &cfg_hex);
    defer testing_alloc.free(by_hash);
    try std.testing.expectEqualStrings(export_json, by_hash);

    // Content-addressed states are stored once; provenance block appended;
    // and a resumed run from the persisted export matches a direct run.
    var opened = try recovery.open(std.testing.io, testing_alloc, path);
    defer opened.store.deinit();
    try std.testing.expect(opened.store.blockCount() >= 1);
    const state_hex = std.fmt.bytesToHex(result.states.items[0].hash, .lower);
    const key = try std.fmt.allocPrint(testing_alloc, "multiway:state:{s}", .{state_hex});
    defer testing_alloc.free(key);
    try std.testing.expectEqualStrings("A", opened.store.get(key).?);

    var full_config = config;
    full_config.max_depth = 5;
    var resumed = try resume_(testing_alloc, loaded, full_config, null);
    defer resumed.deinit();
    var direct = try run(testing_alloc, full_config, null);
    defer direct.deinit();
    try std.testing.expectEqual(direct.states.items.len, resumed.states.items.len);
    try std.testing.expectEqual(direct.events.items.len, resumed.events.items.len);
}

test {
    std.testing.refAllDecls(@This());
}
