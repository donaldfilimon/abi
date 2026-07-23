//! Multiway evolution engine and resume path.
const std = @import("std");
const foundation_time = @import("../../foundation/time.zig");
const types = @import("multiway_types.zig");

const Config = types.Config;
const Result = types.Result;
const State = types.State;
const Event = types.Event;
const Termination = types.Termination;
const FORMAT_VERSION = types.FORMAT_VERSION;
const validateConfig = types.validateConfig;
const hashPayload = types.hashPayload;

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

test {
    std.testing.refAllDecls(@This());
}
