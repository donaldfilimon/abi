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

/// Causal edge between events: `parent` produced tokens later consumed by `child`.
pub const CausalEdge = struct {
    parent: u32,
    child: u32,
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

pub fn hashPayload(payload: []const u8) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload, &digest, .{});
    return digest;
}

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

test {
    std.testing.refAllDecls(@This());
}
