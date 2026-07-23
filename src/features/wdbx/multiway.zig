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
//! Causal note: token lineage is derived at export time from rule applications
//! (which output bytes a later match consumed). Hypergraph backends remain Proposed.

const std = @import("std");

const types = @import("multiway_types.zig");
const engine = @import("multiway_engine.zig");
const metrics_mod = @import("multiway_metrics.zig");
const export_mod = @import("multiway_export.zig");
const persist_mod = @import("multiway_persist.zig");
const recovery = @import("recovery.zig");

pub const FORMAT_VERSION = types.FORMAT_VERSION;
pub const MAX_RULES = types.MAX_RULES;
pub const Rule = types.Rule;
pub const Traversal = types.Traversal;
pub const Canonicalization = types.Canonicalization;
pub const DedupPolicy = types.DedupPolicy;
pub const Config = types.Config;
pub const ConfigError = types.ConfigError;
pub const validateConfig = types.validateConfig;
pub const Termination = types.Termination;
pub const State = types.State;
pub const Event = types.Event;
pub const CausalEdge = types.CausalEdge;
pub const Result = types.Result;
pub const hashPayload = types.hashPayload;
pub const ParseRuleError = types.ParseRuleError;
pub const parseRule = types.parseRule;

pub const run = engine.run;
pub const resume_ = engine.resume_;
pub const ResumeError = engine.ResumeError;

pub const Metrics = metrics_mod.Metrics;
pub const computeMetrics = metrics_mod.computeMetrics;

pub const configHash = export_mod.configHash;
pub const exportCanonicalJson = export_mod.exportCanonicalJson;
pub const exportHashHex = export_mod.exportHashHex;
pub const exportDot = export_mod.exportDot;

pub const EXPORT_KEY_LATEST = persist_mod.EXPORT_KEY_LATEST;
pub const persistToWdbx = persist_mod.persistToWdbx;
pub const loadExportFromWdbx = persist_mod.loadExportFromWdbx;

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
