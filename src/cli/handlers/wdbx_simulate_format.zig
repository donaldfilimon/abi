//! Help text and human summary formatting for `abi wdbx simulate`.
const std = @import("std");
const features = @import("abi").features;
const wdbx = features.wdbx;

pub fn diag(comptime fmt: []const u8, args: anytype) u8 {
    std.debug.print("simulate: " ++ fmt ++ "\n", .{} ++ args);
    return 2;
}

pub fn help() u8 {
    std.debug.print(
        \\usage: abi wdbx simulate [options]
        \\
        \\Run a bounded multiway (Wolfram-style) string-rewriting experiment.
        \\Simulates a finite, explicitly bounded slice of rule space; it does not
        \\enumerate "the ruliad" and makes no physics claims.
        \\
        \\Rules & initial states
        \\  --initial <STATE>        Initial state (repeatable; at least one required)
        \\  --rule '<LHS->RHS>'      Inline rewriting rule (repeatable)
        \\  --rules-file <PATH>      One rule per line; blank lines and # comments ignored
        \\  --config <PATH>          JSON experiment config (flags override file values)
        \\
        \\Bounds (hard limits; partial results are returned when one is reached)
        \\  --depth <N>              Maximum depth (default 5)
        \\  --max-states <N>         Maximum unique states (default 10000)
        \\  --max-events <N>         Maximum events (default 100000)
        \\  --max-payload <N>        Maximum state payload bytes (default 4096)
        \\  --deadline-ms <N>        Wall-clock budget in ms (0 = unlimited)
        \\  --max-memory <N>         Approximate engine memory budget in bytes (0 = unlimited)
        \\
        \\Reproducibility
        \\  --seed <N>               Recorded random seed (engine is deterministic)
        \\  --workers <N>            Recorded worker count (expansion is single-threaded)
        \\
        \\Output & persistence
        \\  --format <summary|json|dot>  Output format (default summary)
        \\  --output <PATH>          Write json/dot export to a file instead of printing
        \\  --store <PATH>           Persist experiment into a WDBX checkpoint at PATH
        \\  --resume <PATH>          Resume from a canonical JSON export file
        \\  --resume-wdbx <PATH>     Resume from the latest experiment in a WDBX checkpoint
        \\  --dry-run                Validate configuration and print the plan, then exit
        \\  --quiet                  Suppress the human summary
        \\  --verbose                Print per-depth tables and extra detail
        \\
        \\Ctrl-C cancels a running experiment; the partial result is still
        \\summarized/exported with termination reason "cancelled".
        \\
        \\example:
        \\  abi wdbx simulate --initial A --rule 'A->AB' --rule 'A->BA' --rule 'BB->A' \
        \\      --depth 5 --max-states 500 --max-events 5000 --format json --output experiment.json
        \\
    , .{});
    return 0;
}

pub fn printSummary(result: *const wdbx.multiway.Result, metrics: *const wdbx.multiway.Metrics, export_hash: *const [64]u8, verbose: bool) void {
    const elapsed_ms = @as(f64, @floatFromInt(result.elapsed_ns)) / @as(f64, std.time.ns_per_ms);
    const elapsed_s = @max(elapsed_ms / 1000.0, 1e-9);
    std.debug.print(
        \\multiway simulation
        \\  states (unique):        {d}
        \\  events:                 {d}
        \\  transitions (unique):   {d}
        \\  termination:            {s}
        \\  exhaustive in domain:   {}
        \\  mean out-degree:        {d:.3} (unique transitions / unique states)
        \\  max out-degree:         {d}
        \\  median out-degree:      {d:.1}
        \\  convergent states:      {d} (distinct-predecessor in-degree > 1)
        \\  self-loops:             {d}
        \\  cycle present:          {}
        \\  weakly connected comps: {d}
        \\  payload bytes max/mean: {d}/{d:.2}
        \\  runtime:                {d:.3} ms ({d:.0} states/s, {d:.0} events/s)
        \\  export sha256:          {s}
        \\
    , .{
        metrics.unique_states,
        metrics.event_count,
        metrics.unique_transitions,
        metrics.termination.label(),
        metrics.exhaustive,
        metrics.mean_out_degree,
        metrics.max_out_degree,
        metrics.median_out_degree,
        metrics.convergent_states,
        metrics.self_loops,
        metrics.has_cycle,
        metrics.weakly_connected_components,
        metrics.max_payload_bytes,
        metrics.mean_payload_bytes,
        elapsed_ms,
        @as(f64, @floatFromInt(metrics.unique_states)) / elapsed_s,
        @as(f64, @floatFromInt(metrics.event_count)) / elapsed_s,
        export_hash,
    });
    if (verbose) {
        std.debug.print("  depth  states  events  growth\n", .{});
        for (metrics.states_per_depth, 0..) |states, depth| {
            const growth: f64 = if (depth >= 1 and depth - 1 < metrics.growth_rates.len) metrics.growth_rates[depth - 1] else 0.0;
            std.debug.print("  {d: >5}  {d: >6}  {d: >6}  {d: >6.2}\n", .{ depth, states, metrics.events_per_depth[depth], growth });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (run through the public wdbx handler surface)
// ---------------------------------------------------------------------------

test {
    std.testing.refAllDecls(@This());
}
