//! Filter Step — User-provided predicate.
//!
//! Evaluates a predicate against the pipeline context.
//! If false and halt_on_false is set, returns FilterHalted
//! to stop pipeline execution early.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;
const PipelineError = types.PipelineError;

pub fn execute(pctx: *PipelineContext, cfg: types.FilterConfig) !void {
    // Cast PipelineContext pointer to opaque for the filter function
    const ctx_ptr: *const anyopaque = @ptrCast(pctx);
    const passed = cfg.predicate(ctx_ptr);

    if (!passed and cfg.halt_on_false) {
        return PipelineError.FilterHalted;
    }
}

fn alwaysTrue(_: *const anyopaque) bool {
    return true;
}

fn alwaysFalse(_: *const anyopaque) bool {
    return false;
}

test "filter passes when predicate returns true" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-1", 1);
    defer pctx.deinit();

    try execute(&pctx, .{ .predicate = &alwaysTrue, .halt_on_false = true });
}

test "filter halts when predicate returns false and halt_on_false set" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-2", 2);
    defer pctx.deinit();

    const result = execute(&pctx, .{ .predicate = &alwaysFalse, .halt_on_false = true });
    try std.testing.expectError(PipelineError.FilterHalted, result);
}

test "filter does not halt when halt_on_false is false" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "session-3", 3);
    defer pctx.deinit();

    // Should not error even though predicate returns false
    try execute(&pctx, .{ .predicate = &alwaysFalse, .halt_on_false = false });
}
