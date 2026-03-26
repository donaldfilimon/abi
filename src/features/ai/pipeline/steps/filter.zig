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
