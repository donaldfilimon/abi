//! Pipeline Executor
//!
//! Runs a sequence of pipeline steps, threading a PipelineContext through each.
//! Every step execution is recorded as a WDBX ConversationBlock with pipeline
//! metadata (pipeline_id, step_index, step_tag).

const std = @import("std");
const types = @import("types.zig");
const ctx_mod = @import("context.zig");
const persistence = @import("persistence.zig");
const Step = types.Step;
const StepKind = types.StepKind;
const BlockChain = types.BlockChain;
const PipelineResult = types.PipelineResult;
const PipelineError = types.PipelineError;
const PipelineContext = ctx_mod.PipelineContext;

// Step implementations
const retrieve_step = @import("steps/retrieve.zig");
const template_step = @import("steps/template.zig");
const store_step = @import("steps/store.zig");
const route_step = @import("steps/route.zig");
const generate_step = @import("steps/generate.zig");
const validate_step = @import("steps/validate.zig");
const modulate_step = @import("steps/modulate.zig");
const reason_step = @import("steps/reason.zig");
const transform_step = @import("steps/transform.zig");
const filter_step = @import("steps/filter.zig");

pub const Pipeline = struct {
    allocator: std.mem.Allocator,
    session_id: []const u8,
    steps: []Step,
    chain: ?*BlockChain,
    pipeline_id: u64,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        session_id: []const u8,
        steps: []Step,
        chain: ?*BlockChain,
    ) Self {
        // Generate a unique pipeline ID from timestamp + hash
        const ts: u64 = @bitCast(std.time.timestamp());
        const session_hash = std.hash.Fnv1a_64.hash(session_id);
        const pid = ts ^ session_hash;

        return .{
            .allocator = allocator,
            .session_id = session_id,
            .steps = steps,
            .chain = chain,
            .pipeline_id = pid,
        };
    }

    /// Execute the pipeline with the given input.
    pub fn run(self: *Self, input: []const u8) !PipelineResult {
        const start_ts = std.time.timestamp();

        var pctx = try PipelineContext.init(
            self.allocator,
            input,
            self.session_id,
            self.pipeline_id,
        );
        defer pctx.deinit();
        pctx.chain = self.chain;

        var steps_executed: u16 = 0;

        for (self.steps, 0..) |step, idx| {
            pctx.current_step = @intCast(idx);

            const step_result = executeStep(&pctx, step);

            if (step_result) |_| {
                // Record WDBX block for this step
                if (self.chain != null) {
                    const block_id = persistence.PipelineBlockAdapter.recordStepBlock(
                        &pctx,
                        step.kind,
                    ) catch 0;
                    if (block_id > 0) {
                        pctx.recordBlock(block_id) catch {};
                    }
                }
                steps_executed += 1;
            } else |err| switch (err) {
                PipelineError.FilterHalted => break,
                else => return err,
            }
        }

        const end_ts = std.time.timestamp();
        const elapsed: u64 = @intCast(end_ts - start_ts);

        // Build result — transfer ownership of response and block_ids
        const response = if (pctx.generated_response) |r|
            try self.allocator.dupe(u8, r)
        else
            null;

        const block_ids = if (pctx.block_ids.items.len > 0)
            try self.allocator.dupe(u64, pctx.block_ids.items)
        else
            &[_]u64{};

        return PipelineResult{
            .response = response,
            .block_ids = block_ids,
            .pipeline_id = self.pipeline_id,
            .steps_executed = steps_executed,
            .validation_passed = pctx.validation_passed,
            .elapsed_ms = elapsed * 1000,
            .allocator = self.allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.steps) |step| {
            switch (step.config) {
                .template => |cfg| self.allocator.free(cfg.template_str),
                else => {},
            }
        }
        self.allocator.free(self.steps);
    }
};

fn executeStep(pctx: *PipelineContext, step: Step) !void {
    switch (step.config) {
        .retrieve => |cfg| try retrieve_step.execute(pctx, cfg),
        .template => |cfg| try template_step.execute(pctx, cfg),
        .route => |cfg| try route_step.execute(pctx, cfg),
        .modulate => |cfg| try modulate_step.execute(pctx, cfg),
        .generate => |cfg| try generate_step.execute(pctx, cfg),
        .validate => |cfg| try validate_step.execute(pctx, cfg),
        .store => |cfg| try store_step.execute(pctx, cfg),
        .transform => |cfg| try transform_step.execute(pctx, cfg),
        .filter => |cfg| try filter_step.execute(pctx, cfg),
        .reason => |cfg| try reason_step.execute(pctx, cfg),
    }
}
