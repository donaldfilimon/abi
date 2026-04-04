//! Pipeline Builder
//!
//! Chainable builder that accumulates pipeline steps at runtime.
//! The step types are compile-time (tagged union), but the list
//! is runtime since template strings and configs are runtime data.
//!
//! Usage:
//!   var builder = PipelineBuilder.init(allocator, "session-123");
//!   var pipeline = builder
//!       .retrieve(.wdbx, .{ .k = 5 })
//!       .template("Given {context}, respond as Abbey...")
//!       .route(.adaptive)
//!       .generate(.{})
//!       .validate(.constitution)
//!       .store(.wdbx)
//!       .build();

const std = @import("std");
const types = @import("types.zig");
const Step = types.Step;
const StepKind = types.StepKind;
const StepConfig = types.StepConfig;
const BlockChain = types.BlockChain;

pub const PipelineBuilder = struct {
    allocator: std.mem.Allocator,
    session_id: []const u8,
    session_id_owned: bool = false,
    steps: std.ArrayListUnmanaged(Step),
    wdbx_chain: ?*BlockChain = null,
    build_error: ?anyerror = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, session_id: []const u8) Self {
        const owned_id = allocator.dupe(u8, session_id) catch {
            return .{
                .allocator = allocator,
                .session_id = session_id,
                .steps = .empty,
            };
        };
        return .{
            .allocator = allocator,
            .session_id = owned_id,
            .session_id_owned = true,
            .steps = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free owned template strings
        for (self.steps.items) |step| {
            switch (step.config) {
                .template => |cfg| self.allocator.free(cfg.template_str),
                else => {},
            }
        }
        self.steps.deinit(self.allocator);
        if (self.session_id_owned) {
            self.allocator.free(self.session_id);
        }
    }

    /// Attach a WDBX block chain for persistence.
    pub fn withChain(self: *Self, wdbx_chain: *BlockChain) *Self {
        self.wdbx_chain = wdbx_chain;
        return self;
    }

    /// Add a retrieve step — pull context from WDBX.
    pub fn retrieve(self: *Self, _: types.RetrieveSource, cfg: types.RetrieveConfig) *Self {
        self.appendStep(.{ .kind = .retrieve, .config = .{ .retrieve = cfg } });
        return self;
    }

    /// Add a template step — render prompt with {variable} interpolation.
    pub fn template(self: *Self, template_str: []const u8) *Self {
        const owned = self.allocator.dupe(u8, template_str) catch return self;
        self.appendStep(.{ .kind = .template, .config = .{ .template = .{ .template_str = owned } } });
        return self;
    }

    /// Add a route step — profile routing decision.
    pub fn route(self: *Self, strategy: types.RouteStrategy) *Self {
        self.appendStep(.{ .kind = .route, .config = .{ .route = .{ .strategy = strategy } } });
        return self;
    }

    /// Add a modulate step — EMA preference adjustment.
    pub fn modulate(self: *Self) *Self {
        self.appendStep(.{ .kind = .modulate, .config = .{ .modulate = .{} } });
        return self;
    }

    /// Add a generate step — LLM inference.
    pub fn generate(self: *Self, cfg: types.GenerateConfig) *Self {
        self.appendStep(.{ .kind = .generate, .config = .{ .generate = cfg } });
        return self;
    }

    /// Add a validate step — constitution check.
    pub fn validate(self: *Self, target: types.ValidateTarget) *Self {
        self.appendStep(.{ .kind = .validate, .config = .{ .validate = .{ .target = target } } });
        return self;
    }

    /// Add a store step — persist to WDBX.
    pub fn store(self: *Self, target: types.StoreTarget) *Self {
        self.appendStep(.{ .kind = .store, .config = .{ .store = .{ .target = target } } });
        return self;
    }

    /// Add a transform step — user-provided transformation.
    pub fn transform(self: *Self, transform_fn: types.TransformFn) *Self {
        self.appendStep(.{ .kind = .transform, .config = .{ .transform = .{ .transform_fn = transform_fn } } });
        return self;
    }

    /// Add a filter step — user-provided predicate.
    pub fn filter(self: *Self, predicate: types.FilterFn, halt_on_false: bool) *Self {
        self.appendStep(.{ .kind = .filter, .config = .{ .filter = .{ .predicate = predicate, .halt_on_false = halt_on_false } } });
        return self;
    }

    /// Add a reason step — chain-of-thought reasoning.
    pub fn reason(self: *Self, cfg: types.ReasonConfig) *Self {
        self.appendStep(.{ .kind = .reason, .config = .{ .reason = cfg } });
        return self;
    }

    /// Build the pipeline. Consumes the step list.
    /// Returns an error if any step addition failed (e.g., OOM).
    pub fn build(self: *Self) !@import("executor.zig").Pipeline {
        if (self.build_error) |err| {
            // Clean up any partially-added steps
            for (self.steps.items) |step| {
                switch (step.config) {
                    .template => |cfg| self.allocator.free(cfg.template_str),
                    else => {},
                }
            }
            self.steps.deinit(self.allocator);
            self.steps = .empty;
            return err;
        }

        const steps = self.steps.toOwnedSlice(self.allocator) catch |err| {
            // OOM: free owned step data before clearing to avoid leaks
            for (self.steps.items) |step| {
                switch (step.config) {
                    .template => |cfg| self.allocator.free(cfg.template_str),
                    else => {},
                }
            }
            self.steps.deinit(self.allocator);
            self.steps = .empty;
            return err;
        };
        self.steps = .empty;

        return @import("executor.zig").Pipeline.init(
            self.allocator,
            self.session_id,
            steps,
            self.wdbx_chain,
        );
    }

    fn appendStep(self: *Self, step: Step) void {
        if (self.build_error) |_| return;
        self.steps.append(self.allocator, step) catch |err| {
            self.build_error = err;
        };
    }
};
