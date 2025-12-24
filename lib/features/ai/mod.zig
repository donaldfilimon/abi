//! AI Feature Module
//!
//! Neural networks, transformers, and machine learning

const std = @import("std");

const Conn = @import("../connectors/mod.zig");
const Wdbx = @import("../database/wdbx_adapter.zig");
pub const agent = @import("agent.zig");
pub const federated = @import("federated/mod.zig");
pub const layers = @import("layers.zig");
pub const model_registry = @import("model_registry.zig");
pub const optimization = @import("optimization/mod.zig");
const Policy = @import("policy.zig");
pub const reinforcement_learning = @import("reinforcement_learning/mod.zig");
const Retry = @import("retry.zig");
const Schema = @import("schema.zig");
pub const training = @import("training/mod.zig");
pub const transformer = @import("transformer/mod.zig");

/// Initialize the AI feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global AI state to initialize
}

/// Deinitialize the AI feature module
pub fn deinit() void {
    // Currently no global AI state to cleanup
}

// Legacy compatibility
pub const Envelope = struct {
    id: []const u8,
    intent: []const u8,
    payload: []const u8,
    sensitivity: enum { low, medium, high } = .low,
};

pub const Controller = struct {
    allocator: std.mem.Allocator,
    policy: Policy.Policy,
    connector: Conn.Connector,

    pub fn init(allocator: std.mem.Allocator, policy: Policy.Policy, connector: Conn.Connector) !Controller {
        try connector.init(allocator);
        return .{ .allocator = allocator, .policy = policy, .connector = connector };
    }

    pub fn summarize(self: *Controller, input: Schema.SummarizeInput) !Schema.SummarizeOutput {
        try input.validate();

        const doc = try Wdbx.getDocument(self.allocator, input.doc_id);
        defer self.allocator.free(doc);

        const prompt = try std.fmt.allocPrint(self.allocator, "Summarize:\n{s}\n", .{doc});
        defer self.allocator.free(prompt);

        var attempt: u8 = 0;
        while (true) {
            const res = try self.connector.call(self.allocator, .{
                .model = "gpt-oss-default",
                .prompt = prompt,
                .max_tokens = input.max_tokens,
            });
            defer if (res.ok and res.content.len > 0) self.allocator.free(res.content);

            if (res.ok) {
                const summary_copy = try std.mem.dupe(self.allocator, u8, res.content);
                var out = Schema.SummarizeOutput{
                    .summary = summary_copy,
                    .tokens_used = res.tokens_in + res.tokens_out,
                };
                errdefer self.allocator.free(summary_copy);
                try out.validate();
                try Wdbx.persistSummary(self.allocator, input.doc_id, out.summary);
                return out;
            }

            attempt += 1;
            if (attempt >= self.policy.retry.max_attempts) {
                return error.ModelFailed;
            }
            const delay = Retry.backoff_ms(attempt, self.policy.retry.base_ms, self.policy.retry.factor);
            std.time.sleep(@as(u64, delay) * std.time.ns_per_ms);
        }
    }
};
