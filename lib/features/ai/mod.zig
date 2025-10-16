const std = @import("std");
const Schema = @import("schema.zig");
const Conn = @import("../connectors/mod.zig");
const Retry = @import("retry.zig");
const Policy = @import("policy.zig");
const Wdbx = @import("../features/database/wdbx_adapter.zig");

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
