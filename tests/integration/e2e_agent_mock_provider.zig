const std = @import("std");
const Agent = @import("../../src/agent/mod.zig");
const Policy = @import("../../src/agent/policy.zig");
const Mock = @import("../../src/connectors/mock.zig");
const Schema = @import("../../src/agent/schema.zig");

test "agent summarizes with mock provider" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var controller = try Agent.Controller.init(allocator, Policy.Policy{ .providers = &.{.{ .name = "mock", .allowed = true }} }, Mock.get());
    const input = Schema.SummarizeInput{ .doc_id = "doc-1", .max_tokens = 64 };
    const result = try controller.summarize(input);
    defer allocator.free(result.summary);
    try std.testing.expect(result.summary.len > 0);
}
