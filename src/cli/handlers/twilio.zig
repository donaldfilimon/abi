const std = @import("std");
const ai = @import("../../features/ai/mod.zig");
const twilio_conn = @import("../../connectors/twilio.zig");
const usage_mod = @import("../usage.zig");

pub fn handleTwilio(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4 or !std.mem.eql(u8, args[2], "simulate")) return usage_mod.usageError("usage: abi twilio simulate <input>");

    const input = args[3];
    const agent_reply = try ai.run(allocator, input);
    defer allocator.free(agent_reply);

    var client = twilio_conn.Client.init(allocator, twilio_conn.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "local-conversation",
        .customer_id = "local-customer",
        .transcript = input,
    }, agent_reply);
    defer response.deinit(allocator);

    std.debug.print("Twilio ConversationRelay simulation\n", .{});
    std.debug.print("response: {s}\n", .{response.text});
    if (response.escalation) |payload| {
        std.debug.print("escalation: true\n", .{});
        std.debug.print("reason: {s}\n", .{payload.reason_code});
        std.debug.print("routing_hints: {s}\n", .{payload.routing_hints});
        std.debug.print("summary: {s}\n", .{payload.summary});
    } else {
        std.debug.print("escalation: false\n", .{});
    }
    return 0;
}

test "twilio dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    // Wrong arity and a non-`simulate` subcommand both reject with usage (exit 2)
    // before any AI run or connector init.
    try std.testing.expectEqual(@as(u8, 2), try handleTwilio(allocator, &.{ "abi", "twilio" }));
    try std.testing.expectEqual(@as(u8, 2), try handleTwilio(allocator, &.{ "abi", "twilio", "notsimulate", "hi" }));
    try std.testing.expectEqual(@as(u8, 2), try handleTwilio(allocator, &.{ "abi", "twilio", "simulate", "a", "b" }));
}

test {
    std.testing.refAllDecls(@This());
}
