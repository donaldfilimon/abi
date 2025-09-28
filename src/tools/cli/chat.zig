const std = @import("std");
const abi = @import("abi");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "chat",
    .summary = "Interact with the ABI conversational agent",
    .usage = "abi chat [--persona <type>] [--backend <provider>] [--model <name>] [--interactive] [message]",
    .details = "  --persona      Select persona (creative, analytical, helpful)\n" ++
        "  --backend      Choose backend provider (openai, ollama)\n" ++
        "  --model        Model identifier\n" ++
        "  --interactive  Start interactive chat session\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    var persona: ?[]const u8 = null;
    var backend: ?[]const u8 = null;
    var model: ?[]const u8 = null;
    var interactive: bool = false;
    var message: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--persona") and i + 1 < args.len) {
            persona = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            backend = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            model = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--interactive")) {
            interactive = true;
        } else if (i == 2 and !std.mem.startsWith(u8, args[i], "--")) {
            message = args[i];
        }
    }

    const final_persona = persona orelse "creative";
    const final_backend = backend orelse "openai";
    const final_model = model orelse "gpt-3.5-turbo";

    const agent_config = abi.ai.enhanced_agent.AgentConfig{
        .name = "ABI Assistant",
        .enable_logging = true,
        .max_concurrent_requests = 5,
    };

    var agent = try abi.ai.enhanced_agent.EnhancedAgent.init(allocator, agent_config);
    defer agent.deinit();

    if (message) |msg| {
        const response = try agent.processInput(msg);
        defer allocator.free(response);
        std.debug.print("{s}\n", .{response});
        return;
    }

    if (interactive) {
        std.debug.print("Chat mode (type 'quit' to exit, 'help' for commands)\n", .{});
        std.debug.print("Persona: {s}, Backend: {any}, Model: {any}\n", .{ final_persona, final_backend, final_model });
        std.debug.print("Interactive Chat Mode - Type 'quit' to exit, 'help' for commands\n", .{});
        std.debug.print("Note: Full interactive mode requires additional I/O implementation\n", .{});
        std.debug.print("For now, this is a demonstration of the chat framework.\n", .{});
        return;
    }

    std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
}
