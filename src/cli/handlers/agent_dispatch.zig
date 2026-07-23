//! Top-level `abi agent <subcommand>` dispatch.
//!
//! Routes the frozen agent surface (`plan|train|tui|os|multi|spawn|browser`)
//! to sibling leaf handlers. Argc checks stay here; execution lives in leaves.

const std = @import("std");
const usage_mod = @import("../usage.zig");
const help = @import("agent_help.zig");
const plan = @import("agent_plan.zig");
const train = @import("agent_train.zig");
const tui = @import("agent_tui.zig");
const multi = @import("agent_multi.zig");
const os = @import("agent_os.zig");

/// `abi agent <plan|train|tui|os|multi|spawn|browser> ...`: dispatch agent subcommands.
/// Returns the process exit code.
pub fn handleAgent(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi agent <plan|train|tui|os|multi|spawn|browser> ...");

    const sub_cmd = args[2];
    if (usage_mod.isHelpToken(sub_cmd)) return usage_mod.printCommandHelp("agent");
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentPlanHelp();
        return handleAgentPlan(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentTrainHelp();
        return handleAgentTrain(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentTuiHelp();
        return handleAgentTui(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "os")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentOsHelp();
        return os.handleAgentOs(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "multi")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentMultiHelp();
        return handleAgentMulti(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "spawn")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentSpawnHelp();
        return multi.handleAgentSpawnArgv(io, allocator, args);
    } else if (std.mem.eql(u8, sub_cmd, "browser")) {
        if (args.len == 4 and usage_mod.isHelpToken(args[3])) return help.agentBrowserHelp();
        return multi.handleAgentBrowserArgv(io, allocator, args);
    } else {
        return usage_mod.usageError("usage: abi agent <plan|train|tui|os|multi|spawn|browser> ...");
    }
}

fn handleAgentPlan(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent plan <input>");
    return plan.handleAgentPlanInput(io, allocator, args[3]);
}

fn handleAgentTrain(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent train <abbey|aviva|abi|all>");
    return train.handleAgentTrainProfile(io, allocator, args[3]);
}

fn handleAgentTui(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3) return usage_mod.usageError("usage: abi agent tui");
    return tui.handleAgentTuiNoArgs(io, allocator);
}

fn handleAgentMulti(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4) return usage_mod.usageError("usage: abi agent multi <input>");
    return multi.handleAgentMultiInput(io, allocator, args[3]);
}

test "agent dispatch rejects malformed grammar with exit code 2" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "bogus" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "plan" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "train" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "execute", "ls" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "execute", "ls", "--confirm" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "multi" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn" }));
    try std.testing.expectEqual(@as(u8, 2), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn", "--workers" }));
    try std.testing.expectEqual(@as(u8, 2), try multi.handleAgentBrowserArgv(t, allocator, &.{ "abi", "agent", "browser", "--execute", "open docs" }));
    try std.testing.expectEqual(@as(u8, 2), try multi.handleAgentBrowserArgv(t, allocator, &.{ "abi", "agent", "browser" }));
}

test "agent handler help returns success before side effects" {
    const allocator = std.testing.allocator;
    const t = std.testing.io;
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "plan", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "train", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "tui", "help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "os", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "multi", "--help" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "spawn", "-h" }));
    try std.testing.expectEqual(@as(u8, 0), try handleAgent(t, allocator, &.{ "abi", "agent", "browser", "help" }));
}

test {
    std.testing.refAllDecls(@This());
}
