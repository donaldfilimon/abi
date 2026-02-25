const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

const ProviderId = abi.ai.llm.providers.ProviderId;
const provider_parser = abi.ai.llm.providers.parser;

pub fn runProviders(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len > 0) {
        const cmd = std.mem.sliceTo(args[0], 0);
        if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "help")) {
            printProvidersHelp();
            return;
        }
        if (std.mem.eql(u8, cmd, "check")) {
            if (args.len < 2) {
                utils.output.println("Usage: abi llm providers check <id>", .{});
                return;
            }
            const id_text = std.mem.sliceTo(args[1], 0);
            const provider = provider_parser.parseProviderId(id_text) orelse {
                utils.output.printError("Unknown provider: {s}", .{id_text});
                return;
            };
            const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
            utils.output.println("{s}: {s}", .{ provider.label(), if (available) "available" else "unavailable" });
            return;
        }
    }

    printProviderTable(allocator);
}

fn printProviderTable(allocator: std.mem.Allocator) void {
    utils.output.printHeader("LLM providers (local-first)");

    inline for (abi.ai.llm.providers.registry.all_providers) |provider| {
        const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
        utils.output.println("  {s:12}  {s}", .{ provider.label(), if (available) "available" else "unavailable" });
    }

    utils.output.println("", .{});
    utils.output.println("Default chain (model path):", .{});
    printChain(abi.ai.llm.providers.registry.file_model_chain[0..]);
    utils.output.println("Default chain (model id):", .{});
    printChain(abi.ai.llm.providers.registry.model_name_chain[0..]);
}

fn printChain(chain: []const ProviderId) void {
    for (chain, 0..) |provider, idx| {
        if (idx != 0) utils.output.print(" -> ", .{});
        utils.output.print("{s}", .{provider.label()});
    }
    utils.output.println("", .{});
}

pub fn printProvidersHelp() void {
    utils.output.print(
        "Usage: abi llm providers [list|check <id>]\\n\\n" ++
            "List provider availability and routing order.\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm providers\\n" ++
            "  abi llm providers check ollama\\n",
        .{},
    );
}
