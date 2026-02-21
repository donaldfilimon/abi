const std = @import("std");
const abi = @import("abi");

const ProviderId = abi.ai.llm.providers.ProviderId;

pub fn runProviders(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len > 0) {
        const cmd = std.mem.sliceTo(args[0], 0);
        if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "help")) {
            printProvidersHelp();
            return;
        }
        if (std.mem.eql(u8, cmd, "check")) {
            if (args.len < 2) {
                std.debug.print("Usage: abi llm providers check <id>\n", .{});
                return;
            }
            const id_text = std.mem.sliceTo(args[1], 0);
            const provider = parseProviderId(id_text) orelse {
                std.debug.print("Unknown provider: {s}\n", .{id_text});
                return;
            };
            const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
            std.debug.print("{s}: {s}\n", .{ provider.label(), if (available) "available" else "unavailable" });
            return;
        }
    }

    printProviderTable(allocator);
}

fn printProviderTable(allocator: std.mem.Allocator) void {
    std.debug.print("LLM providers (local-first)\n", .{});
    std.debug.print("===========================\n", .{});

    inline for (abi.ai.llm.providers.registry.all_providers) |provider| {
        const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
        std.debug.print("  {s:12}  {s}\n", .{ provider.label(), if (available) "available" else "unavailable" });
    }

    std.debug.print("\nDefault chain (model path):\n", .{});
    printChain(abi.ai.llm.providers.registry.file_model_chain[0..]);
    std.debug.print("Default chain (model id):\n", .{});
    printChain(abi.ai.llm.providers.registry.model_name_chain[0..]);
}

fn printChain(chain: []const ProviderId) void {
    for (chain, 0..) |provider, idx| {
        if (idx != 0) std.debug.print(" -> ", .{});
        std.debug.print("{s}", .{provider.label()});
    }
    std.debug.print("\n", .{});
}

fn parseProviderId(value: []const u8) ?ProviderId {
    if (ProviderId.fromString(value)) |provider| return provider;

    if (std.mem.eql(u8, value, "llama-cpp")) return .llama_cpp;
    if (std.mem.eql(u8, value, "lm-studio")) return .lm_studio;
    if (std.mem.eql(u8, value, "plugin-http")) return .plugin_http;
    if (std.mem.eql(u8, value, "plugin-native")) return .plugin_native;
    if (std.mem.eql(u8, value, "local-gguf")) return .local_gguf;

    return null;
}

pub fn printProvidersHelp() void {
    std.debug.print(
        "Usage: abi llm providers [list|check <id>]\\n\\n" ++
            "List provider availability and routing order.\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm providers\\n" ++
            "  abi llm providers check ollama\\n",
        .{},
    );
}
