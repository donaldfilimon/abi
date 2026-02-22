const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");

pub fn runDiscover(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len > 0) {
        const arg = std.mem.sliceTo(args[0], 0);
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printHelp();
            return;
        }
    }

    std.debug.print("\nLLM Provider Discovery\n", .{});
    std.debug.print("======================\n\n", .{});

    const providers = .{
        .{ "local_gguf", "Built-in GGUF engine", "Always available — load .gguf model files directly" },
        .{ "llama_cpp", "llama.cpp server", "Set LLAMA_CPP_HOST or run: llama-server -m model.gguf" },
        .{ "mlx", "Apple MLX server", "Set MLX_HOST or run: python -m mlx_lm.server --model <model>" },
        .{ "ollama", "Ollama", "Set OLLAMA_HOST or run: ollama serve" },
        .{ "lm_studio", "LM Studio", "Set LM_STUDIO_HOST or open LM Studio with server enabled" },
        .{ "vllm", "vLLM server", "Set VLLM_HOST or run: python -m vllm.entrypoints.openai.api_server" },
        .{ "anthropic", "Anthropic Claude", "Set ANTHROPIC_API_KEY (get key at console.anthropic.com)" },
        .{ "openai", "OpenAI", "Set OPENAI_API_KEY (get key at platform.openai.com)" },
        .{ "plugin_http", "HTTP Plugin", "Add via: abi llm plugins add-http <id> --url <base_url>" },
        .{ "plugin_native", "Native Plugin", "Add via: abi llm plugins add-native <id> --library <path>" },
    };

    var available_count: usize = 0;

    inline for (providers) |entry| {
        const id_str = entry[0];
        const display_name = entry[1];
        const hint = entry[2];

        const provider_id = comptime abi.ai.llm.providers.ProviderId.fromString(id_str).?;
        const is_available = abi.ai.llm.providers.health.isAvailable(allocator, provider_id, null);

        if (is_available) {
            available_count += 1;
            std.debug.print("  [OK]  {s:16} {s}\n", .{ display_name, "" });
        } else {
            std.debug.print("  [ ]   {s:16} {s}\n", .{ display_name, hint });
        }
    }

    std.debug.print("\n{d}/{d} providers available\n\n", .{ available_count, providers.len });

    std.debug.print("Default routing (model file path):\n  ", .{});
    for (abi.ai.llm.providers.registry.file_model_chain[0..], 0..) |p, idx| {
        if (idx != 0) std.debug.print(" -> ", .{});
        std.debug.print("{s}", .{p.label()});
    }
    std.debug.print("\n\n", .{});

    std.debug.print("Default routing (model name):\n  ", .{});
    for (abi.ai.llm.providers.registry.model_name_chain[0..], 0..) |p, idx| {
        if (idx != 0) std.debug.print(" -> ", .{});
        std.debug.print("{s}", .{p.label()});
    }
    std.debug.print("\n\n", .{});

    std.debug.print("Quick start:\n", .{});
    std.debug.print("  abi llm session --model llama3 --backend ollama\n", .{});
    std.debug.print("  abi llm session --model claude-3-5-sonnet-20241022 --backend anthropic\n", .{});
    std.debug.print("  abi llm session --model gpt-4 --backend openai\n", .{});
    std.debug.print("  abi ralph run --backend anthropic --model claude-3-5-sonnet-20241022 --task \"...\"\n", .{});
    std.debug.print("  abi ralph improve --backend ollama --model llama3\n", .{});
}

fn printHelp() void {
    std.debug.print(
        "Usage: abi llm discover\n\n" ++
            "Auto-discover available LLM providers and show routing configuration.\n\n" ++
            "Probes all known provider endpoints (local servers, cloud APIs, plugins)\n" ++
            "and displays their availability with setup hints.\n",
        .{},
    );
}
