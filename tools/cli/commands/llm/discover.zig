const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

pub fn runDiscover(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (args.len > 0) {
        const arg = std.mem.sliceTo(args[0], 0);
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help")) {
            printHelp();
            return;
        }
    }

    utils.output.printHeader("LLM Provider Discovery");
    utils.output.println("", .{});

    const providers = .{
        .{ "local_gguf", "Built-in GGUF engine", "Always available — load .gguf model files directly" },
        .{ "llama_cpp", "llama.cpp server", "Set LLAMA_CPP_HOST or run: llama-server -m model.gguf" },
        .{ "mlx", "Apple MLX server", "Set MLX_HOST or run: python -m mlx_lm.server --model <model>" },
        .{ "ollama", "Ollama", "Set OLLAMA_HOST or run: ollama serve" },
        .{ "ollama_passthrough", "Ollama passthrough", "Set OLLAMA_PASSTHROUGH_URL for OpenAI-compatible passthrough endpoint" },
        .{ "lm_studio", "LM Studio", "Set LM_STUDIO_HOST or open LM Studio with server enabled" },
        .{ "vllm", "vLLM server", "Set VLLM_HOST or run: python -m vllm.entrypoints.openai.api_server" },
        .{ "codex", "Codex", "Set CODEX_API_KEY and optional CODEX_BASE_URL/CODEX_MODEL" },
        .{ "opencode", "OpenCode", "Set OPENCODE_API_KEY and optional OPENCODE_BASE_URL/OPENCODE_MODEL" },
        .{ "claude", "Claude", "Set CLAUDE_API_KEY (or ANTHROPIC_API_KEY fallback)" },
        .{ "gemini", "Gemini", "Set GEMINI_API_KEY and optional GEMINI_MODEL/GEMINI_BASE_URL" },
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

        const provider_id = comptime abi.features.ai.llm.providers.ProviderId.fromString(id_str).?;
        const is_available = abi.features.ai.llm.providers.health.isAvailable(allocator, provider_id, null);

        if (is_available) {
            available_count += 1;
            utils.output.printStatusLineFmt("{s:16}", .{display_name}, true);
        } else {
            utils.output.printStatusLineFmt("{s:16} {s}", .{ display_name, hint }, false);
        }
    }

    utils.output.printCountSummary(available_count, providers.len, "providers available");
    utils.output.println("", .{});

    utils.output.println("Default routing (model file path):", .{});
    utils.output.print("  ", .{});
    for (abi.features.ai.llm.providers.registry.file_model_chain[0..], 0..) |p, idx| {
        if (idx != 0) utils.output.print(" -> ", .{});
        utils.output.print("{s}", .{p.label()});
    }
    utils.output.println("", .{});
    utils.output.println("", .{});

    utils.output.println("Default routing (model name):", .{});
    utils.output.print("  ", .{});
    for (abi.features.ai.llm.providers.registry.model_name_chain[0..], 0..) |p, idx| {
        if (idx != 0) utils.output.print(" -> ", .{});
        utils.output.print("{s}", .{p.label()});
    }
    utils.output.println("", .{});
    utils.output.println("", .{});

    utils.output.println("Quick start:", .{});
    utils.output.println("  abi llm session --model llama3 --backend ollama", .{});
    utils.output.println("  abi llm session --model claude-3-5-sonnet-20241022 --backend anthropic", .{});
    utils.output.println("  abi llm session --model gpt-4 --backend openai", .{});
    utils.output.println("  abi llm session --sync --sync-providers codex,opencode,claude,gemini,ollama_passthrough,ollama", .{});
    utils.output.println("  abi ralph run --backend anthropic --model claude-3-5-sonnet-20241022 --task \"...\"", .{});
    utils.output.println("  abi ralph improve --backend ollama --model llama3", .{});
}

fn printHelp() void {
    utils.output.print(
        "Usage: abi llm discover\n\n" ++
            "Auto-discover available LLM providers and show routing configuration.\n\n" ++
            "Probes all known provider endpoints (local servers, cloud APIs, plugins)\n" ++
            "and displays their availability with setup hints.\n",
        .{},
    );
}

test {
    std.testing.refAllDecls(@This());
}
