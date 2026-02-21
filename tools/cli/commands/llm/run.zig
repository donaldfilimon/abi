const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

const ProviderId = abi.ai.llm.providers.ProviderId;

const RunOptions = struct {
    model: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    backend: ?ProviderId = null,
    fallback: []ProviderId = &.{},
    strict_backend: bool = false,
    plugin_id: ?[]const u8 = null,
    max_tokens: u32 = 256,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 40,
    repetition_penalty: f32 = 1.1,
    json: bool = false,
};

pub fn runRun(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printRunHelp();
        return;
    }

    const parsed = try parseRunArgs(allocator, args);
    defer allocator.free(parsed.fallback);

    if (parsed.model == null) {
        std.debug.print("Error: --model is required.\n\n", .{});
        printRunHelp();
        return;
    }
    if (parsed.prompt == null) {
        std.debug.print("Error: --prompt is required.\n\n", .{});
        printRunHelp();
        return;
    }

    var result = abi.ai.llm.providers.generate(allocator, .{
        .model = parsed.model.?,
        .prompt = parsed.prompt.?,
        .backend = parsed.backend,
        .fallback = parsed.fallback,
        .strict_backend = parsed.strict_backend,
        .plugin_id = parsed.plugin_id,
        .max_tokens = parsed.max_tokens,
        .temperature = parsed.temperature,
        .top_p = parsed.top_p,
        .top_k = parsed.top_k,
        .repetition_penalty = parsed.repetition_penalty,
    }) catch |err| {
        std.debug.print("LLM run failed: {t}\n", .{err});
        return err;
    };
    defer result.deinit(allocator);

    if (parsed.json) {
        try printResultJson(allocator, &result);
        return;
    }

    std.debug.print("provider: {s}\n", .{result.provider.label()});
    std.debug.print("model:    {s}\n\n", .{result.model_used});
    std.debug.print("{s}\n", .{result.content});
}

pub fn parseRunArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !RunOptions {
    var options = RunOptions{};
    var fallback_builder = std.ArrayListUnmanaged(ProviderId).empty;
    errdefer fallback_builder.deinit(allocator);

    var positional_model: ?[]const u8 = null;
    var positional_prompt: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &.{ "--model", "-m" })) {
            if (i < args.len) {
                options.model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--prompt", "-p" })) {
            if (i < args.len) {
                options.prompt = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--backend")) {
            if (i < args.len) {
                const value = std.mem.sliceTo(args[i], 0);
                i += 1;
                options.backend = parseProviderId(value) orelse {
                    std.debug.print("Unknown backend: {s}\n", .{value});
                    return error.InvalidBackend;
                };
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--fallback")) {
            if (i < args.len) {
                const csv = std.mem.sliceTo(args[i], 0);
                i += 1;
                try parseFallbackCsv(allocator, &fallback_builder, csv);
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--strict-backend")) {
            options.strict_backend = true;
            continue;
        }

        if (std.mem.eql(u8, arg, "--plugin")) {
            if (i < args.len) {
                options.plugin_id = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--max-tokens", "-n" })) {
            if (i < args.len) {
                options.max_tokens = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch options.max_tokens;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &.{ "--temperature", "-t" })) {
            if (i < args.len) {
                options.temperature = std.fmt.parseFloat(f32, std.mem.sliceTo(args[i], 0)) catch options.temperature;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--top-p")) {
            if (i < args.len) {
                options.top_p = std.fmt.parseFloat(f32, std.mem.sliceTo(args[i], 0)) catch options.top_p;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--top-k")) {
            if (i < args.len) {
                options.top_k = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch options.top_k;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--repeat-penalty")) {
            if (i < args.len) {
                options.repetition_penalty = std.fmt.parseFloat(f32, std.mem.sliceTo(args[i], 0)) catch options.repetition_penalty;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--json")) {
            options.json = true;
            continue;
        }

        if (std.mem.startsWith(u8, arg, "-")) {
            continue;
        }

        if (positional_model == null) {
            positional_model = arg;
        } else if (positional_prompt == null) {
            positional_prompt = arg;
        }
    }

    options.fallback = try fallback_builder.toOwnedSlice(allocator);

    if (options.model == null) options.model = positional_model;
    if (options.prompt == null) options.prompt = positional_prompt;

    return options;
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

fn parseFallbackCsv(
    allocator: std.mem.Allocator,
    fallback_builder: *std.ArrayListUnmanaged(ProviderId),
    csv: []const u8,
) !void {
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |piece| {
        const trimmed = std.mem.trim(u8, piece, " \t\r\n");
        if (trimmed.len == 0) continue;
        const provider = parseProviderId(trimmed) orelse {
            std.debug.print("Unknown fallback backend: {s}\n", .{trimmed});
            return error.InvalidBackend;
        };
        try appendUnique(allocator, fallback_builder, provider);
    }
}

fn appendUnique(
    allocator: std.mem.Allocator,
    list: *std.ArrayListUnmanaged(ProviderId),
    provider: ProviderId,
) !void {
    for (list.items) |existing| {
        if (existing == provider) return;
    }
    try list.append(allocator, provider);
}

fn printResultJson(allocator: std.mem.Allocator, result: *const abi.ai.llm.providers.GenerateResult) !void {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"provider\":\"");
    try appendEscaped(allocator, &out, result.provider.label());
    try out.appendSlice(allocator, "\",\"model\":\"");
    try appendEscaped(allocator, &out, result.model_used);
    try out.appendSlice(allocator, "\",\"content\":\"");
    try appendEscaped(allocator, &out, result.content);
    try out.appendSlice(allocator, "\"}\n");

    std.debug.print("{s}", .{out.items});
}

fn appendEscaped(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    text: []const u8,
) !void {
    for (text) |char| {
        switch (char) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, char),
        }
    }
}

pub fn printRunHelp() void {
    std.debug.print(
        "Usage: abi llm run --model <id|path> --prompt <text> [options]\\n\\n" ++
            "Run one-shot LLM inference through the local-first provider router.\\n\\n" ++
            "Options:\\n" ++
            "  -m, --model <id|path>   Model id or local file path (.gguf, etc.)\\n" ++
            "  -p, --prompt <text>     Prompt text\\n" ++
            "  --backend <id>          Pin backend (local_gguf, llama_cpp, mlx, ollama, lm_studio, vllm, plugin_http, plugin_native)\\n" ++
            "  --fallback <csv>        Comma-separated fallback backend chain\\n" ++
            "  --strict-backend        Disable fallback when backend is unavailable\\n" ++
            "  --plugin <id>           Pin plugin id for plugin_http/plugin_native\\n" ++
            "  -n, --max-tokens <n>    Max tokens (default: 256)\\n" ++
            "  -t, --temperature <f>   Temperature (default: 0.7)\\n" ++
            "  --top-p <f>             Top-p (default: 0.9)\\n" ++
            "  --top-k <n>             Top-k (default: 40)\\n" ++
            "  --repeat-penalty <f>    Repetition penalty (default: 1.1)\\n" ++
            "  --json                  Print JSON output\\n\\n" ++
            "Examples:\\n" ++
            "  abi llm run --model ./models/qwen.gguf --prompt \"hello\"\\n" ++
            "  abi llm run --model llama3 --prompt \"summarize\" --backend ollama --strict-backend\\n" ++
            "  abi llm run --model phi4 --prompt \"test\" --fallback mlx,ollama\\n",
        .{},
    );
}

test "parseRunArgs parses strict backend and fallback" {
    const allocator = std.testing.allocator;
    const args = [_][:0]const u8{
        "--model",
        "llama3",
        "--prompt",
        "hello",
        "--backend",
        "llama_cpp",
        "--fallback",
        "mlx,ollama",
        "--strict-backend",
    };

    const parsed = try parseRunArgs(allocator, &args);
    defer allocator.free(parsed.fallback);

    try std.testing.expect(parsed.model != null);
    try std.testing.expect(parsed.prompt != null);
    try std.testing.expect(parsed.backend != null);
    try std.testing.expect(parsed.backend.? == .llama_cpp);
    try std.testing.expect(parsed.strict_backend);
    try std.testing.expectEqual(@as(usize, 2), parsed.fallback.len);
    try std.testing.expect(parsed.fallback[0] == .mlx);
    try std.testing.expect(parsed.fallback[1] == .ollama);
}
