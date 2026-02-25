//! LLM generate subcommand - Generate text from a prompt.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const mod = @import("mod.zig");
const info = @import("info.zig");

pub fn runGenerate(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 256;
    var temperature: f32 = 0.7;
    var top_p: f32 = 0.9;
    var top_k: u32 = 40;
    // New generation options
    var repeat_penalty: f32 = 1.1;
    var seed: ?u64 = null;
    var stream: bool = false;
    var allow_ollama_fallback: bool = true;
    var ollama_model: ?[]const u8 = null;
    var stop_sequences = std.ArrayListUnmanaged([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    // Advanced sampling options (llama.cpp parity)
    var tfs_z: f32 = 1.0; // tail-free sampling (1.0 = disabled)
    var mirostat: u8 = 0; // 0 = disabled, 1 = v1, 2 = v2
    var mirostat_tau: f32 = 5.0;
    var mirostat_eta: f32 = 0.1;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--model", "-m" })) {
            if (i < args.len) {
                model_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--prompt", "-p" })) {
            if (i < args.len) {
                prompt = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--max-tokens", "-n" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_tokens = std.fmt.parseInt(u32, val, 10) catch 256;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--temperature", "-t" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                temperature = std.fmt.parseFloat(f32, val) catch 0.7;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--top-p")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                top_p = std.fmt.parseFloat(f32, val) catch 0.9;
                i += 1;
            }
            continue;
        }

        // New options
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--repeat-penalty")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                repeat_penalty = std.fmt.parseFloat(f32, val) catch 1.1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--seed")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                seed = std.fmt.parseInt(u64, val, 10) catch null;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--stop")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                try stop_sequences.append(allocator, val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--stream")) {
            stream = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-ollama-fallback")) {
            allow_ollama_fallback = false;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--ollama-model")) {
            if (i < args.len) {
                ollama_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--top-k")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                top_k = std.fmt.parseInt(u32, val, 10) catch 40;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--tfs")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                tfs_z = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat = std.fmt.parseInt(u8, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat-tau")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat_tau = std.fmt.parseFloat(f32, val) catch 5.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat-eta")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat_eta = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        // Positional: model path or prompt
        if (model_path == null) {
            model_path = std.mem.sliceTo(arg, 0);
        } else if (prompt == null) {
            prompt = std.mem.sliceTo(arg, 0);
        }
    }

    if (model_path == null) {
        utils.output.printError("Model path required", .{});
        utils.output.println("Usage: abi llm generate <model> --prompt <text>", .{});
        return;
    }

    if (prompt == null) {
        utils.output.printError("Prompt required", .{});
        utils.output.println("Usage: abi llm generate <model> --prompt <text>", .{});
        return;
    }

    utils.output.printKeyValueFmt("Loading model", "{s}", .{model_path.?});
    utils.output.printKeyValueFmt("Prompt", "{s}", .{prompt.?});
    utils.output.println("Max tokens: {d}, Temperature: {d:.2}, Top-p: {d:.2}, Top-k: {d}, Repeat penalty: {d:.2}", .{ max_tokens, temperature, top_p, top_k, repeat_penalty });
    if (seed) |s| {
        utils.output.printKeyValueFmt("Seed", "{d}", .{s});
    }
    if (tfs_z < 1.0) {
        utils.output.println("Tail-free sampling: z={d:.2}", .{tfs_z});
    }
    if (mirostat > 0) {
        utils.output.println("Mirostat v{d}: tau={d:.2}, eta={d:.2}", .{ mirostat, mirostat_tau, mirostat_eta });
    }
    if (stream) {
        utils.output.println("Streaming: enabled", .{});
    }
    if (!allow_ollama_fallback) {
        utils.output.println("Ollama fallback: disabled", .{});
    } else if (ollama_model) |name| {
        utils.output.printKeyValueFmt("Ollama model override", "{s}", .{name});
    }
    utils.output.println("", .{});
    utils.output.println("Generating...", .{});
    utils.output.println("", .{});

    // Create inference engine
    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repetition_penalty = repeat_penalty,
        .allow_ollama_fallback = allow_ollama_fallback,
        .ollama_model = ollama_model,
    });
    defer engine.deinit();

    // Load model
    engine.loadModel(model_path.?) catch |err| {
        utils.output.printError("loading model: {t}", .{err});
        if (err == error.FileTooLarge) {
            info.printModelFileSizeHint(allocator, model_path.?);
            return;
        }
        if (err == error.UnsupportedArchitecture) {
            utils.output.println("", .{});
            utils.output.printWarning("This GGUF architecture is not yet supported by ABI local inference.", .{});
            utils.output.println("Current local engine targets LLaMA-compatible transformer layouts.", .{});
            utils.output.printInfo("Tip: remove `--no-ollama-fallback` to run this model via Ollama.", .{});
            info.printUnsupportedLayoutSummary(allocator, model_path.?);
            return;
        }
        utils.output.println("", .{});
        utils.output.printInfo("GGUF model loading requires a valid GGUF file.", .{});
        utils.output.println("You can download models from: https://huggingface.co/TheBloke", .{});
        return;
    };

    const backend = engine.getBackend();
    utils.output.printKeyValueFmt("Backend", "{s}", .{backend.label()});
    if (backend == .ollama) {
        if (engine.getBackendModelName()) |name| {
            utils.output.printKeyValueFmt("Ollama model", "{s}", .{name});
        }
    }
    utils.output.println("", .{});

    // Generate
    const gen_output = engine.generate(allocator, prompt.?) catch |err| {
        utils.output.printError("during generation: {t}", .{err});
        return;
    };
    defer allocator.free(gen_output);

    utils.output.println("{s}", .{gen_output});

    // Print stats
    const stats = engine.getStats();
    utils.output.println("", .{});
    utils.output.printSeparator(3);
    utils.output.println("Stats: {d:.1} tok/s prefill, {d:.1} tok/s decode", .{
        stats.prefillTokensPerSecond(),
        stats.decodeTokensPerSecond(),
    });
}
