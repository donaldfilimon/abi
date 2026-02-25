//! LLM serve subcommand - Start streaming inference HTTP server.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

/// Start the streaming inference HTTP server
///
/// Provides OpenAI-compatible endpoints for LLM inference:
/// - POST /v1/chat/completions (streaming with SSE)
/// - GET /health
/// - GET /v1/models
pub fn runServe(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printServeHelp();
        return;
    }

    // Parse arguments
    var model_path: ?[]const u8 = null;
    var address: []const u8 = "127.0.0.1:8080";
    var auth_token: ?[]const u8 = null;
    var preload: bool = false;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--model", "-m" })) {
            if (i < args.len) {
                model_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--address", "-a" })) {
            if (i < args.len) {
                address = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--auth-token")) {
            if (i < args.len) {
                auth_token = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--preload")) {
            preload = true;
            continue;
        }

        // First positional argument is model path
        if (model_path == null) {
            model_path = arg;
        }
    }

    // Create server configuration
    const server_config = abi.ai.streaming.ServerConfig{
        .address = address,
        .auth_token = auth_token,
        .default_model_path = model_path,
        .preload_model = preload,
        .default_backend = .local,
        .enable_openai_compat = true,
        .enable_websocket = true,
    };

    // Print startup banner
    utils.output.println("", .{});
    utils.output.println("\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97", .{});
    utils.output.println("\xe2\x95\x91          ABI Streaming Inference Server                   \xe2\x95\x91", .{});
    utils.output.println("\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d", .{});
    utils.output.println("", .{});

    if (model_path) |path| {
        utils.output.printKeyValue("Model", path);
        if (preload) {
            utils.output.printKeyValue("Mode", "Pre-loading model on startup...");
        } else {
            utils.output.printKeyValue("Mode", "Lazy loading (model loads on first request)");
        }
    } else {
        utils.output.printKeyValue("Model", "None configured");
        utils.output.printInfo("Use -m <path> to specify a GGUF model", .{});
    }

    utils.output.printKeyValueFmt("Address", "{s}", .{address});
    if (auth_token != null) {
        utils.output.printKeyValue("Auth", "Bearer token required");
    } else {
        utils.output.printKeyValue("Auth", "Disabled (open access)");
    }
    utils.output.println("", .{});

    // Initialize server
    var server = abi.ai.streaming.StreamingServer.init(allocator, server_config) catch |err| {
        utils.output.printError("Failed to initialize server: {t}", .{err});
        return err;
    };
    defer server.deinit();

    // Print endpoints
    utils.output.println("Endpoints:", .{});
    utils.output.println("  POST /v1/chat/completions  - OpenAI-compatible chat (stream=true for SSE)", .{});
    utils.output.println("  POST /api/stream           - ABI streaming endpoint", .{});
    utils.output.println("  GET  /api/stream/ws        - WebSocket streaming", .{});
    utils.output.println("  GET  /v1/models            - List available models", .{});
    utils.output.println("  GET  /health               - Health check", .{});
    utils.output.println("", .{});

    // Print usage example
    utils.output.println("Test with:", .{});
    if (auth_token) |token| {
        utils.output.println("  curl -X POST http://{s}/v1/chat/completions \\", .{address});
        utils.output.println("    -H \"Content-Type: application/json\" \\", .{});
        utils.output.println("    -H \"Authorization: Bearer {s}\" \\", .{token});
        utils.output.println("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'", .{});
    } else {
        utils.output.println("  curl -X POST http://{s}/v1/chat/completions \\", .{address});
        utils.output.println("    -H \"Content-Type: application/json\" \\", .{});
        utils.output.println("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'", .{});
    }
    utils.output.println("", .{});
    utils.output.println("Press Ctrl+C to stop the server.", .{});
    utils.output.printSeparator(64);

    // Start serving (blocking)
    server.serve() catch |err| {
        utils.output.printError("Server error: {t}", .{err});
        return err;
    };
}

pub fn printServeHelp() void {
    const help_text =
        "Usage: abi llm serve [options]\n\n" ++
        "Start an HTTP server for streaming LLM inference.\n\n" ++
        "The server provides OpenAI-compatible endpoints, allowing you to use\n" ++
        "standard OpenAI SDKs and clients with local GGUF models.\n\n" ++
        "Options:\n" ++
        "  -m, --model <path>      Path to GGUF model file\n" ++
        "  -a, --address <addr>    Listen address (default: 127.0.0.1:8080)\n" ++
        "  --auth-token <token>    Bearer token for authentication (optional)\n" ++
        "  --preload               Pre-load model on startup\n" ++
        "  -h, --help              Show this help message\n\n" ++
        "Endpoints:\n" ++
        "  POST /v1/chat/completions  OpenAI-compatible chat completions\n" ++
        "  POST /api/stream           Custom ABI streaming endpoint\n" ++
        "  GET  /api/stream/ws        WebSocket upgrade for streaming\n" ++
        "  GET  /v1/models            List available models\n" ++
        "  GET  /health               Health check (no auth required)\n\n" ++
        "Examples:\n" ++
        "  abi llm serve -m ./llama-7b.gguf\n" ++
        "  abi llm serve -m ./model.gguf -a 0.0.0.0:8000 --preload\n" ++
        "  abi llm serve -m ./model.gguf --auth-token my-secret-token\n\n" ++
        "Testing:\n" ++
        "  # Health check\n" ++
        "  curl http://127.0.0.1:8080/health\n\n" ++
        "  # Streaming chat completion\n" ++
        "  curl -N http://127.0.0.1:8080/v1/chat/completions \\\n" ++
        "    -H \"Content-Type: application/json\" \\\n" ++
        "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}'\n";
    utils.output.print("{s}", .{help_text});
}

test {
    std.testing.refAllDecls(@This());
}
