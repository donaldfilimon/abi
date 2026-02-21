//! LLM serve subcommand - Start streaming inference HTTP server.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");

/// Start the streaming inference HTTP server
///
/// Provides OpenAI-compatible endpoints for LLM inference:
/// - POST /v1/chat/completions (streaming with SSE)
/// - GET /health
/// - GET /v1/models
pub fn runServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
    std.debug.print("\n", .{});
    std.debug.print("\xe2\x95\x94\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x97\n", .{});
    std.debug.print("\xe2\x95\x91          ABI Streaming Inference Server                   \xe2\x95\x91\n", .{});
    std.debug.print("\xe2\x95\x9a\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x90\xe2\x95\x9d\n", .{});
    std.debug.print("\n", .{});

    if (model_path) |path| {
        std.debug.print("  Model: {s}\n", .{path});
        if (preload) {
            std.debug.print("  Mode:  Pre-loading model on startup...\n", .{});
        } else {
            std.debug.print("  Mode:  Lazy loading (model loads on first request)\n", .{});
        }
    } else {
        std.debug.print("  Model: None configured\n", .{});
        std.debug.print("  Note:  Use -m <path> to specify a GGUF model\n", .{});
    }

    std.debug.print("  Address: {s}\n", .{address});
    if (auth_token != null) {
        std.debug.print("  Auth: Bearer token required\n", .{});
    } else {
        std.debug.print("  Auth: Disabled (open access)\n", .{});
    }
    std.debug.print("\n", .{});

    // Initialize server
    var server = abi.ai.streaming.StreamingServer.init(allocator, server_config) catch |err| {
        std.debug.print("Failed to initialize server: {t}\n", .{err});
        return err;
    };
    defer server.deinit();

    // Print endpoints
    std.debug.print("Endpoints:\n", .{});
    std.debug.print("  POST /v1/chat/completions  - OpenAI-compatible chat (stream=true for SSE)\n", .{});
    std.debug.print("  POST /api/stream           - ABI streaming endpoint\n", .{});
    std.debug.print("  GET  /api/stream/ws        - WebSocket streaming\n", .{});
    std.debug.print("  GET  /v1/models            - List available models\n", .{});
    std.debug.print("  GET  /health               - Health check\n", .{});
    std.debug.print("\n", .{});

    // Print usage example
    std.debug.print("Test with:\n", .{});
    if (auth_token) |token| {
        std.debug.print("  curl -X POST http://{s}/v1/chat/completions \\\n", .{address});
        std.debug.print("    -H \"Content-Type: application/json\" \\\n", .{});
        std.debug.print("    -H \"Authorization: Bearer {s}\" \\\n", .{token});
        std.debug.print("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'\n", .{});
    } else {
        std.debug.print("  curl -X POST http://{s}/v1/chat/completions \\\n", .{address});
        std.debug.print("    -H \"Content-Type: application/json\" \\\n", .{});
        std.debug.print("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'\n", .{});
    }
    std.debug.print("\n", .{});
    std.debug.print("Press Ctrl+C to stop the server.\n", .{});
    std.debug.print("\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\n", .{});

    // Start serving (blocking)
    server.serve() catch |err| {
        std.debug.print("Server error: {t}\n", .{err});
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
    std.debug.print("{s}", .{help_text});
}
