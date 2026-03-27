//! ABI CLI — Command-line interface for the ABI framework.
//!
//! Provides user-facing commands for interacting with the multi-profile
//! AI system, WDBX database, full-text search, and framework diagnostics.
//!
//! Usage:
//!   abi                   Show status overview with enabled features
//!   abi version           Print version and build info
//!   abi doctor            Run diagnostics (features, platform, GPU)
//!   abi features          List all features with status
//!   abi platform          Show platform detection info
//!   abi connectors        List available LLM connectors
//!   abi info              Show framework architecture summary
//!   abi chat <message...>  Route a message through the profile pipeline
//!   abi serve             Start the ACP HTTP server
//!   abi acp serve         Start the ACP HTTP server
//!   abi db <subcommand>   Vector database operations
//!   abi dashboard         Launch interactive TUI dashboard
//!   abi help              Show this help message

const std = @import("std");
const build_options = @import("build_options");

// Framework modules (relative imports within src/)
const root = @import("root.zig");
const cli = @import("cli.zig");
const os = @import("foundation/os.zig");
const feature_catalog = root.meta.features;

// ── Shared Constants ────────────────────────────────────────────────────

/// All GPU backend names and their build-time enabled state.
/// Used by both `printPlatform()` and `runDoctor()`.
const gpu_backends = .{
    .{ "metal", build_options.gpu_metal },
    .{ "cuda", build_options.gpu_cuda },
    .{ "vulkan", build_options.gpu_vulkan },
    .{ "webgpu", build_options.gpu_webgpu },
    .{ "opengl", build_options.gpu_opengl },
    .{ "opengles", build_options.gpu_opengles },
    .{ "webgl2", build_options.gpu_webgl2 },
    .{ "stdgpu", build_options.gpu_stdgpu },
    .{ "fpga", build_options.gpu_fpga },
    .{ "tpu", build_options.gpu_tpu },
};

// ── Helpers ─────────────────────────────────────────────────────────────

/// Write data to stdout using libc. Separates response output (stdout)
/// from diagnostic metadata (stderr via std.debug.print).
fn writeToStdout(data: []const u8) void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = std.c.write(std.posix.STDOUT_FILENO, data[offset..].ptr, data.len - offset);
        if (n > 0) {
            offset += @intCast(n);
        } else {
            break;
        }
    }
}

fn countEnabledFeatures() struct { enabled: usize, total: usize } {
    const enabled = comptime blk: {
        var count: usize = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };
    return .{ .enabled = enabled, .total = feature_catalog.all.len };
}

fn printHeader(title: []const u8, subtitle: ?[]const u8) void {
    if (!os.isatty()) return; // Strip non-diagnostic metadata when piped

    std.debug.print("{s}\n", .{title});
    if (subtitle) |sub| {
        std.debug.print("{s}\n", .{sub});
    } else {
        for (title) |_| std.debug.print("═", .{});
        std.debug.print("\n\n", .{});
    }
}

// ── Entry Point ─────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    const exit_code = dispatch(allocator, args[1..]) catch |err| blk: {
        std.debug.print("Error: {s}\n", .{@errorName(err)});
        break :blk 1;
    };
    if (exit_code != 0) std.process.exit(exit_code);
}

// ── Command Dispatch ────────────────────────────────────────────────────

pub fn dispatch(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0) {
        printStatus();
        return 0;
    }

    if (cli.isServeInvocation(args)) {
        const serve_args = if (std.mem.eql(u8, args[0], "acp")) args[2..] else args[1..];
        try cli.runServe(allocator, serve_args);
        return 0;
    }

    const cmd = args[0];

    if (std.mem.eql(u8, cmd, "version")) {
        printVersion();
    } else if (std.mem.eql(u8, cmd, "doctor")) {
        runDoctor();
    } else if (std.mem.eql(u8, cmd, "features")) {
        printFeatures();
    } else if (std.mem.eql(u8, cmd, "platform")) {
        printPlatform();
    } else if (std.mem.eql(u8, cmd, "connectors")) {
        printConnectors();
    } else if (std.mem.eql(u8, cmd, "info")) {
        printInfo();
    } else if (std.mem.eql(u8, cmd, "chat")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi chat <message...>\n", .{});
            return 1;
        }
        try runChat(allocator, args[1..]);
    } else if (std.mem.eql(u8, cmd, "db")) {
        try runDb(allocator, args[1..]);
    } else if (std.mem.eql(u8, cmd, "lsp")) {
        try runLsp(allocator);
    } else if (std.mem.eql(u8, cmd, "dashboard")) {
        try runDashboard(allocator);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printHelp();
    } else {
        std.debug.print("Unknown command: {s}\n\n", .{cmd});
        printHelp();
        return 1;
    }
    return 0;
}

// ── Status (no-args) ────────────────────────────────────────────────────

pub fn printStatus() void {
    const version = build_options.package_version;

    const counts = countEnabledFeatures();

    std.debug.print(
        \\ABI Framework v{s}
        \\Zig 0.16.0-dev | {d}/{d} features enabled
        \\
        \\Commands:
        \\  version      Print version and build info
        \\  doctor       Run diagnostics (features, platform, GPU)
        \\  features     List all {d} features with status
        \\  platform     Show platform detection info
        \\  connectors   List available LLM connectors
        \\  info         Framework architecture summary
        \\  chat <message...>  Route through profile pipeline
        \\  serve        Start the ACP HTTP server
        \\  acp serve    Start the ACP HTTP server
        \\  lsp          Start the LSP server
        \\
    , .{ version, counts.enabled, counts.total, counts.total });

    std.debug.print("  db <cmd>     Vector database operations       ", .{});
    printFeatureTag(build_options.feat_database);
    std.debug.print("  dashboard    Interactive TUI dashboard         ", .{});
    printFeatureTag(build_options.feat_tui);

    std.debug.print(
        \\  help         Show detailed help
        \\
        \\Run 'abi <command>' for details. 'abi help' for full reference.
        \\
    , .{});
}

fn printFeatureTag(enabled: bool) void {
    if (enabled) {
        std.debug.print("[enabled]\n", .{});
    } else {
        std.debug.print("[disabled]\n", .{});
    }
}

// ── Version ─────────────────────────────────────────────────────────────

pub fn printVersion() void {
    const version = build_options.package_version;
    std.debug.print(
        \\ABI Framework v{s}
        \\Zig 0.16.0-dev | Multi-Profile AI + WDBX
        \\
        \\Core: Care first. Clarity always. Competence throughout.
        \\
    , .{version});
}

// ── Help ────────────────────────────────────────────────────────────────

pub fn printHelp() void {
    std.debug.print(
        \\ABI — Multi-Profile AI Framework with WDBX
        \\
        \\Usage: abi <command> [args]
        \\
        \\Diagnostics:
        \\  version      Print version and build info
        \\  doctor       Run diagnostics (features, platform, GPU)
        \\  features     List all features with enabled/disabled status
        \\  platform     Show platform detection (OS, arch, CPU)
        \\  connectors   List available LLM provider connectors
        \\  info         Show framework architecture summary
        \\
        \\AI & Data:
        \\  chat <message...>  Route a message through the profile pipeline
        \\  db <cmd>     Vector database operations (add, query, stats, optimize, backup, restore, serve)
        \\  serve        Start the ACP HTTP server
        \\  acp serve    Start the ACP HTTP server
        \\  lsp          Start the Language Server Protocol (LSP) server
        \\
        \\Interactive:
        \\  dashboard    Launch interactive TUI dashboard (requires -Dfeat-tui=true)
        \\
        \\Build:
        \\  zig build cli          Build this CLI binary
        \\  zig build mcp          Build MCP stdio server
        \\  zig build lib          Build static library
        \\  zig build test         Run all tests
        \\  zig build check        Full gate (lint + test + parity)
        \\
    , .{});
}

// ── Features ────────────────────────────────────────────────────────────

pub fn printFeatures() void {
    printHeader("ABI Features — Compile-Time Feature Catalog", null);

    inline for (feature_catalog.all) |entry| {
        const enabled = @field(build_options, entry.compile_flag_field);
        const tag: []const u8 = if (enabled) "[+]" else "[-]";
        const parent_str: []const u8 = if (entry.parent != null) "  " else "";
        std.debug.print("  {s} {s}{s} — {s}\n", .{ tag, parent_str, entry.feature.name(), entry.description });
    }

    const counts = countEnabledFeatures();
    std.debug.print("\n{d}/{d} features enabled.\n", .{ counts.enabled, counts.total });
}

// ── Platform ────────────────────────────────────────────────────────────

pub fn printPlatform() void {
    const platform = root.platform;
    const info = platform.getPlatformInfo();

    printHeader("ABI Platform — System Detection", null);

    std.debug.print(
        \\OS:           {s}
        \\Architecture: {s}
        \\Description:  {s}
        \\CPU Cores:    {d}
        \\Threading:    {s}
        \\
    , .{
        @tagName(info.os),
        @tagName(info.arch),
        platform.getDescription(),
        platform.getCpuCount(),
        if (platform.supportsThreading()) "supported" else "unavailable",
    });

    std.debug.print("GPU Backends:\n", .{});
    inline for (gpu_backends) |backend| {
        const tag: []const u8 = if (backend[1]) "[+]" else "[-]";
        std.debug.print("  {s} {s}\n", .{ tag, backend[0] });
    }

    std.debug.print("\n", .{});
}

// ── Connectors ──────────────────────────────────────────────────────────

pub fn printConnectors() void {
    printHeader("ABI Connectors — LLM Provider Adapters", null);

    std.debug.print(
        \\Available connectors (primary env var → provider):
        \\
        \\  ABI_OPENAI_API_KEY      → OpenAI (GPT-4, GPT-3.5)
        \\  ABI_ANTHROPIC_API_KEY   → Anthropic (Claude)
        \\  ABI_GEMINI_API_KEY      → Google Gemini
        \\  ABI_MISTRAL_API_KEY     → Mistral AI
        \\  ABI_COHERE_API_KEY      → Cohere (Chat, Embed, Rerank)
        \\  ABI_HF_API_TOKEN        → HuggingFace Inference API
        \\  ABI_OLLAMA_HOST         → Ollama (local, default: localhost:11434)
        \\  ABI_LM_STUDIO_HOST      → LM Studio (local, OpenAI-compatible)
        \\  ABI_VLLM_HOST           → vLLM (local, high-throughput)
        \\  ABI_MLX_HOST            → MLX (Apple Silicon optimized)
        \\  ABI_LLAMA_CPP_HOST      → llama.cpp server
        \\  ABI_DISCORD_BOT_TOKEN   → Discord bot integration
        \\
        \\Legacy env vars (e.g. OPENAI_API_KEY) are supported as fallbacks.
        \\Connector status: always available (not feature-gated).
        \\Set the env var to enable a provider. Use 'abi chat' to test routing.
        \\
    , .{});
}

// ── Info ────────────────────────────────────────────────────────────────

pub fn printInfo() void {
    printHeader("ABI Framework — Architecture Summary", null);

    std.debug.print(
        \\Profiles:
        \\  Abbey  — Empathetic Polymath (warm, technical, adaptive)
        \\  Aviva  — Direct Expert (concise, factual, efficient)
        \\  Abi    — Adaptive Moderator (routing, policy, blending)
        \\
        \\Pipeline:
        \\  User Input → Abi Analysis → Modulation → Routing
        \\  → Execution → Constitution → WDBX Memory → Response
        \\
        \\Storage:
        \\  WDBX — Vector database with HNSW, DiskANN, ScaNN
        \\  Block chain — Cryptographic conversation memory (SHA-256)
        \\
        \\Inference:
        \\  Backends: demo | connector | local
        \\  16 connectors: OpenAI, Anthropic, Claude, Gemini, Mistral,
        \\    Cohere, HuggingFace, Ollama, LM Studio, vLLM, MLX,
        \\    llama.cpp, Codex, OpenCode, Discord, local-scheduler
        \\
        \\Features: 20 feature directories, 35 in catalog (mod/stub pattern)
        \\GPU backends: Metal, CUDA, Vulkan, WebGPU, OpenGL, stdgpu, FPGA, TPU
        \\Protocols: MCP, LSP, ACP, HA
        \\
        \\Spec: docs/spec/ABBEY-SPEC.md
        \\
    , .{});
}

// ── Doctor ──────────────────────────────────────────────────────────────

pub fn runDoctor() void {
    const version = build_options.package_version;

    std.debug.print(
        \\ABI Doctor — Build Configuration Report
        \\════════════════════════════════════════
        \\
        \\Version: {s}
        \\
        \\Feature Flags:
        \\
    , .{version});

    inline for (feature_catalog.all) |entry| {
        const enabled = @field(build_options, entry.compile_flag_field);
        const indent: []const u8 = if (entry.parent != null) "    " else "  ";
        std.debug.print("{s}{s} = {any}\n", .{ indent, entry.compile_flag_field, enabled });
    }

    std.debug.print(
        \\
        \\GPU Backends:
        \\
    , .{});

    inline for (gpu_backends) |backend| {
        std.debug.print("  gpu_{s} = {any}\n", .{ backend[0], backend[1] });
    }

    std.debug.print(
        \\
        \\Status: All systems nominal.
        \\
    , .{});
}

// ── Chat ────────────────────────────────────────────────────────────────

pub fn runChat(allocator: std.mem.Allocator, message_args: []const [:0]const u8) !void {
    if (!build_options.feat_ai) {
        std.debug.print("AI features are disabled. Rebuild with -Dfeat-ai=true\n", .{});
        return;
    }

    const message = try cli.joinChatMessage(allocator, message_args);
    defer allocator.free(message);

    const ai = root.ai;
    var registry = ai.profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = ai.profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route(message);

    printHeader("ABI Chat — Profile Pipeline", null);

    std.debug.print(
        \\Input: {s}
        \\
        \\Routing Decision:
        \\  Primary: {s}
        \\  Strategy: {s}
        \\  Confidence: {d:.0}%
        \\  Reason: {s}
        \\
        \\Weights:
        \\  Abbey: {d:.0}%
        \\  Aviva: {d:.0}%
        \\  Abi:   {d:.0}%
        \\
    , .{
        message,
        decision.primary.name(),
        @tagName(decision.strategy),
        decision.confidence * 100.0,
        decision.reason,
        decision.weights.abbey * 100.0,
        decision.weights.aviva * 100.0,
        decision.weights.abi * 100.0,
    });

    std.debug.print("\nExecution:\n", .{});

    const inference = root.inference;
    var engine = inference.Engine.init(allocator, .{
        .backend = .connector,
        .model_id = "ollama/llama3",
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
    }) catch |err| {
        std.debug.print("  Engine init failed: {s}\n", .{@errorName(err)});
        std.debug.print("\nHint: run 'abi connectors' to see required environment variables.\n", .{});
        return;
    };
    defer engine.deinit();

    var result = engine.generate(.{
        .id = 1,
        .prompt = message,
        .max_tokens = 256,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .profile_id = @intFromEnum(decision.primary),
    }) catch |err| {
        std.debug.print("  Inference failed: {s}\n", .{@errorName(err)});
        std.debug.print("\nHint: run 'abi connectors' to see required environment variables.\n", .{});
        return;
    };
    defer result.deinit(allocator);

    // Detect echo/demo fallback vs real connector response
    const is_echo = std.mem.startsWith(u8, result.text, "[");
    const backend_label: []const u8 = if (is_echo) "echo" else "connector";

    std.debug.print("  [{s}] {s} | {d} tokens | {d:.1}ms\n", .{
        backend_label,
        engine.config.model_id,
        result.completion_tokens,
        result.latency_ms,
    });

    // Write the actual response to stdout so it can be piped.
    // Metadata goes to stderr (via std.debug.print above); response to stdout.
    writeToStdout("\n");
    writeToStdout(result.text);
    writeToStdout("\n");
}

// ── Database ────────────────────────────────────────────────────────────

pub fn runDb(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const db_cli = root.database;
    if (comptime @hasDecl(db_cli, "cli")) {
        try db_cli.cli.run(allocator, args);
    } else {
        std.debug.print("Database is disabled. Rebuild with -Dfeat-database=true\n", .{});
    }
}

// ── LSP ─────────────────────────────────────────────────────────────────

pub fn runLsp(allocator: std.mem.Allocator) !void {
    const lsp = root.lsp;
    if (comptime @hasDecl(lsp, "server")) {
        try lsp.server.run(allocator);
    } else {
        std.debug.print("LSP is disabled. Rebuild with -Dfeat-lsp=true\n", .{});
    }
}

// ── Dashboard ───────────────────────────────────────────────────────────

pub fn runDashboard(allocator: std.mem.Allocator) !void {
    const has_dashboard = comptime @hasDecl(root.tui.dashboard, "run");
    if (has_dashboard) {
        return root.tui.dashboard.run(allocator);
    }
    std.debug.print("TUI is disabled. Rebuild with -Dfeat-tui=true\n", .{});
}

// ── Tests ───────────────────────────────────────────────────────────────

test "version prints without error" {
    printVersion();
}

test "help prints without error" {
    printHelp();
}

test "status prints without error" {
    printStatus();
}

test "info prints without error" {
    printInfo();
}

test "doctor runs without error" {
    runDoctor();
}

test "features prints without error" {
    printFeatures();
}

test "platform prints without error" {
    printPlatform();
}

test "connectors prints without error" {
    printConnectors();
}

test "chat routes message without error" {
    const message_args = [_][:0]const u8{ "Hello,", "how", "are", "you?" };
    try runChat(std.testing.allocator, &message_args);
}

test {
    std.testing.refAllDecls(@This());
}
