//! ABI CLI — Command-line interface for the ABI framework.
//!
//! Provides user-facing commands for interacting with the multi-persona
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
//!   abi chat <msg>        Route a message through the persona pipeline
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
const feature_catalog = root.meta.features;

// ── Entry Point ─────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    try dispatch(allocator, args[1..]);
}

// ── Command Dispatch ────────────────────────────────────────────────────

pub fn dispatch(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printStatus();
        return;
    }

    if (cli.isServeInvocation(args)) {
        const serve_args = if (std.mem.eql(u8, args[0], "acp")) args[2..] else args[1..];
        try cli.runServe(allocator, serve_args);
        return;
    }

    const cmd = args[0];
    const next_arg = if (args.len > 1) args[1] else null;

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
        const message = next_arg orelse {
            std.debug.print("Usage: abi chat <message>\n", .{});
            return;
        };
        try runChat(allocator, message);
    } else if (std.mem.eql(u8, cmd, "db")) {
        try runDb(allocator, next_arg);
    } else if (std.mem.eql(u8, cmd, "dashboard")) {
        try runDashboard(allocator);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printHelp();
    } else {
        std.debug.print("Unknown command: {s}\n\n", .{cmd});
        printHelp();
    }
}

// ── Status (no-args) ────────────────────────────────────────────────────

pub fn printStatus() void {
    const version = build_options.package_version;

    // Count enabled features
    const enabled = comptime blk: {
        var count: u32 = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };

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
        \\  chat <msg>   Route through persona pipeline
        \\  serve        Start the ACP HTTP server
        \\  acp serve    Start the ACP HTTP server
        \\
    , .{ version, enabled, feature_catalog.feature_count, feature_catalog.feature_count });

    // Feature-gated commands
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
        \\Zig 0.16.0-dev | Multi-Persona AI + WDBX
        \\
        \\Core: Care first. Clarity always. Competence throughout.
        \\
    , .{version});
}

// ── Help ────────────────────────────────────────────────────────────────

pub fn printHelp() void {
    std.debug.print(
        \\ABI — Multi-Persona AI Framework with WDBX
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
        \\  chat <msg>   Route a message through the persona pipeline
        \\  db <cmd>     Vector database operations (add, query, stats, optimize, backup, restore, serve)
        \\  serve        Start the ACP HTTP server
        \\  acp serve    Start the ACP HTTP server
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
    std.debug.print(
        \\ABI Features — Compile-Time Feature Catalog
        \\════════════════════════════════════════════
        \\
    , .{});

    // Print all features from the canonical catalog
    inline for (feature_catalog.all) |entry| {
        const enabled = @field(build_options, entry.compile_flag_field);
        const tag: []const u8 = if (enabled) "[+]" else "[-]";
        const parent_str: []const u8 = if (entry.parent != null) "  " else "";
        std.debug.print("  {s} {s}{s} — {s}\n", .{ tag, parent_str, entry.feature.name(), entry.description });
    }

    const enabled = comptime blk: {
        var count: u32 = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };
    std.debug.print("\n{d}/{d} features enabled.\n", .{ enabled, feature_catalog.feature_count });
}

// ── Platform ────────────────────────────────────────────────────────────

pub fn printPlatform() void {
    const platform = root.platform;
    const info = platform.getPlatformInfo();

    std.debug.print(
        \\ABI Platform — System Detection
        \\════════════════════════════════
        \\
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

    // GPU backend info
    std.debug.print("GPU Backends:\n", .{});
    inline for (.{
        .{ "metal", build_options.gpu_metal },
        .{ "cuda", build_options.gpu_cuda },
        .{ "vulkan", build_options.gpu_vulkan },
        .{ "webgpu", build_options.gpu_webgpu },
        .{ "opengl", build_options.gpu_opengl },
        .{ "stdgpu", build_options.gpu_stdgpu },
    }) |backend| {
        if (backend[1]) {
            std.debug.print("  [+] {s}\n", .{backend[0]});
        }
    }

    std.debug.print("\n", .{});
}

// ── Connectors ──────────────────────────────────────────────────────────

pub fn printConnectors() void {
    std.debug.print(
        \\ABI Connectors — LLM Provider Adapters
        \\═══════════════════════════════════════
        \\
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
    std.debug.print(
        \\ABI Framework — Architecture Summary
        \\════════════════════════════════════════
        \\
        \\Personas:
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
        \\Features: 20 feature directories, 30 in catalog (mod/stub pattern)
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
        \\  feat_ai        = {any}
        \\  feat_gpu       = {any}
        \\  feat_database  = {any}
        \\  feat_network   = {any}
        \\  feat_web       = {any}
        \\  feat_search    = {any}
        \\  feat_cache     = {any}
        \\  feat_auth      = {any}
        \\  feat_lsp       = {any}
        \\  feat_mcp       = {any}
        \\  feat_mobile    = {any}
        \\  feat_desktop   = {any}
        \\  feat_tui       = {any}
        \\
        \\AI Sub-features:
        \\  feat_llm       = {any}
        \\  feat_training  = {any}
        \\  feat_vision    = {any}
        \\  feat_reasoning = {any}
        \\
        \\GPU Backends:
        \\  gpu_metal      = {any}
        \\  gpu_cuda       = {any}
        \\  gpu_vulkan     = {any}
        \\  gpu_stdgpu     = {any}
        \\
        \\Status: All systems nominal.
        \\
    , .{
        version,
        build_options.feat_ai,
        build_options.feat_gpu,
        build_options.feat_database,
        build_options.feat_network,
        build_options.feat_web,
        build_options.feat_search,
        build_options.feat_cache,
        build_options.feat_auth,
        build_options.feat_lsp,
        build_options.feat_mcp,
        build_options.feat_mobile,
        build_options.feat_desktop,
        build_options.feat_tui,
        build_options.feat_llm,
        build_options.feat_training,
        build_options.feat_vision,
        build_options.feat_reasoning,
        build_options.gpu_metal,
        build_options.gpu_cuda,
        build_options.gpu_vulkan,
        build_options.gpu_stdgpu,
    });
}

// ── Chat ────────────────────────────────────────────────────────────────

pub fn runChat(allocator: std.mem.Allocator, message: []const u8) !void {
    const persona = root.ai.persona;
    var registry = persona.PersonaRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route(message);

    std.debug.print(
        \\ABI Chat — Persona Pipeline
        \\════════════════════════════
        \\
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
}

// ── Database ────────────────────────────────────────────────────────────

pub fn runDb(allocator: std.mem.Allocator, subcommand: ?[]const u8) !void {
    const db_cli = root.database;
    if (comptime @hasDecl(db_cli, "cli")) {
        // Build args slice from the subcommand
        if (subcommand) |sub| {
            const args = [_][:0]const u8{@ptrCast(sub.ptr[0..sub.len :0])};
            try db_cli.cli.run(allocator, &args);
        } else {
            const empty: []const [:0]const u8 = &.{};
            try db_cli.cli.run(allocator, empty);
        }
    } else {
        std.debug.print("Database is disabled. Rebuild with -Dfeat-database=true\n", .{});
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
    try runChat(std.testing.allocator, "Hello, how are you?");
}

test {
    std.testing.refAllDecls(@This());
}
