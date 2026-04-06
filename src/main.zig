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
//!   abi search <cmd>      Full-text search (create, index, query, delete, stats)
//!   abi dashboard         Launch developer diagnostics shell
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

fn printSharedCliText(comptime renderFn: anytype) !void {
    var writer: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer writer.deinit();
    renderFn(&writer.writer) catch return;
    const output = writer.toOwnedSlice() catch return;
    defer std.heap.page_allocator.free(output);
    writeToStdout(output);
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
    } else if (std.mem.eql(u8, cmd, "search")) {
        try runSearch(allocator, args[1..]);
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
    _ = printSharedCliText(cli.writeStatus);
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
    _ = printSharedCliText(cli.writeVersion);
}

// ── Help ────────────────────────────────────────────────────────────────

pub fn printHelp() void {
    _ = printSharedCliText(cli.writeHelp);
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

    const connectors = [_]struct { env: [:0]const u8, fallback: ?[:0]const u8, name: []const u8 }{
        .{ .env = "ABI_OPENAI_API_KEY", .fallback = "OPENAI_API_KEY", .name = "OpenAI (GPT-4, GPT-3.5)" },
        .{ .env = "ABI_ANTHROPIC_API_KEY", .fallback = "ANTHROPIC_API_KEY", .name = "Anthropic (Claude)" },
        .{ .env = "ABI_GEMINI_API_KEY", .fallback = "GEMINI_API_KEY", .name = "Google Gemini" },
        .{ .env = "ABI_MISTRAL_API_KEY", .fallback = "MISTRAL_API_KEY", .name = "Mistral AI" },
        .{ .env = "ABI_COHERE_API_KEY", .fallback = "COHERE_API_KEY", .name = "Cohere (Chat, Embed, Rerank)" },
        .{ .env = "ABI_HF_API_TOKEN", .fallback = "HF_API_TOKEN", .name = "HuggingFace Inference API" },
        .{ .env = "ABI_OLLAMA_HOST", .fallback = "OLLAMA_HOST", .name = "Ollama (local)" },
        .{ .env = "ABI_LM_STUDIO_HOST", .fallback = null, .name = "LM Studio (local, OpenAI-compat)" },
        .{ .env = "ABI_VLLM_HOST", .fallback = null, .name = "vLLM (local, high-throughput)" },
        .{ .env = "ABI_MLX_HOST", .fallback = null, .name = "MLX (Apple Silicon)" },
        .{ .env = "ABI_LLAMA_CPP_HOST", .fallback = null, .name = "llama.cpp server" },
        .{ .env = "ABI_DISCORD_BOT_TOKEN", .fallback = "DISCORD_BOT_TOKEN", .name = "Discord bot integration" },
    };

    var configured: u32 = 0;
    for (connectors) |c| {
        const has_primary = std.c.getenv(c.env.ptr) != null;
        const has_fallback = if (c.fallback) |fb| std.c.getenv(fb.ptr) != null else false;
        const is_set = has_primary or has_fallback;
        if (is_set) configured += 1;
        const tag = if (is_set) "\x1b[32m[configured]\x1b[0m" else "\x1b[90m[not set]\x1b[0m";
        std.debug.print("  {s: <24} {s} {s}\n", .{ c.env, tag, c.name });
    }

    std.debug.print("\n  {d}/12 providers configured", .{configured});
    if (configured == 0) {
        std.debug.print(" — set env vars to enable providers", .{});
    }
    std.debug.print("\n  Legacy env vars (e.g. OPENAI_API_KEY) are supported as fallbacks.\n", .{});
    std.debug.print("  Use 'abi chat' to test routing.\n\n", .{});
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
        \\Features: 20 feature directories, {d} in catalog (mod/stub pattern)
        \\GPU backends: Metal, CUDA, Vulkan, WebGPU, OpenGL, stdgpu, FPGA, TPU
        \\Protocols: MCP, LSP, ACP, HA
        \\
        \\Spec: docs/spec/ABBEY-SPEC.md
        \\
    , .{feature_catalog.feature_count});
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

// ── Search ──────────────────────────────────────────────────────────────

pub fn runSearch(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const search = root.search;
    if (!comptime build_options.feat_search) {
        std.debug.print("Search is disabled. Rebuild with -Dfeat-search=true\n", .{});
        return;
    }

    search.init(allocator, .{}) catch |err| {
        std.debug.print("Failed to initialize search: {s}\n", .{@errorName(err)});
        return;
    };
    defer search.deinit();

    if (args.len == 0) {
        printSearchHelp();
        return;
    }

    const subcmd = args[0];
    if (std.mem.eql(u8, subcmd, "create")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi search create <index_name>\n", .{});
            return;
        }
        _ = search.createIndex(allocator, args[1]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("Index '{s}' created.\n", .{args[1]});
    } else if (std.mem.eql(u8, subcmd, "index")) {
        if (args.len < 4) {
            std.debug.print("Usage: abi search index <index_name> <doc_id> <content...>\n", .{});
            return;
        }
        // Join remaining args as content
        var content_buf: [4096]u8 = undefined;
        var content_len: usize = 0;
        for (args[3..]) |arg| {
            if (content_len > 0 and content_len < content_buf.len) {
                content_buf[content_len] = ' ';
                content_len += 1;
            }
            const copy_len = @min(arg.len, content_buf.len - content_len);
            @memcpy(content_buf[content_len..][0..copy_len], arg[0..copy_len]);
            content_len += copy_len;
        }
        if (content_len >= content_buf.len) {
            std.debug.print("Warning: content truncated to {d} bytes\n", .{content_buf.len});
        }
        search.indexDocument(args[1], args[2], content_buf[0..content_len]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("Document '{s}' indexed in '{s}'.\n", .{ args[2], args[1] });
    } else if (std.mem.eql(u8, subcmd, "query")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi search query <index_name> <query_text...>\n", .{});
            return;
        }
        var query_buf: [2048]u8 = undefined;
        var query_len: usize = 0;
        for (args[2..]) |arg| {
            if (query_len > 0 and query_len < query_buf.len) {
                query_buf[query_len] = ' ';
                query_len += 1;
            }
            const copy_len = @min(arg.len, query_buf.len - query_len);
            @memcpy(query_buf[query_len..][0..copy_len], arg[0..copy_len]);
            query_len += copy_len;
        }
        if (query_len >= query_buf.len) {
            std.debug.print("Warning: query truncated to {d} bytes\n", .{query_buf.len});
        }
        const results = search.query(allocator, args[1], query_buf[0..query_len]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        defer allocator.free(results);
        if (results.len == 0) {
            std.debug.print("No results found.\n", .{});
        } else {
            for (results, 0..) |result, i| {
                std.debug.print("{d}. [{d:.3}] {s}", .{ i + 1, result.score, result.doc_id });
                if (result.snippet.len > 0) {
                    std.debug.print(" — {s}", .{result.snippet});
                }
                std.debug.print("\n", .{});
            }
        }
    } else if (std.mem.eql(u8, subcmd, "delete")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi search delete <index_name> [doc_id]\n", .{});
            return;
        }
        if (args.len >= 3) {
            const removed = search.deleteDocument(args[1], args[2]) catch |err| {
                std.debug.print("Error: {s}\n", .{@errorName(err)});
                return;
            };
            if (removed) {
                std.debug.print("Document '{s}' deleted from '{s}'.\n", .{ args[2], args[1] });
            } else {
                std.debug.print("Document '{s}' not found in '{s}'.\n", .{ args[2], args[1] });
            }
        } else {
            search.deleteIndex(args[1]) catch |err| {
                std.debug.print("Error: {s}\n", .{@errorName(err)});
                return;
            };
            std.debug.print("Index '{s}' deleted.\n", .{args[1]});
        }
    } else if (std.mem.eql(u8, subcmd, "stats")) {
        const s = search.stats();
        std.debug.print("Search Statistics:\n  Indexes: {d}\n  Documents: {d}\n  Terms: {d}\n", .{
            s.total_indexes, s.total_documents, s.total_terms,
        });
    } else if (std.mem.eql(u8, subcmd, "help")) {
        printSearchHelp();
    } else {
        std.debug.print("Unknown search command: {s}\n", .{subcmd});
        printSearchHelp();
    }
}

fn printSearchHelp() void {
    std.debug.print(
        \\Usage: abi search <command> [args]
        \\
        \\Commands:
        \\  create <index>                  Create a new search index
        \\  index <index> <doc_id> <text>   Add/update document in index
        \\  query <index> <query_text>      BM25 full-text search
        \\  delete <index> [doc_id]         Delete index or document
        \\  stats                           Show search index statistics
        \\  help                            Show this help
        \\
    , .{});
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
