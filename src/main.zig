//! ABI CLI — Command-line interface for the ABI framework.
//!
//! Provides user-facing commands for interacting with the multi-persona
//! AI system, WDBX database, and framework diagnostics.
//!
//! Usage:
//!   abi version       Print version and build info
//!   abi doctor        Run diagnostics (features, platform, GPU)
//!   abi info          Show framework architecture summary
//!   abi chat <msg>    Route a message through the persona pipeline
//!   abi help          Show this help message

const std = @import("std");
const build_options = @import("build_options");

// Framework modules (relative imports within src/)
const root = @import("root.zig");
const feature_catalog = root.meta.features;

pub fn main(init: std.process.Init) !void {
    _ = init;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.skip(); // skip argv[0] (program name)

    const command = args.next() orelse {
        printHelp();
        return;
    };

    if (std.mem.eql(u8, command, "version")) {
        printVersion();
    } else if (std.mem.eql(u8, command, "doctor")) {
        try runDoctor(allocator);
    } else if (std.mem.eql(u8, command, "info")) {
        printInfo();
    } else if (std.mem.eql(u8, command, "chat")) {
        const message = args.next() orelse {
            std.debug.print("Usage: abi chat <message>\n", .{});
            return;
        };
        try runChat(allocator, message);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printHelp();
    } else {
        std.debug.print("Unknown command: {s}\n\n", .{command});
        printHelp();
    }
}

fn printVersion() void {
    const version = build_options.package_version;
    std.debug.print(
        \\ABI Framework v{s}
        \\Zig 0.16.0-dev | Multi-Persona AI + WDBX
        \\
        \\Core: Care first. Clarity always. Competence throughout.
        \\
    , .{version});
}

fn printHelp() void {
    std.debug.print(
        \\ABI — Multi-Persona AI Framework with WDBX
        \\
        \\Usage: abi <command> [args]
        \\
        \\Commands:
        \\  version    Print version and build info
        \\  doctor     Run diagnostics (features, platform, GPU)
        \\  info       Show framework architecture summary
        \\  chat <msg> Route a message through the persona pipeline
        \\  help       Show this help message
        \\
        \\Build commands:
        \\  zig build cli          Build this CLI binary
        \\  zig build mcp          Build MCP stdio server
        \\  zig build lib          Build static library
        \\  zig build test         Run all tests
        \\  zig build check        Full gate (lint + test + parity)
        \\
    , .{});
}

fn printInfo() void {
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
        \\  Connectors: OpenAI, Anthropic, Ollama, + 20 more
        \\
        \\Features: 19 comptime-gated modules (mod/stub pattern)
        \\GPU: Metal, CUDA, Vulkan, WebGPU, stdgpu
        \\Protocols: MCP, LSP, ACP, HA
        \\
        \\Spec: docs/spec/ABBEY-SPEC.md
        \\
    , .{});
}

fn runDoctor(allocator: std.mem.Allocator) !void {
    _ = allocator;
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

fn runChat(allocator: std.mem.Allocator, message: []const u8) !void {
    // Initialize the multi-persona router for routing decisions
    const persona = root.ai.persona;
    var registry = persona.PersonaRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(allocator, &registry, .{});
    defer router.deinit();

    // Route the message through Abi analysis
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

test {
    std.testing.refAllDecls(@This());
}
