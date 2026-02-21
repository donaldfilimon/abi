//! LLM list subcommand - List supported models and local GGUF files.

const std = @import("std");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runList() void {
    std.debug.print("Supported Model Formats\n", .{});
    std.debug.print("=======================\n\n", .{});
    std.debug.print("Recommended Default Model:\n", .{});
    std.debug.print("  GPT-2 (124M parameters) - Open source, no authentication required\n", .{});
    std.debug.print("  Download: https://huggingface.co/TheBloke/gpt2-GGUF\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("GGUF (llama.cpp format)\n", .{});
    std.debug.print("  - GPT-2 (124M, 355M, 774M, 1.5B) - Recommended for local training\n", .{});
    std.debug.print("  - LLaMA 2 (7B, 13B, 70B)\n", .{});
    std.debug.print("  - Mistral (7B)\n", .{});
    std.debug.print("  - Mixtral (8x7B MoE)\n", .{});
    std.debug.print("  - Phi-2, Phi-3\n", .{});
    std.debug.print("  - Qwen, Yi\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Quantization Types:\n", .{});
    std.debug.print("  - F32, F16, BF16 (full precision)\n", .{});
    std.debug.print("  - Q8_0 (8-bit quantization)\n", .{});
    std.debug.print("  - Q4_0, Q4_1 (4-bit quantization)\n", .{});
    std.debug.print("  - Q5_0, Q5_1 (5-bit quantization)\n", .{});
    std.debug.print("  - K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Where to download:\n", .{});
    std.debug.print("  https://huggingface.co/TheBloke\n", .{});
    std.debug.print("  https://huggingface.co/models?other=gguf\n", .{});
}

pub fn runListLocal(allocator: std.mem.Allocator, args: []const [:0]const u8) void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    var search_dir: []const u8 = ".";

    // Parse directory argument
    if (args.len > 0) {
        search_dir = std.mem.sliceTo(args[0], 0);
    }

    std.debug.print("Searching for models in: {s}\n\n", .{search_dir});

    // List .gguf files
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, search_dir, .{ .iterate = true }) catch |err| {
        std.debug.print("Error: Cannot open directory {s}: {t}\n", .{ search_dir, err });
        return;
    };
    defer dir.close(io);

    var count: u32 = 0;
    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".gguf")) {
                std.debug.print("  {s}\n", .{entry.name});
                count += 1;
            }
        } else break;
    }

    if (count == 0) {
        std.debug.print("  No GGUF models found.\n", .{});
        std.debug.print("\nDownload models from:\n", .{});
        std.debug.print("  https://huggingface.co/TheBloke\n", .{});
        std.debug.print("  https://huggingface.co/models?other=gguf\n", .{});
    } else {
        std.debug.print("\nFound {d} model(s).\n", .{count});
    }
}
