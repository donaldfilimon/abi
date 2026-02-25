//! LLM list subcommand - List supported models and local GGUF files.

const std = @import("std");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runList() void {
    utils.output.printHeader("Supported Model Formats");
    utils.output.println("", .{});
    utils.output.println("Recommended Default Model:", .{});
    utils.output.println("  GPT-2 (124M parameters) - Open source, no authentication required", .{});
    utils.output.println("  Download: https://huggingface.co/TheBloke/gpt2-GGUF", .{});
    utils.output.println("", .{});
    utils.output.println("GGUF (llama.cpp format)", .{});
    utils.output.println("  - GPT-2 (124M, 355M, 774M, 1.5B) - Recommended for local training", .{});
    utils.output.println("  - LLaMA 2 (7B, 13B, 70B)", .{});
    utils.output.println("  - Mistral (7B)", .{});
    utils.output.println("  - Mixtral (8x7B MoE)", .{});
    utils.output.println("  - Phi-2, Phi-3", .{});
    utils.output.println("  - Qwen, Yi", .{});
    utils.output.println("", .{});
    utils.output.println("Quantization Types:", .{});
    utils.output.println("  - F32, F16, BF16 (full precision)", .{});
    utils.output.println("  - Q8_0 (8-bit quantization)", .{});
    utils.output.println("  - Q4_0, Q4_1 (4-bit quantization)", .{});
    utils.output.println("  - Q5_0, Q5_1 (5-bit quantization)", .{});
    utils.output.println("  - K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)", .{});
    utils.output.println("", .{});
    utils.output.println("Where to download:", .{});
    utils.output.println("  https://huggingface.co/TheBloke", .{});
    utils.output.println("  https://huggingface.co/models?other=gguf", .{});
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

    utils.output.println("Searching for models in: {s}\n", .{search_dir});

    // List .gguf files
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, search_dir, .{ .iterate = true }) catch |err| {
        utils.output.printError("Cannot open directory {s}: {t}", .{ search_dir, err });
        return;
    };
    defer dir.close(io);

    var count: u32 = 0;
    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".gguf")) {
                utils.output.println("  {s}", .{entry.name});
                count += 1;
            }
        } else break;
    }

    if (count == 0) {
        utils.output.println("  No GGUF models found.", .{});
        utils.output.println("\nDownload models from:", .{});
        utils.output.println("  https://huggingface.co/TheBloke", .{});
        utils.output.println("  https://huggingface.co/models?other=gguf", .{});
    } else {
        utils.output.println("\nFound {d} model(s).", .{count});
    }
}

test {
    std.testing.refAllDecls(@This());
}
