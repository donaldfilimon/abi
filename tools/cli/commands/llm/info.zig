//! LLM info subcommand - Show model information.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi llm info <model-path>\n", .{});
        return;
    }

    const model_path = std.mem.sliceTo(args[0], 0);

    std.debug.print("Loading model: {s}\n", .{model_path});

    // Try to open as GGUF
    var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, model_path) catch |err| {
        std.debug.print("Error: Failed to open model file: {t}\n", .{err});
        if (err == error.FileTooLarge) {
            printModelFileSizeHint(allocator, model_path);
        }
        return;
    };
    defer gguf_file.deinit();

    // Print summary
    gguf_file.printSummaryDebug();

    // Print additional info
    std.debug.print("\n", .{});

    // Estimate memory requirements
    const config = abi.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    const mem_estimate = config.estimateMemory();
    const param_estimate = config.estimateParameters();

    std.debug.print("Estimated Parameters: {d:.2}B\n", .{@as(f64, @floatFromInt(param_estimate)) / 1e9});
    std.debug.print("Estimated Memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(mem_estimate)) / (1024 * 1024 * 1024)});
    std.debug.print("Attention dims: q={d}, kv={d}, v={d}\n", .{ config.queryDim(), config.kvDim(), config.valueDim() });
    std.debug.print("Head dims: q={d}, kv={d}, v={d}\n", .{ config.queryHeadDim(), config.keyHeadDim(), config.valueHeadDim() });
    std.debug.print("Local LLaMA layout: {s}\n", .{if (config.supportsLlamaAttentionLayout()) "compatible" else "unsupported"});

    // List some tensors
    std.debug.print("\nTensors:\n", .{});
    var count: u32 = 0;
    var iter = gguf_file.tensors.iterator();
    while (iter.next()) |entry| {
        if (count >= 10) {
            std.debug.print("  ... and more\n", .{});
            break;
        }
        const info = entry.value_ptr.*;
        std.debug.print("  {s}: [{d}", .{ info.name, info.dims[0] });
        for (1..info.n_dims) |d| {
            std.debug.print(", {d}", .{info.dims[d]});
        }
        std.debug.print("] ({t})\n", .{info.tensor_type});
        count += 1;
    }
}

pub fn printUnsupportedLayoutSummary(allocator: std.mem.Allocator, model_path: []const u8) void {
    var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, model_path) catch return;
    defer gguf_file.deinit();

    const config = abi.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    std.debug.print("Detected architecture: {s}\n", .{config.arch});
    std.debug.print("Detected dims: hidden={d}, q={d}, kv={d}, v={d}\n", .{
        config.dim,
        config.queryDim(),
        config.kvDim(),
        config.valueDim(),
    });
    std.debug.print("Detected heads: q={d}, kv={d} (head dims q/kv/v={d}/{d}/{d})\n", .{
        config.n_heads,
        config.n_kv_heads,
        config.queryHeadDim(),
        config.keyHeadDim(),
        config.valueHeadDim(),
    });
}

pub fn printModelFileSizeHint(allocator: std.mem.Allocator, model_path: []const u8) void {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = blk: {
        if (std.fs.path.isAbsolute(model_path)) {
            break :blk std.Io.Dir.openFileAbsolute(io, model_path, .{}) catch return;
        }
        break :blk std.Io.Dir.cwd().openFile(io, model_path, .{}) catch return;
    };
    defer file.close(io);

    const stat = file.stat(io) catch return;
    if (stat.size == 0) {
        std.debug.print("Model file is empty (0 bytes). Re-download or use the real Ollama blob path.\n", .{});
    }
}
