//! LLM info subcommand - Show model information.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        utils.output.println("Usage: abi llm info <model-path>", .{});
        return;
    }

    const model_path = std.mem.sliceTo(args[0], 0);

    utils.output.println("Loading model: {s}", .{model_path});

    // Try to open as GGUF
    var gguf_file = abi.features.ai.llm.io.GgufFile.open(allocator, model_path) catch |err| {
        utils.output.printError("Failed to open model file: {t}", .{err});
        if (err == error.FileTooLarge) {
            printModelFileSizeHint(allocator, model_path);
        }
        return;
    };
    defer gguf_file.deinit();

    // Print summary
    gguf_file.printSummaryDebug();

    // Print additional info
    utils.output.println("", .{});

    // Estimate memory requirements
    const config = abi.features.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    const mem_estimate = config.estimateMemory();
    const param_estimate = config.estimateParameters();

    utils.output.printKeyValueFmt("Estimated Parameters", "{d:.2}B", .{@as(f64, @floatFromInt(param_estimate)) / 1e9});
    utils.output.printKeyValueFmt("Estimated Memory", "{d:.2} GB", .{@as(f64, @floatFromInt(mem_estimate)) / (1024 * 1024 * 1024)});
    utils.output.println("Attention dims: q={d}, kv={d}, v={d}", .{ config.queryDim(), config.kvDim(), config.valueDim() });
    utils.output.println("Head dims: q={d}, kv={d}, v={d}", .{ config.queryHeadDim(), config.keyHeadDim(), config.valueHeadDim() });
    utils.output.printKeyValueFmt("Local LLaMA layout", "{s}", .{if (config.supportsLlamaAttentionLayout()) "compatible" else "unsupported"});

    // List some tensors
    utils.output.printHeader("Tensors");
    var count: u32 = 0;
    var iter = gguf_file.tensors.iterator();
    while (iter.next()) |entry| {
        if (count >= 10) {
            utils.output.println("  ... and more", .{});
            break;
        }
        const info_val = entry.value_ptr.*;
        utils.output.print("  {s}: [{d}", .{ info_val.name, info_val.dims[0] });
        for (1..info_val.n_dims) |d| {
            utils.output.print(", {d}", .{info_val.dims[d]});
        }
        utils.output.println("] ({t})", .{info_val.tensor_type});
        count += 1;
    }
}

pub fn printUnsupportedLayoutSummary(allocator: std.mem.Allocator, model_path: []const u8) void {
    var gguf_file = abi.features.ai.llm.io.GgufFile.open(allocator, model_path) catch return;
    defer gguf_file.deinit();

    const config = abi.features.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    utils.output.printKeyValueFmt("Detected architecture", "{s}", .{config.arch});
    utils.output.println("Detected dims: hidden={d}, q={d}, kv={d}, v={d}", .{
        config.dim,
        config.queryDim(),
        config.kvDim(),
        config.valueDim(),
    });
    utils.output.println("Detected heads: q={d}, kv={d} (head dims q/kv/v={d}/{d}/{d})", .{
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
        utils.output.printWarning("Model file is empty (0 bytes). Re-download or use the real Ollama blob path.", .{});
    }
}

test {
    std.testing.refAllDecls(@This());
}
