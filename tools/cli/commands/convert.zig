//! Dataset and Model Conversion CLI command.
//!
//! Commands:
//! - convert dataset --input <file> --output <file> --format <to-wdbx|to-tokenbin>
//! - convert model --input <gguf> --output <file> --format <to-gguf|to-safetensors|info>
//! - convert embeddings --input <file> --output <file> --format <json|binary|csv>
//! - convert help

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

// Wrapper functions for comptime children dispatch
fn wrapDataset(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runDataset(ctx, args);
}
fn wrapModel(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runModel(ctx, args);
}
fn wrapEmbeddings(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    try runEmbeddings(ctx, args);
}

pub const meta: command_mod.Meta = .{
    .name = "convert",
    .description = "Dataset conversion tools (tokenbin, text, jsonl, wdbx)",
    .subcommands = &.{ "dataset", "model", "embeddings" },
    .children = &.{
        .{ .name = "dataset", .description = "Convert between dataset formats", .handler = wrapDataset },
        .{ .name = "model", .description = "Convert between model formats", .handler = wrapModel },
        .{ .name = "embeddings", .description = "Convert embedding file formats", .handler = wrapEmbeddings },
    },
};

const convert_subcommands = [_][]const u8{
    "dataset", "model", "embeddings", "help",
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = ctx;
    if (args.len == 0) {
        printHelp();
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp();
        return;
    }
    // Unknown subcommand
    utils.output.printError("Unknown convert command: {s}", .{cmd});
    if (utils.args.suggestCommand(cmd, &convert_subcommands)) |suggestion| {
        utils.output.println("Did you mean: {s}", .{suggestion});
    }
}

fn runDataset(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: ?[]const u8 = null;
    var block_size: usize = 2048;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--input")) {
            if (i < args.len) {
                input_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--format")) {
            if (i < args.len) {
                format = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--block-size")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                block_size = std.fmt.parseInt(usize, val, 10) catch 2048;
                i += 1;
            }
            continue;
        }
    }

    if (input_path == null or output_path == null or format == null) {
        utils.output.printError("--input, --output, and --format are required.", .{});
        return;
    }

    if (std.mem.eql(u8, format.?, "to-wdbx")) {
        utils.output.printInfo("Converting TokenBin {s} -> WDBX {s}...", .{ input_path.?, output_path.? });
        abi.ai.database.tokenBinToWdbx(allocator, input_path.?, output_path.?, block_size) catch |err| {
            utils.output.printError("Conversion failed: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Success.", .{});
    } else if (std.mem.eql(u8, format.?, "to-tokenbin")) {
        utils.output.printInfo("Converting WDBX {s} -> TokenBin {s}...", .{ input_path.?, output_path.? });
        abi.ai.database.wdbxToTokenBin(allocator, input_path.?, output_path.?) catch |err| {
            utils.output.printError("Conversion failed: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Success.", .{});
    } else {
        utils.output.printError("Unknown format: {s}", .{format.?});
    }
}

fn runModel(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = ctx;
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--input")) {
            if (i < args.len) {
                input_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--format")) {
            if (i < args.len) {
                format = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    if (input_path == null) {
        utils.output.printError("--input is required.", .{});
        return;
    }

    // Default format is info
    const fmt = format orelse "info";

    if (std.mem.eql(u8, fmt, "info")) {
        utils.output.println("Model Information: {s}", .{input_path.?});
        utils.output.println("", .{});

        // Check file extension to determine format
        if (std.mem.endsWith(u8, input_path.?, ".gguf")) {
            utils.output.println("Format: GGUF (GGML Universal Format)", .{});
            utils.output.println("Description: Quantized model format for llama.cpp compatible inference", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".safetensors")) {
            utils.output.println("Format: SafeTensors", .{});
            utils.output.println("Description: Safe tensor serialization format by HuggingFace", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".bin") or std.mem.endsWith(u8, input_path.?, ".pt")) {
            utils.output.println("Format: PyTorch Binary", .{});
            utils.output.println("Description: PyTorch model checkpoint", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".onnx")) {
            utils.output.println("Format: ONNX", .{});
            utils.output.println("Description: Open Neural Network Exchange format", .{});
        } else {
            utils.output.println("Format: Unknown", .{});
        }

        utils.output.println("", .{});
        utils.output.println("Supported conversions:", .{});
        utils.output.println("  GGUF <-> SafeTensors (quantization preserved)", .{});
        utils.output.println("  GGUF -> Info (metadata extraction)", .{});
        utils.output.println("", .{});
        utils.output.println("Use --format to specify output format:", .{});
        utils.output.println("  --format to-safetensors  Convert GGUF to SafeTensors", .{});
        utils.output.println("  --format to-gguf         Convert SafeTensors to GGUF", .{});
        return;
    }

    if (output_path == null) {
        utils.output.printError("--output is required for conversion.", .{});
        return;
    }

    if (std.mem.eql(u8, fmt, "to-safetensors")) {
        utils.output.printInfo("Converting {s} -> SafeTensors {s}...", .{ input_path.?, output_path.? });

        // Check AI feature
        if (!abi.ai.isEnabled()) {
            utils.output.printError("AI feature is disabled. Rebuild with: zig build -Denable-ai=true", .{});
            return;
        }

        // Model conversion would use the LLM module's export functionality
        utils.output.println("", .{});
        utils.output.println("Model conversion requires the LLM module.", .{});
        utils.output.println("This operation will:", .{});
        utils.output.println("  1. Parse GGUF metadata and tensor data", .{});
        utils.output.println("  2. Dequantize weights (if quantized)", .{});
        utils.output.println("  3. Write to SafeTensors format", .{});
        utils.output.println("", .{});
        utils.output.println("Note: Full conversion requires additional model loading support.", .{});
        utils.output.println("For now, use 'abi llm export' for model export operations.", .{});
        return;
    }

    if (std.mem.eql(u8, fmt, "to-gguf")) {
        utils.output.printInfo("Converting {s} -> GGUF {s}...", .{ input_path.?, output_path.? });

        // Check AI feature
        if (!abi.ai.isEnabled()) {
            utils.output.printError("AI feature is disabled. Rebuild with: zig build -Denable-ai=true", .{});
            return;
        }

        utils.output.println("", .{});
        utils.output.println("Model conversion requires the LLM module.", .{});
        utils.output.println("This operation will:", .{});
        utils.output.println("  1. Parse SafeTensors metadata and weights", .{});
        utils.output.println("  2. Apply quantization (Q4_0, Q4_1, Q8_0, etc.)", .{});
        utils.output.println("  3. Write to GGUF format", .{});
        utils.output.println("", .{});
        utils.output.println("Supported quantization levels:", .{});
        utils.output.println("  Q4_0 - 4-bit quantization (smallest)", .{});
        utils.output.println("  Q4_1 - 4-bit with scale", .{});
        utils.output.println("  Q5_0 - 5-bit quantization", .{});
        utils.output.println("  Q5_1 - 5-bit with scale", .{});
        utils.output.println("  Q8_0 - 8-bit quantization (best quality)", .{});
        utils.output.println("", .{});
        utils.output.println("Use 'abi train export' for full model export with quantization.", .{});
        return;
    }

    utils.output.printError("Unknown format: {s}", .{fmt});
    utils.output.println("Supported formats: info, to-safetensors, to-gguf", .{});
}

fn runEmbeddings(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--input")) {
            if (i < args.len) {
                input_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--format")) {
            if (i < args.len) {
                format = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }
    }

    if (input_path == null or output_path == null or format == null) {
        utils.output.printError("--input, --output, and --format are required.", .{});
        utils.output.println("\nUsage: abi convert embeddings --input <file> --output <file> --format <fmt>", .{});
        utils.output.println("\nFormats:", .{});
        utils.output.println("  json    JSON array format", .{});
        utils.output.println("  binary  Raw float32 binary", .{});
        utils.output.println("  csv     Comma-separated values", .{});
        return;
    }

    utils.output.printInfo("Converting embeddings: {s} -> {s} (format: {s})", .{ input_path.?, output_path.?, format.? });

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Read input file
    const content = std.Io.Dir.cwd().readFileAlloc(io, input_path.?, allocator, .limited(100 * 1024 * 1024)) catch |err| {
        utils.output.printError("Error reading input file: {t}", .{err});
        return;
    };
    defer allocator.free(content);

    utils.output.println("Read {d} bytes from input", .{content.len});

    // Determine input format by trying to parse
    var embeddings: []f32 = &[_]f32{};
    var owns_embeddings = false;

    // Try JSON first
    if (std.json.parseFromSlice(std.json.Value, allocator, content, .{})) |parsed| {
        defer parsed.deinit();

        if (parsed.value == .object) {
            if (parsed.value.object.get("embedding")) |emb_val| {
                if (emb_val == .array) {
                    var list = std.ArrayListUnmanaged(f32).empty;
                    for (emb_val.array.items) |item| {
                        if (item == .float) {
                            list.append(allocator, @floatCast(item.float)) catch continue;
                        } else if (item == .integer) {
                            list.append(allocator, @floatFromInt(item.integer)) catch continue;
                        }
                    }
                    embeddings = list.toOwnedSlice(allocator) catch &[_]f32{};
                    owns_embeddings = true;
                    utils.output.println("Parsed {d} embeddings from JSON", .{embeddings.len});
                }
            }
        } else if (parsed.value == .array) {
            var list = std.ArrayListUnmanaged(f32).empty;
            for (parsed.value.array.items) |item| {
                if (item == .float) {
                    list.append(allocator, @floatCast(item.float)) catch continue;
                } else if (item == .integer) {
                    list.append(allocator, @floatFromInt(item.integer)) catch continue;
                }
            }
            embeddings = list.toOwnedSlice(allocator) catch &[_]f32{};
            owns_embeddings = true;
            utils.output.println("Parsed {d} embeddings from JSON array", .{embeddings.len});
        }
    } else |_| {
        // Try binary format (raw f32 array)
        if (content.len >= 4 and content.len % 4 == 0) {
            const float_count = content.len / 4;
            embeddings = allocator.alloc(f32, float_count) catch {
                utils.output.printError("Error allocating memory", .{});
                return;
            };
            owns_embeddings = true;

            const bytes = @as([*]const u8, @ptrCast(content.ptr));
            for (0..float_count) |j| {
                const offset = j * 4;
                embeddings[j] = @bitCast([4]u8{ bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3] });
            }
            utils.output.println("Parsed {d} embeddings from binary format", .{float_count});
        }
    }

    defer if (owns_embeddings) allocator.free(embeddings);

    if (embeddings.len == 0) {
        utils.output.printError("Could not parse embeddings from input file", .{});
        return;
    }

    // Write output in requested format
    var output = std.ArrayListUnmanaged(u8).empty;
    defer output.deinit(allocator);

    if (std.mem.eql(u8, format.?, "json")) {
        try output.appendSlice(allocator, "{\"embedding\":[");
        for (embeddings, 0..) |v, j| {
            if (j > 0) try output.append(allocator, ',');
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d:.8}", .{v}) catch "0";
            try output.appendSlice(allocator, s);
        }
        try output.appendSlice(allocator, "]}\n");
    } else if (std.mem.eql(u8, format.?, "csv")) {
        for (embeddings, 0..) |v, j| {
            if (j > 0) try output.append(allocator, ',');
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d:.8}", .{v}) catch "0";
            try output.appendSlice(allocator, s);
        }
        try output.append(allocator, '\n');
    } else if (std.mem.eql(u8, format.?, "binary")) {
        // Write raw f32 bytes
        const byte_ptr: [*]const u8 = @ptrCast(embeddings.ptr);
        try output.appendSlice(allocator, byte_ptr[0 .. embeddings.len * 4]);
    } else {
        utils.output.printError("Unknown format: {s}", .{format.?});
        return;
    }

    // Write output file
    var file = std.Io.Dir.cwd().createFile(io, output_path.?, .{ .truncate = true }) catch |err| {
        utils.output.printError("Error creating output file: {t}", .{err});
        return;
    };
    defer file.close(io);
    file.writeStreamingAll(io, output.items) catch |err| {
        utils.output.printError("Error writing output: {t}", .{err});
        return;
    };

    utils.output.printSuccess("Success. Wrote {d} bytes to {s}", .{ output.items.len, output_path.? });
}

fn printHelp() void {
    utils.output.print(
        \\Usage: abi convert <command> [options]
        \\
        \\Commands:
        \\  dataset      Convert between dataset formats
        \\  model        Convert between model formats
        \\  embeddings   Convert embedding file formats
        \\
        \\Dataset Options:
        \\  --input <file>     Input file path
        \\  --output <file>    Output file path
        \\  --format <fmt>     Output format: to-wdbx, to-tokenbin
        \\  --block-size <n>   Block size for tokenization (default: 2048)
        \\
        \\Model Options:
        \\  --input <file>     Input model path
        \\  --output <file>    Output model path
        \\  --format <fmt>     Output format: info, to-safetensors, to-gguf
        \\
        \\Embeddings Options:
        \\  --input <file>     Input embeddings file
        \\  --output <file>    Output embeddings file
        \\  --format <fmt>     Output format: json, binary, csv
        \\
        \\Examples:
        \\  abi convert dataset --input data.bin --output data.wdbx --format to-wdbx
        \\  abi convert model --input model.gguf --format info
        \\  abi convert embeddings --input emb.json --output emb.bin --format binary
        \\
    , .{});
}
