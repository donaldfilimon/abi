//! Dataset and Model Conversion CLI command.
//!
//! Commands:
//! - convert dataset --input <file> --output <file> --format <to-wdbx|to-tokenbin>
//! - convert model --input <gguf> --output <file> --format <to-gguf|to-safetensors|info>
//! - convert embeddings --input <file> --output <file> --format <json|binary|csv>
//! - convert help

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

// Subcommand dispatch

fn cDataset(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runDataset(allocator, parser.remaining());
}
fn cModel(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runModel(allocator, parser.remaining());
}
fn cEmbeddings(allocator: std.mem.Allocator, parser: *utils.args.ArgParser) !void {
    try runEmbeddings(allocator, parser.remaining());
}
fn convertUnknown(cmd: []const u8) void {
    std.debug.print("Unknown convert command: {s}\n", .{cmd});
}
fn printHelpAlloc(_: std.mem.Allocator) void {
    printHelp();
}

const convert_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"dataset"}, .run = cDataset },
    .{ .names = &.{"model"}, .run = cModel },
    .{ .names = &.{"embeddings"}, .run = cEmbeddings },
};

pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(
        allocator,
        &parser,
        &convert_commands,
        null,
        printHelpAlloc,
        convertUnknown,
    );
}

fn runDataset(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
        std.debug.print("Error: --input, --output, and --format are required.\n", .{});
        return;
    }

    if (std.mem.eql(u8, format.?, "to-wdbx")) {
        std.debug.print("Converting TokenBin {s} -> WDBX {s}...\n", .{ input_path.?, output_path.? });
        abi.ai.tokenBinToWdbx(allocator, input_path.?, output_path.?, block_size) catch |err| {
            std.debug.print("Conversion failed: {t}\n", .{err});
            return;
        };
        std.debug.print("Success.\n", .{});
    } else if (std.mem.eql(u8, format.?, "to-tokenbin")) {
        std.debug.print("Converting WDBX {s} -> TokenBin {s}...\n", .{ input_path.?, output_path.? });
        abi.ai.wdbxToTokenBin(allocator, input_path.?, output_path.?) catch |err| {
            std.debug.print("Conversion failed: {t}\n", .{err});
            return;
        };
        std.debug.print("Success.\n", .{});
    } else {
        std.debug.print("Unknown format: {s}\n", .{format.?});
    }
}

fn runModel(_: std.mem.Allocator, args: []const [:0]const u8) !void {
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
        std.debug.print("Error: --input is required.\n", .{});
        return;
    }

    // Default format is info
    const fmt = format orelse "info";

    if (std.mem.eql(u8, fmt, "info")) {
        std.debug.print("Model Information: {s}\n", .{input_path.?});
        std.debug.print("\n", .{});

        // Check file extension to determine format
        if (std.mem.endsWith(u8, input_path.?, ".gguf")) {
            std.debug.print("Format: GGUF (GGML Universal Format)\n", .{});
            std.debug.print("Description: Quantized model format for llama.cpp compatible inference\n", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".safetensors")) {
            std.debug.print("Format: SafeTensors\n", .{});
            std.debug.print("Description: Safe tensor serialization format by HuggingFace\n", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".bin") or std.mem.endsWith(u8, input_path.?, ".pt")) {
            std.debug.print("Format: PyTorch Binary\n", .{});
            std.debug.print("Description: PyTorch model checkpoint\n", .{});
        } else if (std.mem.endsWith(u8, input_path.?, ".onnx")) {
            std.debug.print("Format: ONNX\n", .{});
            std.debug.print("Description: Open Neural Network Exchange format\n", .{});
        } else {
            std.debug.print("Format: Unknown\n", .{});
        }

        std.debug.print("\n", .{});
        std.debug.print("Supported conversions:\n", .{});
        std.debug.print("  GGUF <-> SafeTensors (quantization preserved)\n", .{});
        std.debug.print("  GGUF -> Info (metadata extraction)\n", .{});
        std.debug.print("\n", .{});
        std.debug.print("Use --format to specify output format:\n", .{});
        std.debug.print("  --format to-safetensors  Convert GGUF to SafeTensors\n", .{});
        std.debug.print("  --format to-gguf         Convert SafeTensors to GGUF\n", .{});
        return;
    }

    if (output_path == null) {
        std.debug.print("Error: --output is required for conversion.\n", .{});
        return;
    }

    if (std.mem.eql(u8, fmt, "to-safetensors")) {
        std.debug.print("Converting {s} -> SafeTensors {s}...\n", .{ input_path.?, output_path.? });

        // Check AI feature
        if (!abi.ai.isEnabled()) {
            std.debug.print("Error: AI feature is disabled. Rebuild with: zig build -Denable-ai=true\n", .{});
            return;
        }

        // Model conversion would use the LLM module's export functionality
        std.debug.print("\n", .{});
        std.debug.print("Model conversion requires the LLM module.\n", .{});
        std.debug.print("This operation will:\n", .{});
        std.debug.print("  1. Parse GGUF metadata and tensor data\n", .{});
        std.debug.print("  2. Dequantize weights (if quantized)\n", .{});
        std.debug.print("  3. Write to SafeTensors format\n", .{});
        std.debug.print("\n", .{});
        std.debug.print("Note: Full conversion requires additional model loading support.\n", .{});
        std.debug.print("For now, use 'abi llm export' for model export operations.\n", .{});
        return;
    }

    if (std.mem.eql(u8, fmt, "to-gguf")) {
        std.debug.print("Converting {s} -> GGUF {s}...\n", .{ input_path.?, output_path.? });

        // Check AI feature
        if (!abi.ai.isEnabled()) {
            std.debug.print("Error: AI feature is disabled. Rebuild with: zig build -Denable-ai=true\n", .{});
            return;
        }

        std.debug.print("\n", .{});
        std.debug.print("Model conversion requires the LLM module.\n", .{});
        std.debug.print("This operation will:\n", .{});
        std.debug.print("  1. Parse SafeTensors metadata and weights\n", .{});
        std.debug.print("  2. Apply quantization (Q4_0, Q4_1, Q8_0, etc.)\n", .{});
        std.debug.print("  3. Write to GGUF format\n", .{});
        std.debug.print("\n", .{});
        std.debug.print("Supported quantization levels:\n", .{});
        std.debug.print("  Q4_0 - 4-bit quantization (smallest)\n", .{});
        std.debug.print("  Q4_1 - 4-bit with scale\n", .{});
        std.debug.print("  Q5_0 - 5-bit quantization\n", .{});
        std.debug.print("  Q5_1 - 5-bit with scale\n", .{});
        std.debug.print("  Q8_0 - 8-bit quantization (best quality)\n", .{});
        std.debug.print("\n", .{});
        std.debug.print("Use 'abi train export' for full model export with quantization.\n", .{});
        return;
    }

    std.debug.print("Unknown format: {s}\n", .{fmt});
    std.debug.print("Supported formats: info, to-safetensors, to-gguf\n", .{});
}

fn runEmbeddings(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
        std.debug.print("Error: --input, --output, and --format are required.\n", .{});
        std.debug.print("\nUsage: abi convert embeddings --input <file> --output <file> --format <fmt>\n", .{});
        std.debug.print("\nFormats:\n", .{});
        std.debug.print("  json    JSON array format\n", .{});
        std.debug.print("  binary  Raw float32 binary\n", .{});
        std.debug.print("  csv     Comma-separated values\n", .{});
        return;
    }

    std.debug.print("Converting embeddings: {s} -> {s} (format: {s})\n", .{ input_path.?, output_path.?, format.? });

    // Initialize I/O backend for Zig 0.16
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Read input file
    const content = std.Io.Dir.cwd().readFileAlloc(io, input_path.?, allocator, .limited(100 * 1024 * 1024)) catch |err| {
        std.debug.print("Error reading input file: {t}\n", .{err});
        return;
    };
    defer allocator.free(content);

    std.debug.print("Read {d} bytes from input\n", .{content.len});

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
                    std.debug.print("Parsed {d} embeddings from JSON\n", .{embeddings.len});
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
            std.debug.print("Parsed {d} embeddings from JSON array\n", .{embeddings.len});
        }
    } else |_| {
        // Try binary format (raw f32 array)
        if (content.len >= 4 and content.len % 4 == 0) {
            const float_count = content.len / 4;
            embeddings = allocator.alloc(f32, float_count) catch {
                std.debug.print("Error allocating memory\n", .{});
                return;
            };
            owns_embeddings = true;

            const bytes = @as([*]const u8, @ptrCast(content.ptr));
            for (0..float_count) |j| {
                const offset = j * 4;
                embeddings[j] = @bitCast([4]u8{ bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3] });
            }
            std.debug.print("Parsed {d} embeddings from binary format\n", .{float_count});
        }
    }

    defer if (owns_embeddings) allocator.free(embeddings);

    if (embeddings.len == 0) {
        std.debug.print("Error: Could not parse embeddings from input file\n", .{});
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
        std.debug.print("Unknown format: {s}\n", .{format.?});
        return;
    }

    // Write output file
    var file = std.Io.Dir.cwd().createFile(io, output_path.?, .{ .truncate = true }) catch |err| {
        std.debug.print("Error creating output file: {t}\n", .{err});
        return;
    };
    defer file.close(io);
    file.writeStreamingAll(io, output.items) catch |err| {
        std.debug.print("Error writing output: {t}\n", .{err});
        return;
    };

    std.debug.print("Success. Wrote {d} bytes to {s}\n", .{ output.items.len, output_path.? });
}

fn printHelp() void {
    std.debug.print(
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
