//! Synthetic training data generation handler.
//!
//! Handles the `abi train generate-data` subcommand which generates
//! random tokenized data in binary format for pipeline testing.

const std = @import("std");
const utils = @import("../../utils/mod.zig");
const common = @import("common.zig");

const cli_io = common.cli_io;

pub fn runGenerateData(args: []const [:0]const u8) !void {
    var num_samples: u32 = 1024;
    var seq_length: u32 = 128;
    var vocab_size: u32 = 32000;
    var output_path: []const u8 = "synthetic.bin";

    if (utils.args.containsHelpArgs(args)) {
        utils.output.print(
            \\Usage: abi train generate-data [options]
            \\
            \\Generate synthetic tokenized training data for pipeline testing.
            \\Produces random token ID sequences in binary format compatible
            \\with TokenizedDataset.load().
            \\
            \\Options:
            \\  --num-samples <n>    Number of sequences to generate (default: 1024)
            \\  --seq-length <n>     Tokens per sequence (default: 128)
            \\  --vocab-size <n>     Vocabulary size / max token ID (default: 32000)
            \\  --output <path>      Output file path (default: synthetic.bin)
            \\
            \\Examples:
            \\  abi train generate-data
            \\  abi train generate-data --num-samples 100 --seq-length 32 --vocab-size 1000
            \\  abi train generate-data --output /tmp/train.bin
            \\
        , .{});
        return;
    }

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--num-samples")) {
            if (i < args.len) {
                num_samples = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch 1024;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--seq-length")) {
            if (i < args.len) {
                seq_length = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch 128;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--vocab-size")) {
            if (i < args.len) {
                vocab_size = std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10) catch 32000;
                i += 1;
            }
        } else if (std.mem.eql(u8, arg, "--output")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
        }
    }

    const total_tokens: u64 = @as(u64, num_samples) * @as(u64, seq_length);
    const file_size = 8 + total_tokens * 4; // header (2 x u32) + tokens (u32 each)

    utils.output.printHeader("Generating synthetic training data");
    utils.output.printKeyValueFmt("Samples", "{d}", .{num_samples});
    utils.output.printKeyValueFmt("Seq length", "{d}", .{seq_length});
    utils.output.printKeyValueFmt("Vocab size", "{d}", .{vocab_size});
    utils.output.printKeyValueFmt("Total tokens", "{d}", .{total_tokens});
    utils.output.printKeyValueFmt("File size", "{d} bytes", .{file_size});
    utils.output.printKeyValueFmt("Output", "{s}", .{output_path});

    // Initialize I/O backend for file writing
    var io_backend = cli_io.initIoBackend(std.heap.page_allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Create output file
    const file = std.Io.Dir.cwd().createFile(io, output_path, .{}) catch |err| {
        utils.output.printError("could not create output file '{s}': {t}", .{ output_path, err });
        return;
    };
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = file.writer(io, &write_buf);
    const w = &file_writer.interface;

    // Helper: write a u32 as little-endian via the generic writer interface
    const writeU32LE = struct {
        fn f(writer: *std.Io.Writer, value: u32) !void {
            var tmp: [4]u8 = undefined;
            std.mem.writeInt(u32, &tmp, value, .little);
            try writer.writeAll(&tmp);
        }
    }.f;

    // Write header: num_samples (u32 LE) + seq_length (u32 LE)
    writeU32LE(w, num_samples) catch |err| {
        utils.output.printError("writing header: {t}", .{err});
        return;
    };
    writeU32LE(w, seq_length) catch |err| {
        utils.output.printError("writing header: {t}", .{err});
        return;
    };

    // Generate pseudorandom token sequences using a simple xoshiro PRNG
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var written: u64 = 0;
    for (0..num_samples) |_| {
        for (0..seq_length) |_| {
            const token: u32 = random.intRangeLessThan(u32, 0, vocab_size);
            writeU32LE(w, token) catch |err| {
                utils.output.printError("writing token data: {t}", .{err});
                return;
            };
            written += 1;
        }
    }

    // Flush remaining buffered data
    file_writer.flush() catch |err| {
        utils.output.printError("flushing output: {t}", .{err});
        return;
    };

    utils.output.println("", .{});
    utils.output.printSuccess("Wrote {d} tokens to {s}", .{ written, output_path });
    utils.output.println("Use with: abi train new --dataset-path {s}", .{output_path});
}
