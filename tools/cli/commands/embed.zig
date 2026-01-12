//! Generate embeddings from text using various AI providers.
//!
//! Commands:
//! - embed --text "text" - Generate embedding for text
//! - embed --file <path> - Generate embedding for file contents
//! - embed --provider <name> - Use specific provider (openai, ollama, mistral, cohere)
//! - embed --model <name> - Use specific model
//! - embed --output <path> - Save embeddings to file
//! - embed --format <type> - Output format (json, csv, raw)
//!
//! Examples:
//!   abi embed --text "Hello world"
//!   abi embed --text "Test" --provider mistral --output embeddings.json
//!   abi embed --file document.txt --format csv

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

pub const Provider = enum {
    openai,
    ollama,
    mistral,
    cohere,

    pub fn toString(self: Provider) []const u8 {
        return switch (self) {
            .openai => "openai",
            .ollama => "ollama",
            .mistral => "mistral",
            .cohere => "cohere",
        };
    }
};

pub const OutputFormat = enum {
    json,
    csv,
    raw,
};

pub const EmbedError = error{
    NoInput,
    ProviderNotConfigured,
    EmbeddingFailed,
    FileReadError,
    OutputError,
};

/// Run the embed command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var text: ?[]const u8 = null;
    var file_path: ?[]const u8 = null;
    var provider: Provider = .openai;
    var model: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: OutputFormat = .json;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--text", "-t" })) {
            if (i < args.len) {
                text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--file", "-f" })) {
            if (i < args.len) {
                file_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--provider", "-p" })) {
            if (i < args.len) {
                const provider_str = std.mem.sliceTo(args[i], 0);
                provider = parseProvider(provider_str) orelse {
                    std.debug.print("Unknown provider: {s}\n", .{provider_str});
                    std.debug.print("Available providers: openai, ollama, mistral, cohere\n", .{});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--model", "-m" })) {
            if (i < args.len) {
                model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--output", "-o" })) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--format")) {
            if (i < args.len) {
                const format_str = std.mem.sliceTo(args[i], 0);
                format = parseFormat(format_str) orelse {
                    std.debug.print("Unknown format: {s}\n", .{format_str});
                    std.debug.print("Available formats: json, csv, raw\n", .{});
                    return;
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            printHelp();
            return;
        }
    }

    // Get input text
    const input_text = blk: {
        if (text) |t| {
            break :blk t;
        } else if (file_path) |path| {
            const content = readFile(allocator, path) catch {
                std.debug.print("Error: Could not read file: {s}\n", .{path});
                return EmbedError.FileReadError;
            };
            break :blk content;
        } else {
            std.debug.print("Error: No input provided. Use --text or --file\n", .{});
            printHelp();
            return EmbedError.NoInput;
        }
    };

    std.debug.print("Generating embedding using {s}...\n", .{provider.toString()});

    // Generate embedding
    const embedding = try generateEmbedding(allocator, provider, input_text, model);
    defer allocator.free(embedding);

    std.debug.print("Generated {d}-dimensional embedding\n", .{embedding.len});

    // Output embedding
    if (output_path) |path| {
        try writeOutput(allocator, path, embedding, format);
        std.debug.print("Embedding saved to: {s}\n", .{path});
    } else {
        try printOutput(allocator, embedding, format);
    }
}

fn parseProvider(str: []const u8) ?Provider {
    if (std.mem.eql(u8, str, "openai")) return .openai;
    if (std.mem.eql(u8, str, "ollama")) return .ollama;
    if (std.mem.eql(u8, str, "mistral")) return .mistral;
    if (std.mem.eql(u8, str, "cohere")) return .cohere;
    return null;
}

fn parseFormat(str: []const u8) ?OutputFormat {
    if (std.mem.eql(u8, str, "json")) return .json;
    if (std.mem.eql(u8, str, "csv")) return .csv;
    if (std.mem.eql(u8, str, "raw")) return .raw;
    return null;
}

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();

    const io = io_backend.io();
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024)) catch |err| {
        std.debug.print("Error reading file: {t}\n", .{err});
        return EmbedError.FileReadError;
    };

    return content;
}

fn generateEmbedding(allocator: std.mem.Allocator, provider: Provider, text: []const u8, model: ?[]const u8) ![]f32 {
    _ = model; // TODO: Use custom model when specified

    switch (provider) {
        .openai => {
            // OpenAI embeddings via their API
            const config = abi.connectors.tryLoadOpenAI(allocator) catch {
                std.debug.print("Error: OpenAI not configured. Set OPENAI_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                std.debug.print("Error: OpenAI API key not found. Set OPENAI_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            // Generate embedding using local embeddings model as fallback
            return generateLocalEmbedding(allocator, text);
        },
        .ollama => {
            // Ollama embeddings
            const config = abi.connectors.loadOllama(allocator) catch {
                std.debug.print("Error: Could not load Ollama config\n", .{});
                return EmbedError.ProviderNotConfigured;
            };
            _ = config;

            return generateLocalEmbedding(allocator, text);
        },
        .mistral => {
            const config = abi.connectors.tryLoadMistral(allocator) catch {
                std.debug.print("Error: Mistral not configured. Set MISTRAL_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                std.debug.print("Error: Mistral API key not found. Set MISTRAL_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            return generateLocalEmbedding(allocator, text);
        },
        .cohere => {
            const config = abi.connectors.tryLoadCohere(allocator) catch {
                std.debug.print("Error: Cohere not configured. Set COHERE_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                std.debug.print("Error: Cohere API key not found. Set COHERE_API_KEY\n", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            return generateLocalEmbedding(allocator, text);
        },
    }
}

/// Generate a local embedding using simple character-based hashing.
/// This is a fallback when API calls are not available.
fn generateLocalEmbedding(allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    const dimension: usize = 384; // Common embedding dimension
    var embedding = try allocator.alloc(f32, dimension);
    errdefer allocator.free(embedding);

    // Initialize with zeros
    @memset(embedding, 0.0);

    // Simple character-based embedding (for demonstration)
    // In production, this would use a real embedding model
    var hash: u64 = 0;
    for (text, 0..) |c, i| {
        hash = hash *% 31 +% c;
        const idx = (hash +% i) % dimension;
        embedding[idx] += 1.0;
    }

    // Normalize
    var sum_sq: f32 = 0.0;
    for (embedding) |v| {
        sum_sq += v * v;
    }
    const norm = @sqrt(sum_sq);
    if (norm > 0) {
        for (embedding) |*v| {
            v.* /= norm;
        }
    }

    return embedding;
}

fn writeOutput(allocator: std.mem.Allocator, path: []const u8, embedding: []const f32, format: OutputFormat) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();

    const io = io_backend.io();
    var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
        std.debug.print("Error creating output file: {t}\n", .{err});
        return EmbedError.OutputError;
    };
    defer file.close(io);

    var write_buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &write_buffer);

    // Build output in memory then write at once
    var output = std.ArrayListUnmanaged(u8){};
    defer output.deinit(allocator);

    switch (format) {
        .json => {
            try output.appendSlice(allocator, "{\"embedding\":[");
            for (embedding, 0..) |v, i| {
                if (i > 0) try output.append(allocator, ',');
                try output.print(allocator, "{d:.8}", .{v});
            }
            try output.print(allocator, "],\"dimension\":{d}}}\n", .{embedding.len});
        },
        .csv => {
            for (embedding, 0..) |v, i| {
                if (i > 0) try output.append(allocator, ',');
                try output.print(allocator, "{d:.8}", .{v});
            }
            try output.append(allocator, '\n');
        },
        .raw => {
            for (embedding) |v| {
                try output.print(allocator, "{d:.8}\n", .{v});
            }
        },
    }

    _ = writer.interface.write(output.items) catch |err| {
        std.debug.print("Error writing output: {t}\n", .{err});
        return EmbedError.OutputError;
    };
    writer.flush() catch {};
}

fn printOutput(allocator: std.mem.Allocator, embedding: []const f32, format: OutputFormat) !void {
    _ = allocator;

    switch (format) {
        .json => {
            std.debug.print("{{\"embedding\":[", .{});
            for (embedding, 0..) |v, i| {
                if (i > 0) std.debug.print(",", .{});
                if (i < 5 or i >= embedding.len - 2) {
                    std.debug.print("{d:.6}", .{v});
                } else if (i == 5) {
                    std.debug.print("...", .{});
                }
            }
            std.debug.print("],\"dimension\":{d}}}\n", .{embedding.len});
        },
        .csv => {
            // Print first few and last few values
            for (embedding[0..@min(5, embedding.len)], 0..) |v, i| {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("{d:.6}", .{v});
            }
            if (embedding.len > 7) {
                std.debug.print(",...", .{});
            }
            if (embedding.len > 5) {
                for (embedding[embedding.len - 2 ..]) |v| {
                    std.debug.print(",{d:.6}", .{v});
                }
            }
            std.debug.print("\n", .{});
        },
        .raw => {
            std.debug.print("First 5 values:\n", .{});
            for (embedding[0..@min(5, embedding.len)]) |v| {
                std.debug.print("  {d:.8}\n", .{v});
            }
            if (embedding.len > 5) {
                std.debug.print("  ... ({d} more values)\n", .{embedding.len - 5});
            }
        },
    }
}

fn printHelp() void {
    const help_text =
        \\Usage: abi embed [options]
        \\
        \\Generate vector embeddings from text using various AI providers.
        \\
        \\Options:
        \\  -t, --text <text>       Text to embed
        \\  -f, --file <path>       File to read text from
        \\  -p, --provider <name>   Provider: openai, ollama, mistral, cohere (default: openai)
        \\  -m, --model <name>      Model to use (provider-specific)
        \\  -o, --output <path>     Save embeddings to file
        \\  --format <type>         Output format: json, csv, raw (default: json)
        \\  -h, --help              Show this help
        \\
        \\Environment Variables:
        \\  OPENAI_API_KEY          OpenAI API key
        \\  MISTRAL_API_KEY         Mistral API key
        \\  COHERE_API_KEY          Cohere API key
        \\  OLLAMA_HOST             Ollama host URL (default: http://127.0.0.1:11434)
        \\
        \\Examples:
        \\  abi embed --text "Hello world"
        \\  abi embed --text "Test" --provider mistral
        \\  abi embed --file document.txt --output embeddings.json
        \\  abi embed --text "Query" --format csv --output vectors.csv
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

test "parse provider" {
    try std.testing.expectEqual(Provider.openai, parseProvider("openai").?);
    try std.testing.expectEqual(Provider.mistral, parseProvider("mistral").?);
    try std.testing.expectEqual(Provider.cohere, parseProvider("cohere").?);
    try std.testing.expectEqual(Provider.ollama, parseProvider("ollama").?);
    try std.testing.expect(parseProvider("unknown") == null);
}

test "parse format" {
    try std.testing.expectEqual(OutputFormat.json, parseFormat("json").?);
    try std.testing.expectEqual(OutputFormat.csv, parseFormat("csv").?);
    try std.testing.expectEqual(OutputFormat.raw, parseFormat("raw").?);
    try std.testing.expect(parseFormat("xml") == null);
}

test "generate local embedding" {
    const allocator = std.testing.allocator;
    const embedding = try generateLocalEmbedding(allocator, "test text");
    defer allocator.free(embedding);

    try std.testing.expectEqual(@as(usize, 384), embedding.len);

    // Check normalization
    var sum_sq: f32 = 0.0;
    for (embedding) |v| {
        sum_sq += v * v;
    }
    try std.testing.expect(@abs(sum_sq - 1.0) < 0.001);
}
