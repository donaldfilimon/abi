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
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

pub const meta: command_mod.Meta = .{
    .name = "embed",
    .description = "Generate embeddings from text (openai, mistral, cohere, ollama)",
};

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

/// When true, the user explicitly requested local fallback via --local.
var use_local_fallback: bool = false;

pub const EmbedError = error{
    NoInput,
    ProviderNotConfigured,
    EmbeddingFailed,
    FileReadError,
    OutputError,
};

/// Run the embed command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // Check if AI feature is enabled
    if (!abi.ai.isEnabled()) {
        utils.output.printError("AI feature is disabled.", .{});
        utils.output.printInfo("Rebuild with: zig build -Dfeat-ai=true (legacy: -Denable-ai=true)", .{});
        return;
    }

    var text: ?[]const u8 = null;
    var file_path: ?[]const u8 = null;
    var provider: Provider = .openai;
    var model: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var format: OutputFormat = .json;
    use_local_fallback = false;

    while (parser.hasMore()) {
        if (parser.consumeOption(&[_][]const u8{ "--text", "-t" })) |val| {
            text = val;
        } else if (parser.consumeOption(&[_][]const u8{ "--file", "-f" })) |val| {
            file_path = val;
        } else if (parser.consumeOption(&[_][]const u8{ "--provider", "-p" })) |val| {
            provider = parseProvider(val) orelse {
                utils.output.printError("Unknown provider: {s}", .{val});
                utils.output.printInfo("Available providers: openai, ollama, mistral, cohere", .{});
                return;
            };
        } else if (parser.consumeOption(&[_][]const u8{ "--model", "-m" })) |val| {
            model = val;
        } else if (parser.consumeOption(&[_][]const u8{ "--output", "-o" })) |val| {
            output_path = val;
        } else if (parser.consumeOption(&[_][]const u8{"--format"})) |val| {
            format = parseFormat(val) orelse {
                utils.output.printError("Unknown format: {s}", .{val});
                utils.output.printInfo("Available formats: json, csv, raw", .{});
                return;
            };
        } else if (parser.consumeFlag(&[_][]const u8{"--local"})) {
            use_local_fallback = true;
        } else {
            _ = parser.next();
        }
    }

    // Get input text
    const input_text = blk: {
        if (text) |t| {
            break :blk t;
        } else if (file_path) |path| {
            const content = readFile(allocator, path) catch {
                utils.output.printError("Could not read file: {s}", .{path});
                return EmbedError.FileReadError;
            };
            break :blk content;
        } else {
            utils.output.printError("No input provided. Use --text or --file", .{});
            utils.output.println("", .{});
            utils.output.println("Examples:", .{});
            utils.output.println("  abi embed --text \"Hello world\"", .{});
            utils.output.println("  abi embed --file document.txt", .{});
            utils.output.println("  abi embed --text \"Query\" --provider ollama", .{});
            utils.output.println("", .{});
            printHelp(allocator);
            return EmbedError.NoInput;
        }
    };

    utils.output.printInfo("Generating embedding using {s}...", .{provider.toString()});

    // Generate embedding
    const embedding = try generateEmbedding(allocator, provider, input_text, model);
    defer allocator.free(embedding);

    utils.output.printSuccess("Generated {d}-dimensional embedding", .{embedding.len});

    // Output embedding
    if (output_path) |path| {
        try writeOutput(allocator, path, embedding, format);
        utils.output.printSuccess("Embedding saved to: {s}", .{path});
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
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024)) catch |err| {
        utils.output.printError("reading file: {t}", .{err});
        return EmbedError.FileReadError;
    };

    return content;
}

fn generateEmbedding(allocator: std.mem.Allocator, provider: Provider, text: []const u8, model: ?[]const u8) ![]f32 {
    // Log which model is being used
    if (model) |m| {
        utils.output.printInfo("Using custom model: {s}", .{m});
    }

    // If the user explicitly requested local fallback, use it with a warning
    if (use_local_fallback) {
        utils.output.printWarning("Using local character-hash embeddings (not ML-based)", .{});
        utils.output.printInfo("These are NOT real embeddings. For production use, configure a provider API key.", .{});
        return generateLocalEmbedding(allocator, text);
    }

    switch (provider) {
        .openai => {
            const config = abi.connectors.tryLoadOpenAI(allocator) catch {
                utils.output.printError("OpenAI not configured. Set ABI_OPENAI_API_KEY environment variable.", .{});
                utils.output.printInfo("Run 'abi env' to check your environment setup.", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                utils.output.printError("ABI_OPENAI_API_KEY not set.", .{});
                utils.output.printInfo("Run 'abi env' for setup help, or use --local for non-ML fallback.", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            const effective_model = model orelse "text-embedding-3-small";
            utils.output.printKeyValue("Model", effective_model);

            // Real API call not yet implemented — tell the user clearly
            utils.output.printError("OpenAI embedding API integration not yet implemented.", .{});
            utils.output.printInfo("Use --local for character-hash fallback (not ML-based), or try --provider ollama.", .{});
            return EmbedError.EmbeddingFailed;
        },
        .ollama => {
            var config = abi.connectors.loadOllama(allocator) catch {
                utils.output.printError("Ollama not configured. Set ABI_OLLAMA_HOST or start Ollama locally.", .{});
                utils.output.printInfo("Default host: http://127.0.0.1:11434", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer config.deinit(allocator);

            const effective_model = model orelse config.model;
            utils.output.printKeyValue("Model", effective_model);

            // Real API call not yet implemented
            utils.output.printError("Ollama embedding API integration not yet implemented.", .{});
            utils.output.printInfo("Use --local for character-hash fallback (not ML-based).", .{});
            return EmbedError.EmbeddingFailed;
        },
        .mistral => {
            const config = abi.connectors.tryLoadMistral(allocator) catch {
                utils.output.printError("Mistral not configured. Set MISTRAL_API_KEY environment variable.", .{});
                utils.output.printInfo("Run 'abi env' for setup help.", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                utils.output.printError("MISTRAL_API_KEY not set.", .{});
                utils.output.printInfo("Run 'abi env' for setup help, or use --local for non-ML fallback.", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            const effective_model = model orelse "mistral-embed";
            utils.output.printKeyValue("Model", effective_model);

            utils.output.printError("Mistral embedding API integration not yet implemented.", .{});
            utils.output.printInfo("Use --local for character-hash fallback (not ML-based).", .{});
            return EmbedError.EmbeddingFailed;
        },
        .cohere => {
            const config = abi.connectors.tryLoadCohere(allocator) catch {
                utils.output.printError("Cohere not configured. Set COHERE_API_KEY environment variable.", .{});
                utils.output.printInfo("Run 'abi env' for setup help.", .{});
                return EmbedError.ProviderNotConfigured;
            } orelse {
                utils.output.printError("COHERE_API_KEY not set.", .{});
                utils.output.printInfo("Run 'abi env' for setup help, or use --local for non-ML fallback.", .{});
                return EmbedError.ProviderNotConfigured;
            };
            defer {
                var cfg = config;
                cfg.deinit(allocator);
            }

            const effective_model = model orelse "embed-english-v3.0";
            utils.output.printKeyValue("Model", effective_model);

            utils.output.printError("Cohere embedding API integration not yet implemented.", .{});
            utils.output.printInfo("Use --local for character-hash fallback (not ML-based).", .{});
            return EmbedError.EmbeddingFailed;
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
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
        utils.output.printError("creating output file: {t}", .{err});
        return EmbedError.OutputError;
    };
    defer file.close(io);

    // Build output in memory then write at once
    var output = std.ArrayListUnmanaged(u8).empty;
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

    // Use writeStreamingAll for Zig 0.16 compatibility
    file.writeStreamingAll(io, output.items) catch |err| {
        utils.output.printError("writing output: {t}", .{err});
        return EmbedError.OutputError;
    };
}

fn printOutput(allocator: std.mem.Allocator, embedding: []const f32, format: OutputFormat) !void {
    _ = allocator;

    switch (format) {
        .json => {
            utils.output.print("{{\"embedding\":[", .{});
            for (embedding, 0..) |v, i| {
                if (i > 0) utils.output.print(",", .{});
                if (i < 5 or i >= embedding.len - 2) {
                    utils.output.print("{d:.6}", .{v});
                } else if (i == 5) {
                    utils.output.print("...", .{});
                }
            }
            utils.output.println("],\"dimension\":{d}}}", .{embedding.len});
        },
        .csv => {
            // Print first few and last few values
            for (embedding[0..@min(5, embedding.len)], 0..) |v, i| {
                if (i > 0) utils.output.print(",", .{});
                utils.output.print("{d:.6}", .{v});
            }
            if (embedding.len > 7) {
                utils.output.print(",...", .{});
            }
            if (embedding.len > 5) {
                for (embedding[embedding.len - 2 ..]) |v| {
                    utils.output.print(",{d:.6}", .{v});
                }
            }
            utils.output.println("", .{});
        },
        .raw => {
            utils.output.println("First 5 values:", .{});
            for (embedding[0..@min(5, embedding.len)]) |v| {
                utils.output.println("  {d:.8}", .{v});
            }
            if (embedding.len > 5) {
                utils.output.println("  ... ({d} more values)", .{embedding.len - 5});
            }
        },
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi embed", "[options]")
        .description("Generate vector embeddings from text using various AI providers.")
        .section("Options")
        .option(.{ .short = "-t", .long = "--text", .arg = "text", .description = "Text to embed" })
        .option(.{ .short = "-f", .long = "--file", .arg = "path", .description = "File to read text from" })
        .option(.{ .short = "-p", .long = "--provider", .arg = "name", .description = "Provider: openai, ollama, mistral, cohere (default: openai)" })
        .option(.{ .short = "-m", .long = "--model", .arg = "name", .description = "Model to use (provider-specific)" })
        .option(.{ .short = "-o", .long = "--output", .arg = "path", .description = "Save embeddings to file" })
        .option(.{ .long = "--format", .arg = "type", .description = "Output format: json, csv, raw (default: json)" })
        .option(.{ .long = "--local", .description = "Use local character-hash fallback (not ML-based)" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Environment Variables")
        .text("  OPENAI_API_KEY          OpenAI API key\n")
        .text("  MISTRAL_API_KEY         Mistral API key\n")
        .text("  COHERE_API_KEY          Cohere API key\n")
        .text("  OLLAMA_HOST             Ollama host URL (default: http://127.0.0.1:11434)\n")
        .newline()
        .section("Examples")
        .example("abi embed --text \"Hello world\"", "")
        .example("abi embed --text \"Test\" --provider mistral", "")
        .example("abi embed --file document.txt --output embeddings.json", "")
        .example("abi embed --text \"Query\" --format csv --output vectors.csv", "");

    builder.print();
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

test {
    std.testing.refAllDecls(@This());
}
