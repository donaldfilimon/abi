//! Model Management CLI Command
//!
//! Provides commands to download, cache, list, and manage GGUF models locally.
//! Similar to `ollama pull` for model management.
//!
//! Commands:
//! - model list           - List cached models
//! - model info <name>    - Show detailed model information
//! - model download <id>  - Download from HuggingFace or URL
//! - model remove <name>  - Remove a cached model
//! - model search <query> - Search HuggingFace for models
//! - model path           - Show/set cache directory

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

const model_subcommands = [_][]const u8{
    "list",
    "info",
    "download",
    "remove",
    "search",
    "path",
    "help",
};

/// Run the model command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "list")) {
        try runList(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "info")) {
        try runInfo(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "download")) {
        try runDownload(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "remove")) {
        try runRemove(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "search")) {
        try runSearch(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "path")) {
        try runPath(allocator, args[1..]);
        return;
    }

    std.debug.print("Unknown model command: {s}\n", .{command});
    if (utils.args.suggestCommand(command, &model_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
    printHelp();
}

// ============================================================================
// Subcommand Implementations
// ============================================================================

fn runList(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printListHelp();
        return;
    }

    // Parse options
    var format_json = false;
    var show_sizes = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (std.mem.eql(u8, arg, "--json")) {
            format_json = true;
        } else if (std.mem.eql(u8, arg, "--no-size")) {
            show_sizes = false;
        }
    }

    // Initialize manager
    var manager = abi.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
        std.debug.print("Error initializing model manager: {t}\n", .{err});
        return;
    };
    defer manager.deinit();

    // Scan default directories
    scanModelDirectories(allocator, &manager);

    const models = manager.listModels();

    if (models.len == 0) {
        std.debug.print("No models found in cache.\n\n", .{});
        std.debug.print("Download models with:\n", .{});
        std.debug.print("  abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M\n\n", .{});
        std.debug.print("Or place GGUF files in:\n", .{});
        std.debug.print("  {s}\n", .{manager.getCacheDir()});
        return;
    }

    if (format_json) {
        printModelsJson(models);
    } else {
        printModelsTable(models, show_sizes);
    }
}

fn runInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printInfoHelp();
        return;
    }

    const model_ref = std.mem.sliceTo(args[0], 0);

    // Check if it's a path or a name
    const is_path = std.mem.indexOf(u8, model_ref, "/") != null or
        std.mem.indexOf(u8, model_ref, "\\") != null or
        std.mem.endsWith(u8, model_ref, ".gguf");

    if (is_path) {
        // Direct file path - use GGUF parser
        showGgufInfo(allocator, model_ref);
    } else {
        // Model name - look up in cache
        var manager = abi.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
            std.debug.print("Error initializing model manager: {t}\n", .{err});
            return;
        };
        defer manager.deinit();

        scanModelDirectories(allocator, &manager);

        if (manager.getModel(model_ref)) |model| {
            std.debug.print("\nModel: {s}\n", .{model.name});
            std.debug.print("Path: {s}\n", .{model.path});
            std.debug.print("Size: {s}\n", .{formatSize(model.size_bytes)});
            std.debug.print("Format: {t}\n", .{model.format});
            if (model.quantization) |q| {
                std.debug.print("Quantization: {t}\n", .{q});
            }
            if (model.source_url) |url| {
                std.debug.print("Source: {s}\n", .{url});
            }

            // Also show GGUF details if it's a GGUF file
            if (model.format == .gguf) {
                std.debug.print("\n--- GGUF Details ---\n", .{});
                showGgufInfo(allocator, model.path);
            }
        } else {
            std.debug.print("Model not found: {s}\n", .{model_ref});
            std.debug.print("\nUse 'abi model list' to see available models.\n", .{});
        }
    }
}

fn runDownload(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printDownloadHelp();
        return;
    }

    var model_spec: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var verify_checksum = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            if (i + 1 < args.len) {
                i += 1;
                output_path = std.mem.sliceTo(args[i], 0);
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--no-verify")) {
            verify_checksum = false;
            continue;
        }

        if (model_spec == null) {
            model_spec = arg;
        }
    }

    if (model_spec == null) {
        std.debug.print("Error: Model specification required.\n\n", .{});
        printDownloadHelp();
        return;
    }

    const spec = model_spec.?;

    // Check if it's a direct URL
    if (std.mem.startsWith(u8, spec, "http://") or std.mem.startsWith(u8, spec, "https://")) {
        downloadFromUrl(allocator, spec, output_path, verify_checksum);
        return;
    }

    // Parse HuggingFace model specification
    const parsed = abi.ai.models.HuggingFaceClient.parseModelSpec(spec);

    std.debug.print("\nModel: {s}\n", .{parsed.model_id});

    if (parsed.filename) |filename| {
        std.debug.print("File: {s}\n", .{filename});

        // Build download URL
        var hf_client = abi.ai.models.HuggingFaceClient.init(allocator, null);
        defer hf_client.deinit();

        const url = hf_client.resolveDownloadUrl(parsed.model_id, filename) catch |err| {
            std.debug.print("Error building URL: {t}\n", .{err});
            return;
        };
        defer allocator.free(url);

        downloadFromUrl(allocator, url, output_path, verify_checksum);
    } else if (parsed.quantization_hint) |quant| {
        std.debug.print("Quantization: {s}\n", .{quant});

        // Build filename from hint
        var hf_client = abi.ai.models.HuggingFaceClient.init(allocator, null);
        defer hf_client.deinit();

        const filename = hf_client.buildFilenameFromHint(parsed.model_id, quant) catch |err| {
            std.debug.print("Error building filename: {t}\n", .{err});
            return;
        };
        defer allocator.free(filename);

        std.debug.print("Resolved filename: {s}\n", .{filename});

        const url = hf_client.resolveDownloadUrl(parsed.model_id, filename) catch |err| {
            std.debug.print("Error building URL: {t}\n", .{err});
            return;
        };
        defer allocator.free(url);

        downloadFromUrl(allocator, url, output_path, verify_checksum);
    } else {
        // No specific file - show available quantizations
        std.debug.print("\nNo quantization specified. Available options:\n\n", .{});

        const quants = abi.ai.models.HuggingFaceClient.getQuantizationInfo();
        for (quants) |q| {
            std.debug.print("  {s: <10} ({d:.1} bits/weight) - {s}\n", .{ q.name, q.bits, q.desc });
        }

        std.debug.print("\nUsage:\n", .{});
        std.debug.print("  abi model download {s}:Q4_K_M\n", .{parsed.model_id});
        std.debug.print("  abi model download {s}:Q5_K_S\n", .{parsed.model_id});
    }
}

fn runRemove(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printRemoveHelp();
        return;
    }

    const model_name = std.mem.sliceTo(args[0], 0);

    // Check for --force flag
    var force = false;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--force") or
            std.mem.eql(u8, std.mem.sliceTo(arg, 0), "-f"))
        {
            force = true;
        }
    }

    var manager = abi.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
        std.debug.print("Error initializing model manager: {t}\n", .{err});
        return;
    };
    defer manager.deinit();

    scanModelDirectories(allocator, &manager);

    if (manager.getModel(model_name)) |model| {
        std.debug.print("Model: {s}\n", .{model.name});
        std.debug.print("Path: {s}\n", .{model.path});
        std.debug.print("Size: {s}\n\n", .{formatSize(model.size_bytes)});

        if (!force) {
            std.debug.print("To remove this model, run:\n", .{});
            std.debug.print("  abi model remove {s} --force\n\n", .{model_name});
            std.debug.print("Note: This will delete the file from disk.\n", .{});
            return;
        }

        // Remove from catalog
        manager.removeModel(model_name) catch |err| {
            std.debug.print("Error removing model: {t}\n", .{err});
            return;
        };

        std.debug.print("Model removed from catalog and deleted from disk.\n", .{});
    } else {
        std.debug.print("Model not found: {s}\n", .{model_name});
        std.debug.print("\nUse 'abi model list' to see available models.\n", .{});
    }
}

fn runSearch(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    _ = allocator;

    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printSearchHelp();
        return;
    }

    const query = std.mem.sliceTo(args[0], 0);

    std.debug.print("\nSearching HuggingFace for: {s}\n\n", .{query});

    // Note: Full search requires HTTP client
    // For now, show instructions for manual search

    std.debug.print("Search on HuggingFace:\n", .{});
    std.debug.print("  https://huggingface.co/models?search={s}&library=gguf\n\n", .{query});

    std.debug.print("Popular GGUF model authors:\n", .{});
    const authors = abi.ai.models.HuggingFaceClient.getPopularAuthors();
    for (authors) |author| {
        std.debug.print("  • {s}\n", .{author});
    }

    std.debug.print("\nOnce you find a model, download with:\n", .{});
    std.debug.print("  abi model download <author>/<model>:Q4_K_M\n", .{});
}

fn runPath(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printPathHelp();
        return;
    }

    var manager = abi.ai.models.Manager.init(allocator, .{}) catch |err| {
        std.debug.print("Error initializing model manager: {t}\n", .{err});
        return;
    };
    defer manager.deinit();

    if (args.len == 0) {
        // Show current cache directory
        std.debug.print("Model cache directory: {s}\n", .{manager.getCacheDir()});
        std.debug.print("Models cached: {d}\n", .{manager.modelCount()});
        std.debug.print("Total size: {s}\n", .{formatSize(manager.totalCacheSize())});
        return;
    }

    // Check for --reset flag
    const arg = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, arg, "--reset")) {
        std.debug.print("Cache directory reset to default: {s}\n", .{manager.getCacheDir()});
        return;
    }

    // Set new cache directory
    std.debug.print("Setting cache directory: {s}\n", .{arg});
    std.debug.print("\nTo make this permanent, set the environment variable:\n", .{});

    if (builtin.os.tag == .windows) {
        std.debug.print("  setx ABI_MODEL_CACHE \"{s}\"\n", .{arg});
    } else {
        std.debug.print("  export ABI_MODEL_CACHE=\"{s}\"\n", .{arg});
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn scanModelDirectories(allocator: std.mem.Allocator, manager: *abi.ai.models.Manager) void {
    // Initialize I/O backend for directory scanning
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    manager.scanCacheDirWithIo(io) catch {};
}

fn showGgufInfo(allocator: std.mem.Allocator, path: []const u8) void {
    var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, path) catch |err| {
        std.debug.print("Error opening GGUF file: {t}\n", .{err});
        return;
    };
    defer gguf_file.deinit();

    gguf_file.printSummaryDebug();

    // Estimate memory and parameters
    const config = abi.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    const mem_estimate = config.estimateMemory();
    const param_estimate = config.estimateParameters();

    std.debug.print("\nEstimated Parameters: {d:.2}B\n", .{@as(f64, @floatFromInt(param_estimate)) / 1e9});
    std.debug.print("Estimated Memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(mem_estimate)) / (1024 * 1024 * 1024)});
}

fn downloadFromUrl(allocator: std.mem.Allocator, url: []const u8, output_path: ?[]const u8, verify_checksum: bool) void {
    // Determine output filename
    const filename = if (output_path) |path|
        path
    else if (std.mem.lastIndexOf(u8, url, "/")) |idx|
        url[idx + 1 ..]
    else
        "model.gguf";

    // ANSI color codes for pretty output
    const colors = struct {
        const reset = "\x1b[0m";
        const bold = "\x1b[1m";
        const dim = "\x1b[2m";
        const green = "\x1b[32m";
        const cyan = "\x1b[36m";
        const yellow = "\x1b[33m";
        const red = "\x1b[31m";
        const up_line = "\x1b[1A";
        const clear_line = "\x1b[2K";
    };

    std.debug.print("\n{s}{s}Downloading Model{s}\n", .{ colors.bold, colors.cyan, colors.reset });
    std.debug.print("{s}URL:{s} {s}\n", .{ colors.dim, colors.reset, url });
    std.debug.print("{s}Output:{s} {s}\n\n", .{ colors.dim, colors.reset, filename });

    // Initialize I/O backend for HTTP download
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize downloader
    var downloader = abi.ai.models.Downloader.init(allocator);
    defer downloader.deinit();

    // Track progress display state
    const ProgressState = struct {
        var lines_printed: usize = 0;
        var last_percent: u8 = 0;
    };

    // Progress callback with detailed multi-line display
    const progress_callback = struct {
        fn callback(progress: abi.ai.models.DownloadProgress) void {
            // Move cursor up and clear previous lines
            if (ProgressState.lines_printed > 0) {
                var i: usize = 0;
                while (i < ProgressState.lines_printed) : (i += 1) {
                    std.debug.print("{s}{s}", .{ colors.up_line, colors.clear_line });
                }
            }

            // Build progress bar (40 chars wide)
            var bar: [42]u8 = undefined;
            bar[0] = '[';
            const filled = @min(40, (progress.percent * 40) / 100);
            for (1..41) |j| {
                if (j <= filled) {
                    bar[j] = '=';
                } else if (j == filled + 1 and filled < 40) {
                    bar[j] = '>';
                } else {
                    bar[j] = ' ';
                }
            }
            bar[41] = ']';

            // Format sizes
            const downloaded_mb = @as(f64, @floatFromInt(progress.downloaded_bytes)) / (1024 * 1024);
            const total_mb = @as(f64, @floatFromInt(progress.total_bytes)) / (1024 * 1024);
            const speed_mb = @as(f64, @floatFromInt(progress.speed_bytes_per_sec)) / (1024 * 1024);

            // Line 1: Progress bar and percentage
            const bar_color = if (progress.percent >= 100) colors.green else colors.cyan;
            std.debug.print("{s}{s}{s} {d}%{s}\n", .{
                bar_color,
                &bar,
                colors.reset,
                progress.percent,
                colors.reset,
            });

            // Line 2: Size and speed
            if (progress.total_bytes > 0) {
                std.debug.print("{s}Size:{s} {d:.1} / {d:.1} MB  {s}Speed:{s} {d:.1} MB/s\n", .{
                    colors.dim,
                    colors.reset,
                    downloaded_mb,
                    total_mb,
                    colors.dim,
                    colors.reset,
                    speed_mb,
                });
            } else {
                std.debug.print("{s}Downloaded:{s} {d:.1} MB  {s}Speed:{s} {d:.1} MB/s\n", .{
                    colors.dim,
                    colors.reset,
                    downloaded_mb,
                    colors.dim,
                    colors.reset,
                    speed_mb,
                });
            }

            // Line 3: ETA
            if (progress.eta_seconds) |eta| {
                if (eta >= 3600) {
                    const hours = eta / 3600;
                    const mins = (eta % 3600) / 60;
                    std.debug.print("{s}ETA:{s} {d}h {d}m remaining\n", .{
                        colors.dim,
                        colors.reset,
                        hours,
                        mins,
                    });
                } else if (eta >= 60) {
                    const mins = eta / 60;
                    const secs = eta % 60;
                    std.debug.print("{s}ETA:{s} {d}m {d}s remaining\n", .{
                        colors.dim,
                        colors.reset,
                        mins,
                        secs,
                    });
                } else {
                    std.debug.print("{s}ETA:{s} {d}s remaining\n", .{
                        colors.dim,
                        colors.reset,
                        eta,
                    });
                }
            } else {
                std.debug.print("{s}ETA:{s} calculating...\n", .{ colors.dim, colors.reset });
            }

            // Track state for next update
            ProgressState.lines_printed = 3;
            ProgressState.last_percent = progress.percent;
        }
    }.callback;

    // Attempt native download
    const result = downloader.downloadWithIo(io, url, .{
        .output_path = filename,
        .progress_callback = progress_callback,
        .resume_download = true,
        .verify_checksum = verify_checksum,
    });

    if (result) |download_result| {
        // Clear progress lines one final time
        if (ProgressState.lines_printed > 0) {
            var i: usize = 0;
            while (i < ProgressState.lines_printed) : (i += 1) {
                std.debug.print("{s}{s}", .{ colors.up_line, colors.clear_line });
            }
        }

        // Show success summary
        std.debug.print("{s}{s}✓ Download Complete{s}\n\n", .{ colors.bold, colors.green, colors.reset });

        const size_mb = @as(f64, @floatFromInt(download_result.bytes_downloaded)) / (1024 * 1024);
        std.debug.print("{s}File:{s} {s}\n", .{ colors.dim, colors.reset, download_result.path });
        std.debug.print("{s}Size:{s} {d:.2} MB\n", .{ colors.dim, colors.reset, size_mb });
        std.debug.print("{s}SHA256:{s} {s}\n", .{ colors.dim, colors.reset, &download_result.checksum });

        if (download_result.was_resumed) {
            std.debug.print("{s}(Download was resumed from partial file){s}\n", .{ colors.yellow, colors.reset });
        }

        if (download_result.checksum_verified) {
            std.debug.print("{s}✓ Checksum verified{s}\n", .{ colors.green, colors.reset });
        }

        std.debug.print("\nUse 'abi model info {s}' to view model details.\n", .{filename});
    } else |err| {
        // Download failed - show error and fallback instructions
        std.debug.print("\n{s}{s}Download failed: {t}{s}\n\n", .{ colors.bold, colors.red, err, colors.reset });

        // Show fallback curl/wget commands
        std.debug.print("You can download manually with:\n\n", .{});
        std.debug.print("{s}curl:{s}\n  curl -L -o \"{s}\" \"{s}\"\n\n", .{
            colors.dim,
            colors.reset,
            filename,
            url,
        });
        std.debug.print("{s}wget:{s}\n  wget -O \"{s}\" \"{s}\"\n", .{
            colors.dim,
            colors.reset,
            filename,
            url,
        });
    }
}

fn printModelsTable(models: []abi.ai.models.CachedModel, show_sizes: bool) void {
    std.debug.print("\n", .{});

    if (show_sizes) {
        std.debug.print("{s: <40} {s: <12} {s: <10}\n", .{ "NAME", "SIZE", "QUANT" });
        std.debug.print("{s:-<40} {s:-<12} {s:-<10}\n", .{ "", "", "" });
    } else {
        std.debug.print("{s: <40} {s: <10}\n", .{ "NAME", "QUANT" });
        std.debug.print("{s:-<40} {s:-<10}\n", .{ "", "" });
    }

    for (models) |model| {
        if (show_sizes) {
            if (model.quantization) |q| {
                std.debug.print("{s: <40} {s: <12} {t: <10}\n", .{
                    model.name,
                    formatSize(model.size_bytes),
                    q,
                });
            } else {
                std.debug.print("{s: <40} {s: <12} {s: <10}\n", .{
                    model.name,
                    formatSize(model.size_bytes),
                    "-",
                });
            }
        } else {
            if (model.quantization) |q| {
                std.debug.print("{s: <40} {t: <10}\n", .{ model.name, q });
            } else {
                std.debug.print("{s: <40} {s: <10}\n", .{ model.name, "-" });
            }
        }
    }

    std.debug.print("\n{d} model(s) cached.\n", .{models.len});
}

fn printModelsJson(models: []abi.ai.models.CachedModel) void {
    std.debug.print("[\n", .{});
    for (models, 0..) |model, i| {
        std.debug.print("  {{\n", .{});
        std.debug.print("    \"name\": \"{s}\",\n", .{model.name});
        std.debug.print("    \"path\": \"{s}\",\n", .{model.path});
        std.debug.print("    \"size_bytes\": {d},\n", .{model.size_bytes});
        std.debug.print("    \"format\": \"{t}\",\n", .{model.format});
        if (model.quantization) |q| {
            std.debug.print("    \"quantization\": \"{t}\"\n", .{q});
        } else {
            std.debug.print("    \"quantization\": null\n", .{});
        }
        if (i < models.len - 1) {
            std.debug.print("  }},\n", .{});
        } else {
            std.debug.print("  }}\n", .{});
        }
    }
    std.debug.print("]\n", .{});
}

fn formatSize(bytes: u64) [16]u8 {
    var buf: [16]u8 = undefined;
    @memset(&buf, 0);

    const kb: f64 = 1024;
    const mb: f64 = kb * 1024;
    const gb: f64 = mb * 1024;

    const b = @as(f64, @floatFromInt(bytes));

    if (b >= gb) {
        _ = std.fmt.bufPrint(&buf, "{d:.1} GB", .{b / gb}) catch {};
    } else if (b >= mb) {
        _ = std.fmt.bufPrint(&buf, "{d:.1} MB", .{b / mb}) catch {};
    } else if (b >= kb) {
        _ = std.fmt.bufPrint(&buf, "{d:.1} KB", .{b / kb}) catch {};
    } else {
        _ = std.fmt.bufPrint(&buf, "{d} B", .{bytes}) catch {};
    }

    return buf;
}

// ============================================================================
// Help Text
// ============================================================================

fn printHelp() void {
    const help_text =
        \\Usage: abi model <command> [options]
        \\
        \\Download, cache, and manage GGUF models locally.
        \\
        \\Commands:
        \\  list                 List cached models
        \\  info <name|path>     Show model information
        \\  download <spec>      Download model from HuggingFace or URL
        \\  remove <name>        Remove a cached model
        \\  search <query>       Search HuggingFace for models
        \\  path [dir]           Show/set cache directory
        \\  help                 Show this help
        \\
        \\Download Formats:
        \\  <author>/<model>               List available quantizations
        \\  <author>/<model>:Q4_K_M        Download specific quantization
        \\  <author>/<model>/file.gguf     Download specific file
        \\  https://...                    Download from direct URL
        \\
        \\Examples:
        \\  abi model list
        \\  abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M
        \\  abi model info llama-2-7b.Q4_K_M
        \\  abi model search "llama 7b gguf"
        \\
        \\Run 'abi model <command> --help' for command-specific help.
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

fn printListHelp() void {
    std.debug.print(
        \\Usage: abi model list [options]
        \\
        \\List all cached models.
        \\
        \\Options:
        \\  --json       Output in JSON format
        \\  --no-size    Hide file sizes
        \\  --help       Show this help
        \\
    , .{});
}

fn printInfoHelp() void {
    std.debug.print(
        \\Usage: abi model info <name|path>
        \\
        \\Show detailed information about a model.
        \\
        \\Arguments:
        \\  <name>       Model name from cache (use 'abi model list')
        \\  <path>       Path to a GGUF file
        \\
        \\Examples:
        \\  abi model info llama-2-7b.Q4_K_M
        \\  abi model info ./models/mistral-7b.gguf
        \\
    , .{});
}

fn printDownloadHelp() void {
    std.debug.print(
        \\Usage: abi model download <spec> [options]
        \\
        \\Download a GGUF model from HuggingFace or a direct URL.
        \\
        \\Features:
        \\  • Native HTTP/HTTPS download with progress bar
        \\  • Resume interrupted downloads automatically
        \\  • SHA256 checksum verification
        \\
        \\Arguments:
        \\  <spec>   Model specification in one of these formats:
        \\           - author/model-name           (lists available files)
        \\           - author/model-name:QUANT     (downloads specific quant)
        \\           - author/model-name/file.gguf (downloads specific file)
        \\           - https://...                 (direct URL)
        \\
        \\Options:
        \\  -o, --output <path>   Output file path (default: auto from URL)
        \\  --no-verify           Skip SHA256 checksum verification
        \\  --help                Show this help
        \\
        \\Quantization Types:
        \\  Q4_K_M   Medium quality, good balance (recommended)
        \\  Q4_K_S   Small, slightly lower quality
        \\  Q5_K_M   Higher quality, larger size
        \\  Q5_K_S   Medium-high quality
        \\  Q6_K     Very high quality
        \\  Q8_0     Near-lossless, largest
        \\
        \\Examples:
        \\  abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M
        \\  abi model download TheBloke/Mistral-7B-v0.1-GGUF:Q5_K_S
        \\  abi model download https://example.com/model.gguf -o my-model.gguf
        \\
    , .{});
}

fn printRemoveHelp() void {
    std.debug.print(
        \\Usage: abi model remove <name> [options]
        \\
        \\Remove a model from the local cache.
        \\
        \\Arguments:
        \\  <name>       Model name (from 'abi model list')
        \\
        \\Options:
        \\  -f, --force  Actually delete the file (required)
        \\  --help       Show this help
        \\
        \\Examples:
        \\  abi model remove llama-2-7b.Q4_K_M --force
        \\
    , .{});
}

fn printSearchHelp() void {
    std.debug.print(
        \\Usage: abi model search <query>
        \\
        \\Search HuggingFace for GGUF models.
        \\
        \\Arguments:
        \\  <query>      Search terms
        \\
        \\Examples:
        \\  abi model search "llama 7b"
        \\  abi model search "mistral instruct"
        \\  abi model search "code llama"
        \\
    , .{});
}

fn printPathHelp() void {
    std.debug.print(
        \\Usage: abi model path [directory]
        \\
        \\Show or set the model cache directory.
        \\
        \\Arguments:
        \\  [directory]  New cache directory path (optional)
        \\
        \\Options:
        \\  --reset      Reset to default cache directory
        \\  --help       Show this help
        \\
        \\Examples:
        \\  abi model path                    # Show current
        \\  abi model path ~/.my-models       # Set new path
        \\  abi model path --reset            # Reset to default
        \\
    , .{});
}
