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
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

pub const meta: command_mod.Meta = .{
    .name = "model",
    .description = "Model management (list, download, remove, search)",
    .kind = .group,
    .subcommands = &.{ "list", "info", "download", "remove", "search", "path" },
    .children = &.{
        .{ .name = "list", .description = "List cached models", .handler = runList },
        .{ .name = "info", .description = "Show detailed model information", .handler = runInfo },
        .{ .name = "download", .description = "Download model from HuggingFace or URL", .handler = runDownload },
        .{ .name = "remove", .description = "Remove a cached model", .handler = runRemove },
        .{ .name = "search", .description = "Search HuggingFace for models", .handler = runSearch },
        .{ .name = "path", .description = "Show or set cache directory", .handler = runPath },
    },
};

/// Run the model command with the provided arguments.
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
    utils.output.printError("unknown model command: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        utils.output.printInfo("did you mean: {s}", .{suggestion});
    }
    utils.output.printInfo("Run 'abi model help' for usage.", .{});
}

// ============================================================================
// Subcommand Implementations
// ============================================================================

fn runList(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
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
    var manager = abi.features.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
        utils.output.printError("initializing model manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    // Scan default directories
    scanModelDirectories(allocator, &manager);

    const models = manager.listModels();

    if (models.len == 0) {
        utils.output.println("No models found in cache.", .{});
        utils.output.println("", .{});
        utils.output.println("Download models with:", .{});
        utils.output.println("  abi model download TheBloke/Llama-2-7B-GGUF:Q4_K_M", .{});
        utils.output.println("", .{});
        utils.output.println("Or place GGUF files in:", .{});
        utils.output.println("  {s}", .{manager.getCacheDir()});
        return;
    }

    if (format_json) {
        printModelsJson(models);
    } else {
        printModelsTable(models, show_sizes);
    }
}

fn runInfo(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printInfoHelp();
        return;
    }

    const model_ref = std.mem.sliceTo(args[0], 0);

    // Check if it's a path or a name
    const has_model_ext = std.mem.endsWith(u8, model_ref, ".gguf") or
        std.mem.endsWith(u8, model_ref, ".mlx") or
        std.mem.endsWith(u8, model_ref, ".safetensors") or
        std.mem.endsWith(u8, model_ref, ".bin") or
        std.mem.endsWith(u8, model_ref, ".onnx");
    const is_path = std.mem.indexOf(u8, model_ref, "/") != null or
        std.mem.indexOf(u8, model_ref, "\\") != null or
        has_model_ext;

    if (is_path) {
        // Direct file path
        if (std.mem.endsWith(u8, model_ref, ".gguf")) {
            showGgufInfo(allocator, model_ref);
        } else {
            showModelPathInfo(model_ref);
        }
    } else {
        // Model name - look up in cache
        var manager = abi.features.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
            utils.output.printError("initializing model manager: {t}", .{err});
            return;
        };
        defer manager.deinit();

        scanModelDirectories(allocator, &manager);

        if (manager.getModel(model_ref)) |model| {
            utils.output.println("", .{});
            utils.output.printKeyValue("Model", model.name);
            utils.output.printKeyValue("Path", model.path);
            utils.output.printKeyValueFmt("Size", "{s}", .{formatSize(model.size_bytes)});
            utils.output.printKeyValueFmt("Format", "{t}", .{model.format});
            if (model.quantization) |q| {
                utils.output.printKeyValueFmt("Quantization", "{t}", .{q});
            }
            if (model.source_url) |url| {
                utils.output.printKeyValue("Source", url);
            }

            // Also show GGUF details if it's a GGUF file
            if (model.format == .gguf) {
                utils.output.println("", .{});
                utils.output.printHeader("GGUF Details");
                showGgufInfo(allocator, model.path);
            }
        } else {
            utils.output.printError("Model not found: {s}", .{model_ref});
            utils.output.printInfo("Use 'abi model list' to see available models.", .{});
        }
    }
}

fn runDownload(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
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
        utils.output.printError("Model specification required.", .{});
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
    const parsed = abi.features.ai.models.HuggingFaceClient.parseModelSpec(spec);

    utils.output.println("", .{});
    utils.output.printKeyValue("Model", parsed.model_id);

    if (parsed.filename) |filename| {
        utils.output.printKeyValue("File", filename);

        // Build download URL
        var hf_client = abi.features.ai.models.HuggingFaceClient.init(allocator, null);
        defer hf_client.deinit();

        const url = hf_client.resolveDownloadUrl(parsed.model_id, filename) catch |err| {
            utils.output.printError("building URL: {t}", .{err});
            return;
        };
        defer allocator.free(url);

        downloadFromUrl(allocator, url, output_path, verify_checksum);
    } else if (parsed.quantization_hint) |quant| {
        utils.output.printKeyValue("Quantization", quant);

        // Build filename from hint
        var hf_client = abi.features.ai.models.HuggingFaceClient.init(allocator, null);
        defer hf_client.deinit();

        const filename = hf_client.buildFilenameFromHint(parsed.model_id, quant) catch |err| {
            utils.output.printError("building filename: {t}", .{err});
            return;
        };
        defer allocator.free(filename);

        utils.output.printInfo("Resolved filename: {s}", .{filename});

        const url = hf_client.resolveDownloadUrl(parsed.model_id, filename) catch |err| {
            utils.output.printError("building URL: {t}", .{err});
            return;
        };
        defer allocator.free(url);

        downloadFromUrl(allocator, url, output_path, verify_checksum);
    } else {
        // No specific file - show available quantizations
        utils.output.println("", .{});
        utils.output.printWarning("No quantization specified. Available options:", .{});
        utils.output.println("", .{});

        const quants = abi.features.ai.models.HuggingFaceClient.getQuantizationInfo();
        for (quants) |q| {
            utils.output.println("  {s: <10} ({d:.1} bits/weight) - {s}", .{ q.name, q.bits, q.desc });
        }

        utils.output.println("", .{});
        utils.output.println("Usage:", .{});
        utils.output.println("  abi model download {s}:Q4_K_M", .{parsed.model_id});
        utils.output.println("  abi model download {s}:Q5_K_S", .{parsed.model_id});
    }
}

fn runRemove(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
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

    var manager = abi.features.ai.models.Manager.init(allocator, .{ .auto_scan = false }) catch |err| {
        utils.output.printError("initializing model manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    scanModelDirectories(allocator, &manager);

    if (manager.getModel(model_name)) |model| {
        utils.output.printKeyValue("Model", model.name);
        utils.output.printKeyValue("Path", model.path);
        utils.output.printKeyValueFmt("Size", "{s}", .{formatSize(model.size_bytes)});
        utils.output.println("", .{});

        if (!force) {
            utils.output.println("To remove this model, run:", .{});
            utils.output.println("  abi model remove {s} --force", .{model_name});
            utils.output.println("", .{});
            utils.output.printWarning("This will delete the file from disk.", .{});
            return;
        }

        // Remove from catalog
        manager.removeModel(model_name) catch |err| {
            utils.output.printError("removing model: {t}", .{err});
            return;
        };

        utils.output.printSuccess("Model removed from catalog and deleted from disk.", .{});
    } else {
        utils.output.printError("Model not found: {s}", .{model_name});
        utils.output.printInfo("Use 'abi model list' to see available models.", .{});
    }
}

fn runSearch(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    _ = allocator;

    if (utils.args.containsHelpArgs(args) or args.len == 0) {
        printSearchHelp();
        return;
    }

    const query = std.mem.sliceTo(args[0], 0);

    utils.output.println("", .{});
    utils.output.printInfo("Searching HuggingFace for: {s}", .{query});
    utils.output.println("", .{});

    // Note: Full search requires HTTP client
    // For now, show instructions for manual search

    utils.output.println("Search on HuggingFace:", .{});
    utils.output.println("  https://huggingface.co/models?search={s}&library=gguf", .{query});
    utils.output.println("", .{});

    utils.output.println("Popular GGUF model authors:", .{});
    const authors = abi.features.ai.models.HuggingFaceClient.getPopularAuthors();
    for (authors) |author| {
        utils.output.println("  - {s}", .{author});
    }

    utils.output.println("", .{});
    utils.output.println("Once you find a model, download with:", .{});
    utils.output.println("  abi model download <author>/<model>:Q4_K_M", .{});
}

fn runPath(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printPathHelp();
        return;
    }

    var manager = abi.features.ai.models.Manager.init(allocator, .{}) catch |err| {
        utils.output.printError("initializing model manager: {t}", .{err});
        return;
    };
    defer manager.deinit();

    if (args.len == 0) {
        // Show current cache directory
        utils.output.printKeyValue("Model cache directory", manager.getCacheDir());
        utils.output.printKeyValueFmt("Models cached", "{d}", .{manager.modelCount()});
        utils.output.printKeyValueFmt("Total size", "{s}", .{formatSize(manager.totalCacheSize())});
        return;
    }

    // Check for --reset flag
    const arg = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, arg, "--reset")) {
        utils.output.printSuccess("Cache directory reset to default: {s}", .{manager.getCacheDir()});
        return;
    }

    // Set new cache directory
    utils.output.printInfo("Setting cache directory: {s}", .{arg});
    utils.output.println("", .{});
    utils.output.println("To make this permanent, set the environment variable:", .{});

    if (builtin.os.tag == .windows) {
        utils.output.println("  setx ABI_MODEL_CACHE \"{s}\"", .{arg});
    } else {
        utils.output.println("  export ABI_MODEL_CACHE=\"{s}\"", .{arg});
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn scanModelDirectories(allocator: std.mem.Allocator, manager: *abi.features.ai.models.Manager) void {
    // Initialize I/O backend for directory scanning
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    manager.scanCacheDirWithIo(io) catch {};
}

fn showGgufInfo(allocator: std.mem.Allocator, path: []const u8) void {
    var gguf_file = abi.features.ai.llm.io.GgufFile.open(allocator, path) catch |err| {
        utils.output.printError("opening GGUF file: {t}", .{err});
        return;
    };
    defer gguf_file.deinit();

    gguf_file.printSummaryDebug();

    // Estimate memory and parameters
    const config = abi.features.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    const mem_estimate = config.estimateMemory();
    const param_estimate = config.estimateParameters();

    utils.output.println("", .{});
    utils.output.printKeyValueFmt("Estimated Parameters", "{d:.2}B", .{@as(f64, @floatFromInt(param_estimate)) / 1e9});
    utils.output.printKeyValueFmt("Estimated Memory", "{d:.2} GB", .{@as(f64, @floatFromInt(mem_estimate)) / (1024 * 1024 * 1024)});
    utils.output.printKeyValueFmt("Attention dims", "q={d}, kv={d}, v={d}", .{ config.queryDim(), config.kvDim(), config.valueDim() });
    utils.output.printKeyValueFmt("Local LLaMA layout", "{s}", .{if (config.supportsLlamaAttentionLayout()) "compatible" else "unsupported"});
}

fn showModelPathInfo(path: []const u8) void {
    const ext = std.fs.path.extension(path);
    const format = abi.features.ai.discovery.ModelFormat.fromExtension(ext);

    utils.output.println("", .{});
    utils.output.printKeyValue("Model path", path);
    utils.output.printKeyValueFmt("Format", "{t}", .{format});

    if (format == .mlx and builtin.os.tag == .macos) {
        utils.output.printInfo("MLX format detected (macOS): prefer Metal backend for local execution.", .{});
    }
}

fn downloadFromUrl(allocator: std.mem.Allocator, url: []const u8, output_path: ?[]const u8, verify_checksum: bool) void {
    // Determine output filename
    const filename = if (output_path) |path|
        path
    else if (std.mem.lastIndexOf(u8, url, "/")) |idx|
        url[idx + 1 ..]
    else
        "model.gguf";

    const Color = utils.output.Color;
    // Cursor control sequences (not color — always emitted)
    const up_line = "\x1b[1A";
    const clear_line = "\x1b[2K";

    utils.output.println("", .{});
    utils.output.printHeader("Downloading Model");
    utils.output.printKeyValue("URL", url);
    utils.output.printKeyValue("Output", filename);
    utils.output.println("", .{});

    // Initialize I/O backend for HTTP download
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize downloader
    var downloader = abi.features.ai.models.Downloader.init(allocator);
    defer downloader.deinit();

    // Track progress display state
    const ProgressState = struct {
        var lines_printed: usize = 0;
        var last_percent: u8 = 0;
    };

    // Progress callback with detailed multi-line display
    const progress_callback = struct {
        fn callback(progress: abi.features.ai.models.DownloadProgress) void {
            const out = utils.output;
            // Move cursor up and clear previous lines
            if (ProgressState.lines_printed > 0) {
                var i: usize = 0;
                while (i < ProgressState.lines_printed) : (i += 1) {
                    out.print("{s}{s}", .{ up_line, clear_line });
                }
            }

            // Build progress bar (40 chars wide)
            var bar: [42]u8 = undefined;
            bar[0] = '[';
            const filled = @min(@as(usize, 40), (@as(usize, progress.percent) * 40) / 100);
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
            const bar_color = if (progress.percent >= 100) Color.green() else Color.cyan();
            out.println("{s}{s}{s} {d}%{s}", .{
                bar_color,
                &bar,
                Color.reset(),
                progress.percent,
                Color.reset(),
            });

            // Line 2: Size and speed
            if (progress.total_bytes > 0) {
                out.println("{s}Size:{s} {d:.1} / {d:.1} MB  {s}Speed:{s} {d:.1} MB/s", .{
                    Color.dim(),
                    Color.reset(),
                    downloaded_mb,
                    total_mb,
                    Color.dim(),
                    Color.reset(),
                    speed_mb,
                });
            } else {
                out.println("{s}Downloaded:{s} {d:.1} MB  {s}Speed:{s} {d:.1} MB/s", .{
                    Color.dim(),
                    Color.reset(),
                    downloaded_mb,
                    Color.dim(),
                    Color.reset(),
                    speed_mb,
                });
            }

            // Line 3: ETA
            if (progress.eta_seconds) |eta| {
                if (eta >= 3600) {
                    const hours = eta / 3600;
                    const mins = (eta % 3600) / 60;
                    out.println("{s}ETA:{s} {d}h {d}m remaining", .{
                        Color.dim(),
                        Color.reset(),
                        hours,
                        mins,
                    });
                } else if (eta >= 60) {
                    const mins = eta / 60;
                    const secs = eta % 60;
                    out.println("{s}ETA:{s} {d}m {d}s remaining", .{
                        Color.dim(),
                        Color.reset(),
                        mins,
                        secs,
                    });
                } else {
                    out.println("{s}ETA:{s} {d}s remaining", .{
                        Color.dim(),
                        Color.reset(),
                        eta,
                    });
                }
            } else {
                out.println("{s}ETA:{s} calculating...", .{ Color.dim(), Color.reset() });
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
        defer allocator.free(download_result.path);

        // Clear progress lines one final time
        if (ProgressState.lines_printed > 0) {
            var i: usize = 0;
            while (i < ProgressState.lines_printed) : (i += 1) {
                utils.output.print("{s}{s}", .{ up_line, clear_line });
            }
        }

        // Show success summary
        utils.output.printSuccess("Download Complete", .{});
        utils.output.println("", .{});

        const size_mb = @as(f64, @floatFromInt(download_result.bytes_downloaded)) / (1024 * 1024);
        utils.output.printKeyValue("File", download_result.path);
        utils.output.printKeyValueFmt("Size", "{d:.2} MB", .{size_mb});
        utils.output.printKeyValueFmt("SHA256", "{s}", .{&download_result.checksum});

        if (download_result.was_resumed) {
            utils.output.printWarning("Download was resumed from partial file", .{});
        }

        if (download_result.checksum_verified) {
            utils.output.printSuccess("Checksum verified", .{});
        }

        utils.output.println("", .{});
        utils.output.printInfo("Use 'abi model info {s}' to view model details.", .{filename});
    } else |err| {
        // Download failed - show error and fallback instructions
        utils.output.printError("Download failed: {t}", .{err});

        // Show fallback curl/wget commands
        utils.output.println("You can download manually with:", .{});
        utils.output.println("", .{});
        utils.output.println("{s}curl:{s}", .{ Color.dim(), Color.reset() });
        utils.output.println("  curl -L -o \"{s}\" \"{s}\"", .{ filename, url });
        utils.output.println("", .{});
        utils.output.println("{s}wget:{s}", .{ Color.dim(), Color.reset() });
        utils.output.println("  wget -O \"{s}\" \"{s}\"", .{ filename, url });
    }
}

fn printModelsTable(models: []abi.features.ai.models.CachedModel, show_sizes: bool) void {
    utils.output.println("", .{});

    if (show_sizes) {
        utils.output.println("{s: <40} {s: <12} {s: <10}", .{ "NAME", "SIZE", "QUANT" });
        utils.output.println("{s:-<40} {s:-<12} {s:-<10}", .{ "", "", "" });
    } else {
        utils.output.println("{s: <40} {s: <10}", .{ "NAME", "QUANT" });
        utils.output.println("{s:-<40} {s:-<10}", .{ "", "" });
    }

    for (models) |model| {
        if (show_sizes) {
            if (model.quantization) |q| {
                utils.output.println("{s: <40} {s: <12} {t: <10}", .{
                    model.name,
                    formatSize(model.size_bytes),
                    q,
                });
            } else {
                utils.output.println("{s: <40} {s: <12} {s: <10}", .{
                    model.name,
                    formatSize(model.size_bytes),
                    "-",
                });
            }
        } else {
            if (model.quantization) |q| {
                utils.output.println("{s: <40} {t: <10}", .{ model.name, q });
            } else {
                utils.output.println("{s: <40} {s: <10}", .{ model.name, "-" });
            }
        }
    }

    utils.output.println("", .{});
    utils.output.println("{d} model(s) cached.", .{models.len});
}

fn printModelsJson(models: []abi.features.ai.models.CachedModel) void {
    utils.output.println("[", .{});
    for (models, 0..) |model, i| {
        utils.output.println("  {{", .{});
        utils.output.println("    \"name\": \"{s}\",", .{model.name});
        utils.output.println("    \"path\": \"{s}\",", .{model.path});
        utils.output.println("    \"size_bytes\": {d},", .{model.size_bytes});
        utils.output.println("    \"format\": \"{t}\",", .{model.format});
        if (model.quantization) |q| {
            utils.output.println("    \"quantization\": \"{t}\"", .{q});
        } else {
            utils.output.println("    \"quantization\": null", .{});
        }
        if (i < models.len - 1) {
            utils.output.println("  }},", .{});
        } else {
            utils.output.println("  }}", .{});
        }
    }
    utils.output.println("]", .{});
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
    utils.output.print("{s}", .{help_text});
}

fn printListHelp() void {
    utils.output.print(
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
    utils.output.print(
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
    utils.output.print(
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
    utils.output.print(
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
    utils.output.print(
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
    utils.output.print(
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

test {
    std.testing.refAllDecls(@This());
}
