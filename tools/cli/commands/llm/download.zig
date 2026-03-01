//! LLM download subcommand - Download a GGUF model from a URL.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runDownload(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        utils.output.println("Usage: abi llm download <url> [--output <path>]", .{});
        utils.output.println("", .{});
        utils.output.println("Download a GGUF model from a URL.", .{});
        utils.output.println("", .{});
        utils.output.println("Examples:", .{});
        utils.output.println("  abi llm download https://example.com/model.gguf", .{});
        utils.output.println("  abi llm download https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf", .{});
        utils.output.println("", .{});
        utils.output.println("Note: For HuggingFace models, use the 'resolve/main/' URL format.", .{});
        return;
    }

    var url: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output") or std.mem.eql(u8, std.mem.sliceTo(arg, 0), "-o")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (url == null) {
            url = std.mem.sliceTo(arg, 0);
        }
    }

    if (url == null) {
        utils.output.printError("URL required", .{});
        return;
    }

    // Extract filename from URL if no output path specified
    const final_path = output_path orelse blk: {
        // Find last '/' in URL
        if (std.mem.lastIndexOf(u8, url.?, "/")) |idx| {
            break :blk url.?[idx + 1 ..];
        }
        break :blk "model.gguf";
    };

    utils.output.println("Downloading: {s}", .{url.?});
    utils.output.printKeyValueFmt("Output", "{s}", .{final_path});
    utils.output.println("", .{});

    // Initialize I/O backend for HTTP download
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize downloader
    var downloader = abi.features.ai.models.Downloader.init(allocator);
    defer downloader.deinit();

    const ProgressState = struct {
        var last_percent: u8 = 255;
    };

    const progress_callback = struct {
        fn callback(progress: abi.features.ai.models.DownloadProgress) void {
            if (progress.percent == ProgressState.last_percent) return;
            ProgressState.last_percent = progress.percent;

            const downloaded_mb = @as(f64, @floatFromInt(progress.downloaded_bytes)) / (1024 * 1024);
            const total_mb = @as(f64, @floatFromInt(progress.total_bytes)) / (1024 * 1024);
            const speed_mb = @as(f64, @floatFromInt(progress.speed_bytes_per_sec)) / (1024 * 1024);

            if (progress.total_bytes > 0) {
                utils.output.print("\r{d}% ({d:.1}/{d:.1} MB) {d:.1} MB/s", .{
                    progress.percent,
                    downloaded_mb,
                    total_mb,
                    speed_mb,
                });
            } else {
                utils.output.print("\r{d:.1} MB {d:.1} MB/s", .{
                    downloaded_mb,
                    speed_mb,
                });
            }

            if (progress.percent >= 100) {
                utils.output.println("", .{});
            }
        }
    }.callback;

    const result = downloader.downloadWithIo(io, url.?, .{
        .output_path = final_path,
        .progress_callback = progress_callback,
        .resume_download = true,
    });

    if (result) |download_result| {
        defer allocator.free(download_result.path);
        const size_mb = @as(f64, @floatFromInt(download_result.bytes_downloaded)) / (1024 * 1024);
        utils.output.printSuccess("Download complete: {s}", .{download_result.path});
        utils.output.printKeyValueFmt("Size", "{d:.2} MB", .{size_mb});
        utils.output.printKeyValueFmt("SHA256", "{s}", .{&download_result.checksum});
        if (download_result.was_resumed) {
            utils.output.printInfo("Download resumed from partial file.", .{});
        }
        if (download_result.checksum_verified) {
            utils.output.printSuccess("Checksum verified.", .{});
        }
    } else |err| {
        utils.output.println("", .{});
        utils.output.printError("Download failed: {t}", .{err});
        utils.output.println("", .{});
        utils.output.println("Manual download options:", .{});
        utils.output.println("  curl -L -o {s} \"{s}\"", .{ final_path, url.? });
        utils.output.println("  wget -O {s} \"{s}\"", .{ final_path, url.? });
    }
}

test {
    std.testing.refAllDecls(@This());
}
