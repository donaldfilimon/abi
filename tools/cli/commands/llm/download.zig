//! LLM download subcommand - Download a GGUF model from a URL.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runDownload(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi llm download <url> [--output <path>]\n\n", .{});
        std.debug.print("Download a GGUF model from a URL.\n\n", .{});
        std.debug.print("Examples:\n", .{});
        std.debug.print("  abi llm download https://example.com/model.gguf\n", .{});
        std.debug.print("  abi llm download https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf\n\n", .{});
        std.debug.print("Note: For HuggingFace models, use the 'resolve/main/' URL format.\n", .{});
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
        std.debug.print("Error: URL required\n", .{});
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

    std.debug.print("Downloading: {s}\n", .{url.?});
    std.debug.print("Output: {s}\n\n", .{final_path});

    // Initialize I/O backend for HTTP download
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize downloader
    var downloader = abi.ai.models.Downloader.init(allocator);
    defer downloader.deinit();

    const ProgressState = struct {
        var last_percent: u8 = 255;
    };

    const progress_callback = struct {
        fn callback(progress: abi.ai.models.DownloadProgress) void {
            if (progress.percent == ProgressState.last_percent) return;
            ProgressState.last_percent = progress.percent;

            const downloaded_mb = @as(f64, @floatFromInt(progress.downloaded_bytes)) / (1024 * 1024);
            const total_mb = @as(f64, @floatFromInt(progress.total_bytes)) / (1024 * 1024);
            const speed_mb = @as(f64, @floatFromInt(progress.speed_bytes_per_sec)) / (1024 * 1024);

            if (progress.total_bytes > 0) {
                std.debug.print("\r{d}% ({d:.1}/{d:.1} MB) {d:.1} MB/s", .{
                    progress.percent,
                    downloaded_mb,
                    total_mb,
                    speed_mb,
                });
            } else {
                std.debug.print("\r{d:.1} MB {d:.1} MB/s", .{
                    downloaded_mb,
                    speed_mb,
                });
            }

            if (progress.percent >= 100) {
                std.debug.print("\n", .{});
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
        std.debug.print("Download complete: {s}\n", .{download_result.path});
        std.debug.print("Size: {d:.2} MB\n", .{size_mb});
        std.debug.print("SHA256: {s}\n", .{&download_result.checksum});
        if (download_result.was_resumed) {
            std.debug.print("Note: Download resumed from partial file.\n", .{});
        }
        if (download_result.checksum_verified) {
            std.debug.print("Checksum verified.\n", .{});
        }
    } else |err| {
        std.debug.print("\nDownload failed: {t}\n\n", .{err});
        std.debug.print("Manual download options:\n", .{});
        std.debug.print("  curl -L -o {s} \"{s}\"\n", .{ final_path, url.? });
        std.debug.print("  wget -O {s} \"{s}\"\n", .{ final_path, url.? });
    }
}
