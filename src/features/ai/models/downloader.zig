//! Model Downloader - HTTP download with progress and resume support
//!
//! Provides file download functionality with progress callbacks, resume support,
//! checksum verification, and retry logic for downloading large model files.
//!
//! Features:
//! - Native HTTP/HTTPS download using std.http.Client (Zig 0.16)
//! - Streaming download with progress callback
//! - Resume interrupted downloads via HTTP Range headers
//! - SHA256 checksum verification
//! - Configurable retry with exponential backoff

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const Sha256 = std.crypto.hash.sha2.Sha256;
const shared_utils = @import("../../../services/shared/utils.zig");
const platform_time = @import("../../../services/shared/time.zig");

/// Download configuration options.
pub const DownloadConfig = struct {
    /// Custom output path (if null, uses filename from URL).
    output_path: ?[]const u8 = null,
    /// Progress callback function.
    progress_callback: ?*const fn (DownloadProgress) void = null,
    /// Whether to resume interrupted downloads.
    resume_download: bool = true,
    /// Connection timeout in milliseconds.
    timeout_ms: u32 = 60_000,
    /// Maximum retries on failure.
    max_retries: u32 = 3,
    /// Buffer size for streaming download.
    buffer_size: usize = 64 * 1024, // 64KB
    /// Whether to verify SSL certificates.
    verify_ssl: bool = true,
    /// Whether to verify checksum after download (requires expected_checksum).
    verify_checksum: bool = true,
    /// Expected SHA256 checksum (hex string, optional).
    expected_checksum: ?[]const u8 = null,
};

/// Result of a successful download.
pub const DownloadResult = struct {
    /// Path to the downloaded file.
    path: []const u8,
    /// Total bytes downloaded.
    bytes_downloaded: u64,
    /// SHA256 checksum of the downloaded file (hex string).
    checksum: [64]u8,
    /// Whether the download was resumed.
    was_resumed: bool,
    /// Whether checksum was verified successfully.
    checksum_verified: bool,
};

/// Progress information for download callbacks.
pub const DownloadProgress = struct {
    /// Total file size in bytes (0 if unknown).
    total_bytes: u64,
    /// Bytes downloaded so far.
    downloaded_bytes: u64,
    /// Current download speed in bytes per second.
    speed_bytes_per_sec: u64,
    /// Estimated time remaining in seconds (null if unknown).
    eta_seconds: ?u32,
    /// Whether download is resuming from a previous attempt.
    is_resuming: bool,
    /// Percentage complete (0-100).
    percent: u8,

    /// Format progress as a human-readable string.
    pub fn format(self: DownloadProgress) [64]u8 {
        var buf = std.mem.zeroes([64]u8);
        formatProgressInto(&buf, self);
        return buf;
    }
};

fn formatProgressInto(buf: []u8, progress: DownloadProgress) void {
    const downloaded_mb = @as(f64, @floatFromInt(progress.downloaded_bytes)) / (1024 * 1024);
    const total_mb = @as(f64, @floatFromInt(progress.total_bytes)) / (1024 * 1024);
    const speed_mb = @as(f64, @floatFromInt(progress.speed_bytes_per_sec)) / (1024 * 1024);

    if (progress.total_bytes > 0) {
        safeBufPrint(buf, "{d:.1} / {d:.1} MB ({d}%) - {d:.1} MB/s", .{
            downloaded_mb,
            total_mb,
            progress.percent,
            speed_mb,
        });
    } else {
        safeBufPrint(buf, "{d:.1} MB - {d:.1} MB/s", .{
            downloaded_mb,
            speed_mb,
        });
    }
}

/// Download error types.
pub const DownloadError = error{
    /// Network connection failed.
    ConnectionFailed,
    /// HTTP request failed.
    HttpError,
    /// File write failed.
    FileWriteError,
    /// Invalid URL.
    InvalidUrl,
    /// Download was cancelled.
    Cancelled,
    /// Server returned error status.
    ServerError,
    /// Resume not supported by server.
    ResumeNotSupported,
    /// Timeout during download.
    Timeout,
    /// Out of memory.
    OutOfMemory,
    /// Download feature disabled.
    DownloadDisabled,
    /// Checksum verification failed.
    ChecksumMismatch,
    /// Failed to create output file.
    FileCreateError,
    /// Failed to read existing file for resume.
    FileReadError,
};

/// Model file downloader with progress tracking.
pub const Downloader = struct {
    allocator: std.mem.Allocator,
    cancelled: bool,

    const Self = @This();

    /// Initialize the downloader.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .cancelled = false,
        };
    }

    /// Deinitialize the downloader.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Cancel an ongoing download.
    pub fn cancel(self: *Self) void {
        self.cancelled = true;
    }

    /// Reset cancel state for a new download.
    pub fn reset(self: *Self) void {
        self.cancelled = false;
    }

    /// Download a file from a URL.
    ///
    /// This function performs an HTTP GET request and streams the response
    /// to a file. Supports resume via HTTP Range headers if the server supports it.
    ///
    /// Returns the path to the downloaded file.
    pub fn download(self: *Self, url: []const u8, config: DownloadConfig) DownloadError![]const u8 {
        // Validate URL
        if (!isValidUrl(url)) {
            return error.InvalidUrl;
        }

        // Determine output path
        const output_path = if (config.output_path) |path|
            try self.allocator.dupe(u8, path)
        else
            try extractFilenameFromUrl(self.allocator, url);

        errdefer self.allocator.free(output_path);

        // Perform download with retries
        var retries: u32 = 0;
        while (retries <= config.max_retries) : (retries += 1) {
            if (self.cancelled) {
                return error.Cancelled;
            }

            const result = self.performDownload(url, output_path, config);
            if (result) |_| {
                return output_path;
            } else |err| {
                if (retries == config.max_retries) {
                    return err;
                }
                // Wait before retry with exponential backoff
                const delay_ms = @as(u64, 1000) * (@as(u64, 1) << @intCast(retries));
                shared_utils.sleepMs(delay_ms);
            }
        }

        return error.ConnectionFailed;
    }

    /// Download a file with I/O backend (Zig 0.16 compatible).
    ///
    /// This is the full implementation that uses std.http.Client for native HTTP download.
    /// Supports resume via Range headers and computes SHA256 checksum during download.
    pub fn downloadWithIo(
        self: *Self,
        io: std.Io,
        url: []const u8,
        config: DownloadConfig,
    ) DownloadError!DownloadResult {
        // Validate URL
        if (!isValidUrl(url)) {
            return error.InvalidUrl;
        }

        // Determine output path
        const output_path = if (config.output_path) |path|
            self.allocator.dupe(u8, path) catch return error.OutOfMemory
        else
            extractFilenameFromUrl(self.allocator, url) catch return error.OutOfMemory;
        errdefer self.allocator.free(output_path);

        // Perform download with retries
        var retries: u32 = 0;
        var last_error: DownloadError = error.ConnectionFailed;

        while (retries <= config.max_retries) : (retries += 1) {
            if (self.cancelled) {
                return error.Cancelled;
            }

            const result = self.performDownloadWithIo(io, url, output_path, config);
            if (result) |download_result| {
                return download_result;
            } else |err| {
                last_error = err;
                if (retries < config.max_retries) {
                    // Exponential backoff: 1s, 2s, 4s, ...
                    const delay_ms = @as(u64, 1000) * (@as(u64, 1) << @intCast(retries));
                    shared_utils.sleepMs(delay_ms);
                }
            }
        }

        return last_error;
    }

    /// Internal download implementation (placeholder for non-I/O path).
    fn performDownload(
        self: *Self,
        url: []const u8,
        output_path: []const u8,
        config: DownloadConfig,
    ) DownloadError!void {
        _ = self;
        _ = url;
        _ = output_path;
        _ = config;
        // Non-I/O path requires downloadWithIo - fall back to showing instructions
        return error.DownloadDisabled;
    }

    /// Internal download implementation with I/O backend.
    /// Performs streaming HTTP download with progress, resume, and checksum support.
    fn performDownloadWithIo(
        self: *Self,
        io: std.Io,
        url: []const u8,
        output_path: []const u8,
        config: DownloadConfig,
    ) DownloadError!DownloadResult {
        // Initialize HTTP client (uses provided I/O backend)
        var client = std.http.Client{
            .allocator = self.allocator,
            .io = io,
        };
        defer client.deinit();

        const uri = std.Uri.parse(url) catch return error.InvalidUrl;

        // Determine if we should resume
        var resume_from: u64 = 0;
        var was_resumed = false;
        var file: std.Io.File = undefined;

        if (config.resume_download) {
            if (std.Io.Dir.cwd().openFile(io, output_path, .{ .mode = .read_write })) |existing| {
                file = existing;
                const stat = file.stat(io) catch {
                    file.close(io);
                    return error.FileReadError;
                };
                resume_from = stat.size;
                was_resumed = resume_from > 0;
            } else |err| switch (err) {
                error.FileNotFound => {},
                else => return error.FileCreateError,
            }
        }

        if (resume_from == 0) {
            file = std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true, .read = true }) catch
                return error.FileCreateError;
        } else if (!config.resume_download) {
            file = std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true, .read = true }) catch
                return error.FileCreateError;
            resume_from = 0;
            was_resumed = false;
        }
        defer file.close(io);

        // Build request options (Range header for resume)
        var request_options: std.http.Client.RequestOptions = .{};
        request_options.headers.user_agent = .{ .override = "abi-model-downloader" };

        var range_buf: [32]u8 = undefined;
        var range_header: [1]std.http.Header = undefined;
        if (resume_from > 0) {
            const range_value = formatRangeHeader(resume_from, &range_buf);
            range_header[0] = .{ .name = "range", .value = range_value };
            request_options.extra_headers = range_header[0..1];
        }

        var req = client.request(.GET, uri, request_options) catch return error.ConnectionFailed;
        defer req.deinit();

        req.sendBodiless() catch return error.ConnectionFailed;

        var redirect_buffer: [4096]u8 = undefined;
        var response = req.receiveHead(&redirect_buffer) catch return error.HttpError;

        // Handle server response status
        if (response.head.status == .range_not_satisfiable and resume_from > 0) {
            // Assume file is already fully downloaded; verify checksum if needed.
            var hasher = Sha256.init(.{});
            var checksum = std.mem.zeroes([64]u8);

            var hash_buf: [8192]u8 = undefined;
            var offset: u64 = 0;
            while (offset < resume_from) {
                const to_read = @min(hash_buf.len, @as(usize, @intCast(resume_from - offset)));
                const n = file.readPositional(io, &.{hash_buf[0..to_read]}, offset) catch
                    return error.FileReadError;
                if (n == 0) break;
                hasher.update(hash_buf[0..n]);
                offset += @intCast(n);
            }

            const digest = hasher.finalResult();
            checksum = std.fmt.bytesToHex(digest, .lower);

            var checksum_verified = false;
            if (config.verify_checksum and config.expected_checksum != null) {
                checksum_verified = std.ascii.eqlIgnoreCase(config.expected_checksum.?, checksum[0..]);
                if (!checksum_verified) return error.ChecksumMismatch;
            }

            return DownloadResult{
                .path = output_path,
                .bytes_downloaded = resume_from,
                .checksum = checksum,
                .was_resumed = true,
                .checksum_verified = checksum_verified,
            };
        }

        if (response.head.status != .ok and response.head.status != .partial_content) {
            return error.ServerError;
        }

        // If server ignored Range request, restart download from scratch
        if (resume_from > 0 and response.head.status == .ok) {
            file.close(io);
            file = std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true, .read = true }) catch
                return error.FileCreateError;
            resume_from = 0;
            was_resumed = false;
        }

        // Calculate total bytes if known
        const content_length = response.head.content_length orelse 0;
        const total_bytes: u64 = if (resume_from > 0 and response.head.status == .partial_content)
            resume_from + content_length
        else
            content_length;

        // Initialize SHA256 hasher (include existing bytes if resuming)
        var hasher = Sha256.init(.{});
        if (was_resumed) {
            var hash_buf: [8192]u8 = undefined;
            var offset: u64 = 0;
            while (offset < resume_from) {
                const to_read = @min(hash_buf.len, @as(usize, @intCast(resume_from - offset)));
                const n = file.readPositional(io, &.{hash_buf[0..to_read]}, offset) catch
                    return error.FileReadError;
                if (n == 0) break;
                hasher.update(hash_buf[0..n]);
                offset += @intCast(n);
            }
        }

        // Stream response body to file.
        // Keep reader internals and payload copy buffers separate; aliasing them
        // can panic in std.Io.Reader when it copies from its internal window.
        const reader_buffer = self.allocator.alloc(u8, config.buffer_size) catch
            return error.OutOfMemory;
        defer self.allocator.free(reader_buffer);

        const transfer_buffer = self.allocator.alloc(u8, config.buffer_size) catch
            return error.OutOfMemory;
        defer self.allocator.free(transfer_buffer);

        const reader = response.reader(reader_buffer);

        var write_offset: u64 = resume_from;
        var downloaded_total: u64 = resume_from;
        var downloaded_session: u64 = 0;

        const download_start = platform_time.now();
        var last_activity = platform_time.now();

        while (true) {
            if (self.cancelled) return error.Cancelled;

            if (config.timeout_ms > 0) {
                const now = platform_time.now();
                if (platform_time.elapsedMs(last_activity, now) > config.timeout_ms) {
                    return error.Timeout;
                }
            }

            const n = reader.readSliceShort(transfer_buffer) catch return error.ConnectionFailed;
            if (n == 0) break;

            last_activity = platform_time.now();

            file.writePositionalAll(io, transfer_buffer[0..n], write_offset) catch
                return error.FileWriteError;

            write_offset += @intCast(n);
            downloaded_total += @intCast(n);
            downloaded_session += @intCast(n);

            hasher.update(transfer_buffer[0..n]);

            if (config.progress_callback) |cb| {
                const now = platform_time.now();
                const elapsed_ms = platform_time.elapsedMs(download_start, now);
                const speed_bytes_per_sec = if (elapsed_ms > 0)
                    @as(u64, @intCast((downloaded_session * 1000) / elapsed_ms))
                else
                    0;

                const remaining_bytes: u64 = if (total_bytes > downloaded_total)
                    total_bytes - downloaded_total
                else
                    0;
                const eta_seconds: ?u32 = if (remaining_bytes > 0 and speed_bytes_per_sec > 0)
                    @intCast(remaining_bytes / speed_bytes_per_sec)
                else
                    null;

                const percent: u8 = if (total_bytes > 0)
                    @intCast(@min(@as(u64, 100), (downloaded_total * 100) / total_bytes))
                else
                    0;

                cb(.{
                    .total_bytes = total_bytes,
                    .downloaded_bytes = downloaded_total,
                    .speed_bytes_per_sec = speed_bytes_per_sec,
                    .eta_seconds = eta_seconds,
                    .is_resuming = was_resumed,
                    .percent = percent,
                });
            }
        }

        // Finalize checksum
        const digest = hasher.finalResult();
        const checksum = std.fmt.bytesToHex(digest, .lower);

        var checksum_verified = false;
        if (config.verify_checksum and config.expected_checksum != null) {
            checksum_verified = std.ascii.eqlIgnoreCase(config.expected_checksum.?, checksum[0..]);
            if (!checksum_verified) return error.ChecksumMismatch;
        }

        return DownloadResult{
            .path = output_path,
            .bytes_downloaded = downloaded_total,
            .checksum = checksum,
            .was_resumed = was_resumed,
            .checksum_verified = checksum_verified,
        };
    }

    /// Format Range header value for resume using a caller-provided buffer.
    /// Thread-safe: the caller owns the buffer, so there is no shared mutable state.
    fn formatRangeHeader(start_byte: u64, buf: []u8) []const u8 {
        const slice = std.fmt.bufPrint(buf, "bytes={d}-", .{start_byte}) catch return "bytes=0-";
        return slice;
    }

    /// Parse a URL into components.
    pub fn parseUrl(url: []const u8) ?UrlComponents {
        // Check for protocol
        const https = std.mem.startsWith(u8, url, "https://");
        const http = std.mem.startsWith(u8, url, "http://");

        if (!https and !http) return null;

        const protocol_len: usize = if (https) 8 else 7;
        const rest = url[protocol_len..];

        // Find path separator
        const path_start = std.mem.indexOf(u8, rest, "/") orelse rest.len;
        const host_port = rest[0..path_start];
        const path = if (path_start < rest.len) rest[path_start..] else "/";

        // Parse host and port
        const port_sep = std.mem.indexOf(u8, host_port, ":");
        const host = if (port_sep) |sep| host_port[0..sep] else host_port;
        const port: u16 = if (port_sep) |sep| blk: {
            break :blk std.fmt.parseInt(u16, host_port[sep + 1 ..], 10) catch
                if (https) 443 else 80;
        } else if (https) 443 else 80;

        return UrlComponents{
            .scheme = if (https) .https else .http,
            .host = host,
            .port = port,
            .path = path,
        };
    }
};

/// URL components after parsing.
pub const UrlComponents = struct {
    scheme: Scheme,
    host: []const u8,
    port: u16,
    path: []const u8,

    pub const Scheme = enum { http, https };
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a URL is valid.
fn isValidUrl(url: []const u8) bool {
    if (url.len < 10) return false; // Minimum: "http://a.b"

    // Must start with http:// or https://
    if (!std.mem.startsWith(u8, url, "http://") and
        !std.mem.startsWith(u8, url, "https://"))
    {
        return false;
    }

    // Must have a host
    const protocol_len: usize = if (std.mem.startsWith(u8, url, "https://")) 8 else 7;
    const rest = url[protocol_len..];
    if (rest.len == 0) return false;

    // Check for invalid characters
    for (rest) |char| {
        if (char < 0x20 or char > 0x7E) return false;
    }

    return true;
}

/// Extract filename from a URL.
fn extractFilenameFromUrl(allocator: std.mem.Allocator, url: []const u8) ![]const u8 {
    // Find last '/' in URL
    const last_slash = std.mem.lastIndexOf(u8, url, "/");
    if (last_slash) |idx| {
        const filename = url[idx + 1 ..];
        // Remove query string if present
        const query_start = std.mem.indexOf(u8, filename, "?");
        const clean_name = if (query_start) |qs| filename[0..qs] else filename;

        if (clean_name.len > 0) {
            return try allocator.dupe(u8, clean_name);
        }
    }

    // Fallback
    return try allocator.dupe(u8, "model.gguf");
}

/// Format bytes as human-readable string.
pub fn formatBytes(bytes: u64) [32]u8 {
    var buf = std.mem.zeroes([32]u8);

    formatBytesInto(&buf, bytes);
    return buf;
}

fn formatBytesInto(buf: []u8, bytes: u64) void {
    const kb: f64 = 1024;
    const mb: f64 = kb * 1024;
    const gb: f64 = mb * 1024;

    const b = @as(f64, @floatFromInt(bytes));

    if (b >= gb) {
        safeBufPrint(buf, "{d:.2} GB", .{b / gb});
    } else if (b >= mb) {
        safeBufPrint(buf, "{d:.2} MB", .{b / mb});
    } else if (b >= kb) {
        safeBufPrint(buf, "{d:.2} KB", .{b / kb});
    } else {
        safeBufPrint(buf, "{d} B", .{bytes});
    }
}

/// Format duration in seconds as human-readable string.
pub fn formatDuration(seconds: u32) [16]u8 {
    var buf = std.mem.zeroes([16]u8);

    formatDurationInto(&buf, seconds);
    return buf;
}

fn formatDurationInto(buf: []u8, seconds: u32) void {
    if (seconds >= 3600) {
        const hours = seconds / 3600;
        const mins = (seconds % 3600) / 60;
        safeBufPrint(buf, "{d}h {d}m", .{ hours, mins });
    } else if (seconds >= 60) {
        const mins = seconds / 60;
        const secs = seconds % 60;
        safeBufPrint(buf, "{d}m {d}s", .{ mins, secs });
    } else {
        safeBufPrint(buf, "{d}s", .{seconds});
    }
}

fn safeBufPrint(buf: []u8, comptime fmt: []const u8, args: anytype) void {
    if (std.fmt.bufPrint(buf, fmt, args)) |_| {
        return;
    } else |_| {
        writeFallback(buf);
    }
}

fn writeFallback(buf: []u8) void {
    const fallback = "n/a";
    if (buf.len == 0) {
        return;
    }
    const len = @min(buf.len, fallback.len);
    std.mem.copyForwards(u8, buf[0..len], fallback[0..len]);
}

// ============================================================================
// Tests
// ============================================================================

test "downloader init and deinit" {
    var downloader = Downloader.init(std.testing.allocator);
    defer downloader.deinit();

    try std.testing.expect(!downloader.cancelled);
}

test "cancel download" {
    var downloader = Downloader.init(std.testing.allocator);
    defer downloader.deinit();

    downloader.cancel();
    try std.testing.expect(downloader.cancelled);

    downloader.reset();
    try std.testing.expect(!downloader.cancelled);
}

test "isValidUrl" {
    try std.testing.expect(isValidUrl("https://example.com/file.gguf"));
    try std.testing.expect(isValidUrl("http://localhost:8080/model.bin"));
    try std.testing.expect(!isValidUrl("ftp://example.com/file"));
    try std.testing.expect(!isValidUrl("example.com"));
    try std.testing.expect(!isValidUrl(""));
}

test "extractFilenameFromUrl" {
    const name1 = try extractFilenameFromUrl(std.testing.allocator, "https://example.com/path/model.gguf");
    defer std.testing.allocator.free(name1);
    try std.testing.expectEqualStrings("model.gguf", name1);

    const name2 = try extractFilenameFromUrl(std.testing.allocator, "https://example.com/file.bin?token=abc");
    defer std.testing.allocator.free(name2);
    try std.testing.expectEqualStrings("file.bin", name2);
}

test "parseUrl" {
    const url1 = Downloader.parseUrl("https://example.com/path/to/file");
    try std.testing.expect(url1 != null);
    try std.testing.expectEqual(UrlComponents.Scheme.https, url1.?.scheme);
    try std.testing.expectEqualStrings("example.com", url1.?.host);
    try std.testing.expectEqual(@as(u16, 443), url1.?.port);
    try std.testing.expectEqualStrings("/path/to/file", url1.?.path);

    const url2 = Downloader.parseUrl("http://localhost:8080/model");
    try std.testing.expect(url2 != null);
    try std.testing.expectEqual(UrlComponents.Scheme.http, url2.?.scheme);
    try std.testing.expectEqualStrings("localhost", url2.?.host);
    try std.testing.expectEqual(@as(u16, 8080), url2.?.port);
}

test "formatBytes" {
    const kb = formatBytes(1024);
    try std.testing.expect(std.mem.indexOf(u8, &kb, "1.00 KB") != null);

    const mb = formatBytes(1024 * 1024);
    try std.testing.expect(std.mem.indexOf(u8, &mb, "1.00 MB") != null);

    const gb = formatBytes(1024 * 1024 * 1024);
    try std.testing.expect(std.mem.indexOf(u8, &gb, "1.00 GB") != null);
}

test "formatDuration" {
    const secs = formatDuration(45);
    try std.testing.expect(std.mem.indexOf(u8, &secs, "45s") != null);

    const mins = formatDuration(125);
    try std.testing.expect(std.mem.indexOf(u8, &mins, "2m 5s") != null);

    const hours = formatDuration(3665);
    try std.testing.expect(std.mem.indexOf(u8, &hours, "1h 1m") != null);
}

test "formatRangeHeader uses caller buffer" {
    var buf_one: [32]u8 = undefined;
    var buf_two: [32]u8 = undefined;

    const header_one = Downloader.formatRangeHeader(128, &buf_one);
    const header_two = Downloader.formatRangeHeader(4096, &buf_two);

    try std.testing.expectEqualStrings("bytes=128-", header_one);
    try std.testing.expectEqualStrings("bytes=4096-", header_two);
}

test {
    std.testing.refAllDecls(@This());
}
