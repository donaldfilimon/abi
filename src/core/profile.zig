//! Profile configuration utilities for the ABI framework.
//!
//! Provides strongly typed environments (dev/test/prod) with toggles that
//! control runtime features such as streaming responses, structured logging,
//! and function calling. The design keeps overrides merge-friendly and avoids
//! implicit global state so higher layers can compose behaviour per profile.

const std = @import("std");
const ArrayList = std.array_list.Managed;

const TestError = error{InvalidTestValue};

pub const Allocator = std.mem.Allocator;

/// Logging sink selection used by profiles and persona manifests.
pub const LoggingSink = enum {
    stdout,
    stderr,
    file,
};

/// Named profile environments the runtime can operate under.
pub const ProfileKind = enum {
    dev,
    testing,
    prod,
};

/// Partial override configuration so callers can selectively patch values
/// without rebuilding the entire profile struct.
pub const PartialProfileConfig = struct {
    enable_streaming: ?bool = null,
    enable_function_calling: ?bool = null,
    structured_logging: ?bool = null,
    logging_sink: ?LoggingSink = null,
    metrics_enabled: ?bool = null,
    log_to_console: ?bool = null,
    log_to_file: ?bool = null,
    default_log_path: ?[]const u8 = null,
    max_retries: ?u8 = null,
    base_backoff_ms: ?u32 = null,
    max_backoff_ms: ?u32 = null,
};

/// Fully realised profile configuration capturing runtime toggles.
pub const ProfileConfig = struct {
    kind: ProfileKind,
    enable_streaming: bool,
    enable_function_calling: bool,
    structured_logging: bool,
    logging_sink: LoggingSink,
    metrics_enabled: bool,
    log_to_console: bool,
    log_to_file: bool,
    default_log_path: []const u8,
    max_retries: u8,
    base_backoff_ms: u32,
    max_backoff_ms: u32,

    pub fn defaults(kind: ProfileKind) ProfileConfig {
        return switch (kind) {
            .dev => .{
                .kind = .dev,
                .enable_streaming = true,
                .enable_function_calling = true,
                .structured_logging = true,
                .logging_sink = .stdout,
                .metrics_enabled = true,
                .log_to_console = true,
                .log_to_file = false,
                .default_log_path = "logs/dev.jsonl",
                .max_retries = 3,
                .base_backoff_ms = 250,
                .max_backoff_ms = 2_000,
            },
            .testing => .{
                .kind = .testing,
                .enable_streaming = false,
                .enable_function_calling = true,
                .structured_logging = true,
                .logging_sink = .stderr,
                .metrics_enabled = true,
                .log_to_console = false,
                .log_to_file = true,
                .default_log_path = "logs/testing.jsonl",
                .max_retries = 1,
                .base_backoff_ms = 100,
                .max_backoff_ms = 500,
            },
            .prod => .{
                .kind = .prod,
                .enable_streaming = true,
                .enable_function_calling = true,
                .structured_logging = true,
                .logging_sink = .file,
                .metrics_enabled = true,
                .log_to_console = true,
                .log_to_file = true,
                .default_log_path = "/var/log/abi/runtime.jsonl",
                .max_retries = 5,
                .base_backoff_ms = 500,
                .max_backoff_ms = 8_000,
            },
        };
    }

    pub fn applyOverrides(self: ProfileConfig, overrides: PartialProfileConfig) ProfileConfig {
        return .{
            .kind = self.kind,
            .enable_streaming = overrides.enable_streaming orelse self.enable_streaming,
            .enable_function_calling = overrides.enable_function_calling orelse self.enable_function_calling,
            .structured_logging = overrides.structured_logging orelse self.structured_logging,
            .logging_sink = overrides.logging_sink orelse self.logging_sink,
            .metrics_enabled = overrides.metrics_enabled orelse self.metrics_enabled,
            .log_to_console = overrides.log_to_console orelse self.log_to_console,
            .log_to_file = overrides.log_to_file orelse self.log_to_file,
            .default_log_path = overrides.default_log_path orelse self.default_log_path,
            .max_retries = overrides.max_retries orelse self.max_retries,
            .base_backoff_ms = overrides.base_backoff_ms orelse self.base_backoff_ms,
            .max_backoff_ms = overrides.max_backoff_ms orelse self.max_backoff_ms,
        };
    }

    pub fn describe(self: ProfileConfig, writer: anytype) !void {
        try writer.print(
            "profile={s} streaming={} functions={} metrics={} log_console={} log_file={} sink={s}",
            .{
                kindToString(self.kind),
                self.enable_streaming,
                self.enable_function_calling,
                self.metrics_enabled,
                self.log_to_console,
                self.log_to_file,
                @tagName(self.logging_sink),
            },
        );
    }
};

pub fn parseProfileKind(name: []const u8) ?ProfileKind {
    if (std.ascii.eqlIgnoreCase(name, "dev")) return .dev;
    if (std.ascii.eqlIgnoreCase(name, "development")) return .dev;
    if (std.ascii.eqlIgnoreCase(name, "test")) return .testing;
    if (std.ascii.eqlIgnoreCase(name, "testing")) return .testing;
    if (std.ascii.eqlIgnoreCase(name, "prod")) return .prod;
    if (std.ascii.eqlIgnoreCase(name, "production")) return .prod;
    return null;
}

pub fn kindToString(kind: ProfileKind) []const u8 {
    return switch (kind) {
        .dev => "dev",
        .testing => "testing",
        .prod => "prod",
    };
}

pub fn kindFromString(name: []const u8) ?ProfileKind {
    return parseProfileKind(name);
}

pub fn resolve(kind: ProfileKind) ProfileConfig {
    return ProfileConfig.defaults(kind);
}

fn expectProfile(kind: ProfileKind, overrides: PartialProfileConfig, expected: ProfileConfig) !void {
    const testing = std.testing;
    const base = ProfileConfig.defaults(kind);
    const merged = base.applyOverrides(overrides);
    try testing.expectEqual(expected.enable_streaming, merged.enable_streaming);
    try testing.expectEqual(expected.enable_function_calling, merged.enable_function_calling);
    try testing.expectEqual(expected.structured_logging, merged.structured_logging);
    try testing.expectEqual(expected.logging_sink, merged.logging_sink);
    try testing.expectEqual(expected.metrics_enabled, merged.metrics_enabled);
    try testing.expectEqual(expected.log_to_console, merged.log_to_console);
    try testing.expectEqual(expected.log_to_file, merged.log_to_file);
    try testing.expectEqualStrings(expected.default_log_path, merged.default_log_path);
    try testing.expectEqual(expected.max_retries, merged.max_retries);
    try testing.expectEqual(expected.base_backoff_ms, merged.base_backoff_ms);
    try testing.expectEqual(expected.max_backoff_ms, merged.max_backoff_ms);
}

test "profile defaults and overrides" {
    try expectProfile(.dev, .{}, ProfileConfig.defaults(.dev));
    try expectProfile(.testing, .{ .enable_streaming = true, .logging_sink = .stdout }, .{
        .kind = .testing,
        .enable_streaming = true,
        .enable_function_calling = true,
        .structured_logging = true,
        .logging_sink = .stdout,
        .metrics_enabled = true,
        .log_to_console = false,
        .log_to_file = true,
        .default_log_path = "logs/testing.jsonl",
        .max_retries = 1,
        .base_backoff_ms = 100,
        .max_backoff_ms = 500,
    });
}

test "parse profile kind" {
    const testing = std.testing;
    try testing.expectEqual(ProfileKind.dev, parseProfileKind("DEV") orelse return TestError.InvalidTestValue);
    try testing.expectEqual(ProfileKind.testing, parseProfileKind("test") orelse return TestError.InvalidTestValue);
    try testing.expectEqual(ProfileKind.testing, parseProfileKind("testing") orelse return TestError.InvalidTestValue);
    try testing.expectEqual(ProfileKind.prod, parseProfileKind("production") orelse return TestError.InvalidTestValue);
    try testing.expectEqual(@as(?ProfileKind, null), parseProfileKind("unknown"));
}

test "profile resolve matches defaults" {
    const testing = std.testing;
    const dev = resolve(.dev);
    try testing.expectEqual(ProfileKind.dev, dev.kind);
    try testing.expect(dev.log_to_console);
    try testing.expect(!dev.log_to_file);

    const prod = resolve(.prod);
    try testing.expectEqualStrings("/var/log/abi/runtime.jsonl", prod.default_log_path);
}

test "profile describe output" {
    var buffer = ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();

    const profile = ProfileConfig.defaults(.dev);
    try profile.describe(buffer.writer());
    const text = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, text, "profile=dev") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "streaming=true") != null);
}
