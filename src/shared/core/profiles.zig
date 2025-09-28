//! Runtime profile definitions for ABI.
//!
//! Profiles gate optional behaviour such as streaming, function calling, and
//! logging sinks so that development, testing, and production environments stay
//! reproducible.

const std = @import("std");

pub const ProfileKind = enum { dev, test, prod };

pub const ProfileConfig = struct {
    kind: ProfileKind,
    enable_streaming: bool,
    enable_function_calling: bool,
    enable_structured_logging: bool,
    log_to_console: bool,
    log_to_file: bool,
    metrics_enabled: bool,
    default_log_path: []const u8,

    pub fn describe(self: ProfileConfig, writer: anytype) !void {
        try writer.print(
            "profile={s} streaming={} functions={} metrics={} log_console={} log_file={}",
            .{
                kindToString(self.kind),
                self.enable_streaming,
                self.enable_function_calling,
                self.metrics_enabled,
                self.log_to_console,
                self.log_to_file,
            },
        );
    }
};

pub fn resolve(kind: ProfileKind) ProfileConfig {
    return switch (kind) {
        .dev => .{
            .kind = .dev,
            .enable_streaming = true,
            .enable_function_calling = true,
            .enable_structured_logging = true,
            .log_to_console = true,
            .log_to_file = false,
            .metrics_enabled = true,
            .default_log_path = "logs/dev.jsonl",
        },
        .test => .{
            .kind = .test,
            .enable_streaming = false,
            .enable_function_calling = false,
            .enable_structured_logging = true,
            .log_to_console = false,
            .log_to_file = true,
            .metrics_enabled = true,
            .default_log_path = "logs/test.jsonl",
        },
        .prod => .{
            .kind = .prod,
            .enable_streaming = true,
            .enable_function_calling = true,
            .enable_structured_logging = true,
            .log_to_console = true,
            .log_to_file = true,
            .metrics_enabled = true,
            .default_log_path = "/var/log/abi/runtime.jsonl",
        },
    };
}

pub fn kindFromString(value: []const u8) ?ProfileKind {
    if (std.ascii.eqlIgnoreCase(value, "dev")) return .dev;
    if (std.ascii.eqlIgnoreCase(value, "test")) return .test;
    if (std.ascii.eqlIgnoreCase(value, "prod")) return .prod;
    if (std.ascii.eqlIgnoreCase(value, "production")) return .prod;
    return null;
}

pub fn kindToString(kind: ProfileKind) []const u8 {
    return switch (kind) {
        .dev => "dev",
        .test => "test",
        .prod => "prod",
    };
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "profile resolution" {
    const dev = resolve(.dev);
    try std.testing.expect(dev.enable_streaming);
    try std.testing.expect(dev.log_to_console);
    try std.testing.expect(!dev.log_to_file);

    const prod = resolve(.prod);
    try std.testing.expect(prod.log_to_console);
    try std.testing.expect(prod.log_to_file);
    try std.testing.expectEqualStrings("/var/log/abi/runtime.jsonl", prod.default_log_path);
}

