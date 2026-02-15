//! Local Scheduler Connector
//!
//! Provides integration with a local job scheduler service for distributed
//! compute tasks. The scheduler accepts job submissions and manages task
//! execution across available workers.
//!
//! ## Configuration
//!
//! Set the scheduler URL via environment variables:
//! - `ABI_LOCAL_SCHEDULER_URL` (preferred)
//! - `LOCAL_SCHEDULER_URL` (fallback)
//!
//! Default: `http://127.0.0.1:8080`
//!
//! ## Example
//!
//! ```zig
//! const config = try local_scheduler.loadFromEnv(allocator);
//! defer config.deinit(allocator);
//!
//! const health_url = try config.healthUrl(allocator);
//! defer allocator.free(health_url);
//! ```

const std = @import("std");

const connectors = @import("mod.zig");

pub const Config = struct {
    url: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.url);
        self.* = undefined;
    }

    pub fn endpoint(self: *const Config, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        const prefix = if (path.len > 0 and path[0] == '/') "" else "/";
        return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ self.url, prefix, path });
    }

    pub fn healthUrl(self: *const Config, allocator: std.mem.Allocator) ![]u8 {
        return self.endpoint(allocator, "/health");
    }

    pub fn submitUrl(self: *const Config, allocator: std.mem.Allocator) ![]u8 {
        return self.endpoint(allocator, "/jobs/submit");
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LOCAL_SCHEDULER_URL",
        "LOCAL_SCHEDULER_URL",
    })) orelse try allocator.dupe(u8, "http://127.0.0.1:9090");
    return .{ .url = url };
}

test "local scheduler endpoint join" {
    var config = Config{ .url = try std.testing.allocator.dupe(u8, "http://localhost:9090") };
    defer config.deinit(std.testing.allocator);

    const url = try config.submitUrl(std.testing.allocator);
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings("http://localhost:9090/jobs/submit", url);
}

test "local scheduler health url" {
    const allocator = std.testing.allocator;
    var config = Config{ .url = try allocator.dupe(u8, "http://scheduler.local:8080") };
    defer config.deinit(allocator);

    const url = try config.healthUrl(allocator);
    defer allocator.free(url);
    try std.testing.expectEqualStrings("http://scheduler.local:8080/health", url);
}

test "local scheduler custom endpoint" {
    const allocator = std.testing.allocator;
    var config = Config{ .url = try allocator.dupe(u8, "http://localhost:9090") };
    defer config.deinit(allocator);

    // With leading slash
    const url1 = try config.endpoint(allocator, "/api/v1/jobs");
    defer allocator.free(url1);
    try std.testing.expectEqualStrings("http://localhost:9090/api/v1/jobs", url1);

    // Without leading slash — should add one
    const url2 = try config.endpoint(allocator, "status");
    defer allocator.free(url2);
    try std.testing.expectEqualStrings("http://localhost:9090/status", url2);
}

test "local scheduler config lifecycle" {
    const allocator = std.testing.allocator;
    var config = Config{ .url = try allocator.dupe(u8, "http://custom:5555") };
    // deinit should free without leaks
    config.deinit(allocator);
}

test "local scheduler empty path endpoint" {
    const allocator = std.testing.allocator;
    var config = Config{ .url = try allocator.dupe(u8, "http://localhost:9090") };
    defer config.deinit(allocator);

    const url = try config.endpoint(allocator, "");
    defer allocator.free(url);
    // Empty path should produce just the base URL with prefix
    try std.testing.expectEqualStrings("http://localhost:9090/", url);
}
