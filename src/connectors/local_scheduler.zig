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
