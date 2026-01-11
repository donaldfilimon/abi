//! Argument parsing utilities for CLI commands.

const std = @import("std");

/// Check if text matches any of the provided options.
pub fn matchesAny(text: []const u8, options: []const []const u8) bool {
    for (options) |option| {
        if (std.mem.eql(u8, text, option)) return true;
    }
    return false;
}

/// Parse a node status string to enum value.
pub fn parseNodeStatus(text: []const u8) ?@import("abi").network.NodeStatus {
    if (std.ascii.eqlIgnoreCase(text, "healthy")) return .healthy;
    if (std.ascii.eqlIgnoreCase(text, "degraded")) return .degraded;
    if (std.ascii.eqlIgnoreCase(text, "offline")) return .offline;
    return null;
}

test "matchesAny helper function" {
    try std.testing.expect(matchesAny("help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("--help", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("-h", &.{ "help", "--help", "-h" }));
    try std.testing.expect(!matchesAny("invalid", &[_][]const u8{ "help", "--help", "-h" }));
    try std.testing.expect(matchesAny("test", &[_][]const u8{"test"}));
    try std.testing.expect(!matchesAny("test", &[_][]const u8{"other"}));
}
