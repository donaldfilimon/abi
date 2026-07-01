const std = @import("std");

pub const name = "telemetry-exporter";
pub const description = "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.";
pub const version = "0.1.0";
pub const target_feature = "telemetry";

pub fn register() void {}

/// Future execution entry point. Formats the input as a telemetry event line;
/// real exporters will receive the telemetry sink in a later iteration.
pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return try std.fmt.allocPrint(allocator, "telemetry-exporter event (bytes={d})", .{input.len});
}
