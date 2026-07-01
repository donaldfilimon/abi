const std = @import("std");

pub const name = "telemetry-exporter";
pub const description = "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.";
pub const version = "0.1.0";
pub const target_feature = "telemetry";

pub fn register() void {}

pub fn run(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    _ = allocator;
    _ = input;
    return error.FeatureDisabled;
}
