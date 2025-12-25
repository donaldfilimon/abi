const std = @import("std");

pub const TransformerConfig = struct {
    layers: u16 = 4,
    hidden_size: u16 = 256,
};

pub const TransformerModel = struct {
    config: TransformerConfig,

    pub fn init(config: TransformerConfig) TransformerModel {
        return .{ .config = config };
    }

    pub fn infer(_: *TransformerModel, allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        return std.fmt.allocPrint(allocator, "transformer({s})", .{input});
    }
};
