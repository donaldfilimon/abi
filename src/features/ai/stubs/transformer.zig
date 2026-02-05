const std = @import("std");

pub const TransformerConfig = struct {};
pub const TransformerModel = struct {
    pub fn init(_: TransformerConfig) TransformerModel {
        return .{};
    }
};
