const std = @import("std");
const serializer = @import("serializer.zig");

pub const FormatVersion = serializer.FormatVersion;
pub const ModelMetadata = serializer.ModelMetadata;
pub const ModelSerializer = serializer.ModelSerializer;

pub usingnamespace serializer;

test {
    std.testing.refAllDecls(@This());
}
