const std = @import("std");
const serializer_mod = @import("serializer.zig");

pub const serializer = serializer_mod;

pub const FormatVersion = serializer.FormatVersion;
pub const ModelMetadata = serializer.ModelMetadata;
pub const ModelSerializer = serializer.ModelSerializer;

test {
    std.testing.refAllDecls(@This());
}
