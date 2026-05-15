const std = @import("std");

pub const Language = enum {
    zig_kernel,
    wgsl,
    msl,
    spirv_text,
};

pub const ShaderSource = struct {
    name: []const u8,
    language: Language = .zig_kernel,
    source: []const u8,
};

pub const ShaderArtifact = struct {
    name: []const u8,
    language: Language,
    entry_point: []const u8,
    backend: []const u8,
    bytes: []u8,

    pub fn deinit(self: ShaderArtifact, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
    }
};

pub fn languageName(language: Language) []const u8 {
    return switch (language) {
        .zig_kernel => "zig-kernel",
        .wgsl => "wgsl",
        .msl => "msl",
        .spirv_text => "spirv-text",
    };
}

pub fn validate(source: ShaderSource) !void {
    if (source.name.len == 0) return error.InvalidShaderName;
    if (source.source.len == 0) return error.InvalidShaderSource;
    if (std.mem.indexOfScalar(u8, source.name, 0) != null) return error.InvalidShaderName;
    if (std.mem.indexOfScalar(u8, source.source, 0) != null) return error.InvalidShaderSource;
}

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    try validate(source);
    const bytes = try std.fmt.allocPrint(
        allocator,
        "shader={s};language={s};backend=simulated;source_bytes={d}",
        .{ source.name, languageName(source.language), source.source.len },
    );

    return .{
        .name = source.name,
        .language = source.language,
        .entry_point = "main",
        .backend = "simulated",
        .bytes = bytes,
    };
}

test "shader compiler validates source" {
    try std.testing.expectError(error.InvalidShaderName, validate(.{ .name = "", .source = "fn main() void {}" }));
    const artifact = try compile(std.testing.allocator, .{ .name = "copy", .source = "fn main() void {}" });
    defer artifact.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, artifact.bytes, "shader=copy") != null);
}
