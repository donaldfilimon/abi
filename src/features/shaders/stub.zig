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
    _ = source;
}

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    return .{
        .name = source.name,
        .language = source.language,
        .entry_point = "main",
        .backend = "disabled",
        .bytes = try allocator.dupe(u8, "shader feature is disabled"),
    };
}
