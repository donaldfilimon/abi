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

pub const CompilerStatus = struct {
    available: bool,
    backend: []const u8,
    message: []const u8,
};

pub fn languageName(language: Language) []const u8 {
    return switch (language) {
        .zig_kernel => "zig-kernel",
        .wgsl => "wgsl",
        .msl => "msl",
        .spirv_text => "spirv-text",
    };
}

pub fn compilerStatus() CompilerStatus {
    return .{
        .available = false,
        .backend = "disabled",
        .message = "shader feature is disabled",
    };
}

pub fn validate(source: ShaderSource) !void {
    if (source.name.len == 0) return error.InvalidShaderName;
    if (source.source.len == 0) return error.InvalidShaderSource;
    if (std.mem.indexOfScalar(u8, source.name, 0) != null) return error.InvalidShaderName;
    if (std.mem.indexOfScalar(u8, source.source, 0) != null) return error.InvalidShaderSource;
    switch (source.language) {
        .zig_kernel, .wgsl => if (std.mem.indexOf(u8, source.source, "fn main") == null) return error.MissingShaderEntryPoint,
        .msl => if (std.mem.indexOf(u8, source.source, "kernel") == null and std.mem.indexOf(u8, source.source, "main") == null) return error.MissingShaderEntryPoint,
        .spirv_text => if (std.mem.indexOf(u8, source.source, "OpEntryPoint") == null) return error.MissingShaderEntryPoint,
    }
}

test {
    std.testing.refAllDecls(@This());
}

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    try validate(source);
    return .{
        .name = source.name,
        .language = source.language,
        .entry_point = "main",
        .backend = "disabled",
        .bytes = try allocator.dupe(u8, "shader feature is disabled"),
    };
}

test "shader stub mirrors validation before disabled artifact" {
    try std.testing.expectError(error.InvalidShaderName, validate(.{ .name = "", .source = "fn main() void {}" }));
    try std.testing.expectError(error.MissingShaderEntryPoint, validate(.{ .name = "copy", .source = "fn helper() void {}" }));
    const artifact = try compile(std.testing.allocator, .{ .name = "copy", .source = "fn main() void {}" });
    defer artifact.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("disabled", artifact.backend);
}
