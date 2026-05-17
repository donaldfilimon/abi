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
        .backend = "validated-local",
        .message = "external shader compiler toolchains are not linked; source validation artifact is active",
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

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    try validate(source);
    const bytes = try std.fmt.allocPrint(
        allocator,
        "shader={s};language={s};backend={s};entry=main;source_bytes={d};checksum={x}",
        .{ source.name, languageName(source.language), compilerStatus().backend, source.source.len, shaderChecksum(source.source) },
    );

    return .{
        .name = source.name,
        .language = source.language,
        .entry_point = "main",
        .backend = compilerStatus().backend,
        .bytes = bytes,
    };
}

fn shaderChecksum(source: []const u8) u64 {
    var hash = std.hash.Wyhash.init(0);
    hash.update(source);
    return hash.final();
}

test "shader compiler validates source" {
    try std.testing.expectError(error.InvalidShaderName, validate(.{ .name = "", .source = "fn main() void {}" }));
    try std.testing.expectError(error.MissingShaderEntryPoint, validate(.{ .name = "copy", .source = "fn helper() void {}" }));
    const artifact = try compile(std.testing.allocator, .{ .name = "copy", .source = "fn main() void {}" });
    defer artifact.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, artifact.bytes, "shader=copy") != null);
    try std.testing.expect(std.mem.indexOf(u8, artifact.bytes, "validated-local") != null);
    try std.testing.expect(!compilerStatus().available);
}
