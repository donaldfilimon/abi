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

pub const ValidationReport = struct {
    name: []const u8,
    language: Language,
    entry_point: []const u8,
    source_bytes: usize,
    checksum: u64,
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
    _ = try validateDetailed(source);
}

pub fn validateDetailed(source: ShaderSource) !ValidationReport {
    if (source.name.len == 0) return error.InvalidShaderName;
    if (source.source.len == 0) return error.InvalidShaderSource;
    if (std.mem.indexOfScalar(u8, source.name, 0) != null) return error.InvalidShaderName;
    if (std.mem.indexOfScalar(u8, source.source, 0) != null) return error.InvalidShaderSource;
    try validateBalancedDelimiters(source.source);
    const entry_point = try detectEntryPoint(source.language, source.source);
    return .{
        .name = source.name,
        .language = source.language,
        .entry_point = entry_point,
        .source_bytes = source.source.len,
        .checksum = shaderChecksum(source.source),
    };
}

test {
    std.testing.refAllDecls(@This());
}

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    const report = try validateDetailed(source);
    return .{
        .name = report.name,
        .language = report.language,
        .entry_point = report.entry_point,
        .backend = "disabled",
        .bytes = try allocator.dupe(u8, "shader feature is disabled"),
    };
}

fn detectEntryPoint(language: Language, source: []const u8) ![]const u8 {
    return switch (language) {
        .zig_kernel, .wgsl => if (std.mem.indexOf(u8, source, "fn main") != null) "main" else error.MissingShaderEntryPoint,
        .msl => if (std.mem.indexOf(u8, source, "kernel") != null) "kernel" else if (std.mem.indexOf(u8, source, "main") != null) "main" else error.MissingShaderEntryPoint,
        .spirv_text => if (std.mem.indexOf(u8, source, "OpEntryPoint") != null) "OpEntryPoint" else error.MissingShaderEntryPoint,
    };
}

fn validateBalancedDelimiters(source: []const u8) !void {
    var braces: isize = 0;
    var parens: isize = 0;
    var brackets: isize = 0;
    for (source) |byte| {
        switch (byte) {
            '{' => braces += 1,
            '}' => {
                braces -= 1;
                if (braces < 0) return error.UnbalancedShaderDelimiters;
            },
            '(' => parens += 1,
            ')' => {
                parens -= 1;
                if (parens < 0) return error.UnbalancedShaderDelimiters;
            },
            '[' => brackets += 1,
            ']' => {
                brackets -= 1;
                if (brackets < 0) return error.UnbalancedShaderDelimiters;
            },
            else => {},
        }
    }
    if (braces != 0 or parens != 0 or brackets != 0) return error.UnbalancedShaderDelimiters;
}

fn shaderChecksum(source: []const u8) u64 {
    var hash = std.hash.Wyhash.init(0);
    hash.update(source);
    return hash.final();
}

test "shader stub mirrors validation before disabled artifact" {
    try std.testing.expectError(error.InvalidShaderName, validate(.{ .name = "", .source = "fn main() void {}" }));
    try std.testing.expectError(error.MissingShaderEntryPoint, validate(.{ .name = "copy", .source = "fn helper() void {}" }));
    try std.testing.expectError(error.UnbalancedShaderDelimiters, validate(.{ .name = "copy", .source = "fn main() void {" }));
    const artifact = try compile(std.testing.allocator, .{ .name = "copy", .source = "fn main() void {}" });
    defer artifact.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("disabled", artifact.backend);
    try std.testing.expectEqualStrings("main", artifact.entry_point);
    const status = compilerStatus();
    try std.testing.expect(!status.available);
    try std.testing.expectEqualStrings("disabled", status.backend);
}
