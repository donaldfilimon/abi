const std = @import("std");
const validation = @import("../../foundation/validation.zig");

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
        .backend = "validated-local",
        .message = "external shader compiler toolchains are not linked; source validation artifact is active",
    };
}

test {
    std.testing.refAllDecls(@This());
}

pub fn validate(source: ShaderSource) !void {
    _ = try validateDetailed(source);
}

pub fn validateDetailed(source: ShaderSource) !ValidationReport {
    validation.validateNonEmptySlice(source.name) catch return error.InvalidShaderName;
    validation.validateNonEmptySlice(source.source) catch return error.InvalidShaderSource;
    validation.validateNoNullBytes(source.name) catch return error.InvalidShaderName;
    validation.validateNoNullBytes(source.source) catch return error.InvalidShaderSource;
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

pub fn compile(allocator: std.mem.Allocator, source: ShaderSource) !ShaderArtifact {
    const report = try validateDetailed(source);
    const bytes = try std.fmt.allocPrint(
        allocator,
        "shader={s};language={s};backend={s};entry={s};source_bytes={d};checksum={x}",
        .{ report.name, languageName(report.language), compilerStatus().backend, report.entry_point, report.source_bytes, report.checksum },
    );

    return .{
        .name = report.name,
        .language = report.language,
        .entry_point = report.entry_point,
        .backend = compilerStatus().backend,
        .bytes = bytes,
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

test "shader compiler validates source" {
    try std.testing.expectError(error.InvalidShaderName, validate(.{ .name = "", .source = "fn main() void {}" }));
    try std.testing.expectError(error.MissingShaderEntryPoint, validate(.{ .name = "copy", .source = "fn helper() void {}" }));
    try std.testing.expectError(error.UnbalancedShaderDelimiters, validate(.{ .name = "copy", .source = "fn main() void {" }));
    const artifact = try compile(std.testing.allocator, .{ .name = "copy", .source = "fn main() void {}" });
    defer artifact.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, artifact.bytes, "shader=copy") != null);
    try std.testing.expectEqualStrings("main", artifact.entry_point);
    try std.testing.expect(std.mem.indexOf(u8, artifact.bytes, "validated-local") != null);
    const status = compilerStatus();
    try std.testing.expect(!status.available);
    try std.testing.expect(std.mem.indexOf(u8, status.message, "not linked") != null);
}

test "shader compiler reports entry point and checksum by language" {
    const wgsl = try validateDetailed(.{ .name = "wgsl-copy", .language = .wgsl, .source = "@compute @workgroup_size(1) fn main() {}" });
    try std.testing.expectEqualStrings("main", wgsl.entry_point);
    try std.testing.expect(wgsl.checksum != 0);

    const msl = try validateDetailed(.{ .name = "msl-copy", .language = .msl, .source = "kernel void add() {}" });
    try std.testing.expectEqualStrings("kernel", msl.entry_point);

    const spirv = try validateDetailed(.{ .name = "spv-copy", .language = .spirv_text, .source = "OpEntryPoint GLCompute %main \"main\"" });
    try std.testing.expectEqualStrings("OpEntryPoint", spirv.entry_point);
}
