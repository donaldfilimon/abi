const std = @import("std");

pub const Dialect = enum {
    affine,
    linalg,
    tensor,
    gpu,
};

pub const ModuleSpec = struct {
    name: []const u8,
    dialect: Dialect = .linalg,
    operations: []const []const u8 = &.{},
};

pub const LoweringResult = struct {
    module_name: []const u8,
    dialect: Dialect,
    target_backend: []const u8,
    ir: []u8,

    pub fn deinit(self: LoweringResult, allocator: std.mem.Allocator) void {
        allocator.free(self.ir);
    }
};

pub const ToolchainStatus = struct {
    available: bool,
    backend: []const u8,
    message: []const u8,
};

pub const ModuleAnalysis = struct {
    module_name: []const u8,
    dialect: Dialect,
    operation_count: usize,
    checksum: u64,
};

pub fn dialectName(dialect: Dialect) []const u8 {
    return switch (dialect) {
        .affine => "affine",
        .linalg => "linalg",
        .tensor => "tensor",
        .gpu => "gpu",
    };
}

pub fn toolchainStatus() ToolchainStatus {
    return .{
        .available = false,
        .backend = "disabled",
        .message = "mlir feature is disabled",
    };
}

test {
    std.testing.refAllDecls(@This());
}

pub fn lower(allocator: std.mem.Allocator, spec: ModuleSpec) !LoweringResult {
    const analysis = try analyze(spec);
    return .{
        .module_name = analysis.module_name,
        .dialect = analysis.dialect,
        .target_backend = "disabled",
        .ir = try allocator.dupe(u8, "mlir feature is disabled"),
    };
}

pub fn analyze(spec: ModuleSpec) !ModuleAnalysis {
    if (spec.name.len == 0) return error.InvalidValue;
    if (std.mem.indexOfScalar(u8, spec.name, 0) != null) return error.InvalidValue;
    if (!isSymbolName(spec.name)) return error.InvalidMlirModuleName;

    var hash = std.hash.Wyhash.init(0);
    hash.update(spec.name);
    hash.update(dialectName(spec.dialect));
    for (spec.operations) |op| {
        if (op.len == 0) return error.InvalidMlirOperation;
        if (std.mem.indexOfScalar(u8, op, 0) != null) return error.InvalidMlirOperation;
        hash.update(op);
        hash.update(&.{0xff});
    }

    return .{
        .module_name = spec.name,
        .dialect = spec.dialect,
        .operation_count = spec.operations.len,
        .checksum = hash.final(),
    };
}

fn isSymbolName(value: []const u8) bool {
    for (value) |byte| {
        switch (byte) {
            'a'...'z', 'A'...'Z', '0'...'9', '_', '-', '.' => {},
            else => return false,
        }
    }
    return true;
}

test "mlir stub validates module shape before disabled lowering" {
    try std.testing.expectError(error.InvalidValue, lower(std.testing.allocator, .{ .name = "" }));
    try std.testing.expectError(error.InvalidMlirModuleName, lower(std.testing.allocator, .{ .name = "bad name" }));
    try std.testing.expectError(error.InvalidMlirOperation, lower(std.testing.allocator, .{ .name = "train", .operations = &.{""} }));
    const result = try lower(std.testing.allocator, .{ .name = "train", .operations = &.{"matmul"} });
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("disabled", result.target_backend);
    const analysis = try analyze(.{ .name = "train", .operations = &.{"matmul"} });
    try std.testing.expectEqual(@as(usize, 1), analysis.operation_count);
    try std.testing.expect(analysis.checksum != 0);
}
