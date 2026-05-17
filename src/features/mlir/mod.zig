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
        .backend = "textual-local",
        .message = "external MLIR tools are not linked; textual ABI IR lowering is active",
    };
}

pub fn lower(allocator: std.mem.Allocator, spec: ModuleSpec) !LoweringResult {
    if (spec.name.len == 0) return error.InvalidMlirModule;

    var ir_buf = std.ArrayListUnmanaged(u8).empty;
    errdefer ir_buf.deinit(allocator);
    try ir_buf.print(allocator, "module @{s} attributes {{abi.dialect = \"{s}\"}} {{\n", .{ spec.name, dialectName(spec.dialect) });
    if (spec.operations.len == 0) {
        try ir_buf.appendSlice(allocator, "  // no operations requested\n");
    } else {
        for (spec.operations, 0..) |op, i| {
            if (op.len == 0) return error.InvalidMlirOperation;
            try ir_buf.print(allocator, "  abi.op @{s}_{d} attributes {{abi.name = \"{s}\"}}\n", .{ dialectName(spec.dialect), i, op });
        }
    }
    try ir_buf.appendSlice(allocator, "}\n");

    return .{
        .module_name = spec.name,
        .dialect = spec.dialect,
        .target_backend = toolchainStatus().backend,
        .ir = try ir_buf.toOwnedSlice(allocator),
    };
}

test "mlir lowering emits structured textual IR" {
    const result = try lower(std.testing.allocator, .{ .name = "train", .operations = &.{"matmul"} });
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "module @train") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "abi.name = \"matmul\"") != null);
    try std.testing.expect(!toolchainStatus().available);
}
