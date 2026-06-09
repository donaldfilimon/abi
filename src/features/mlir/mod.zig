const std = @import("std");
const validation = @import("../../foundation/validation.zig");

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
        .backend = "textual-local",
        .message = "external MLIR tools are not linked; textual ABI IR lowering is active",
    };
}

test {
    std.testing.refAllDecls(@This());
}

pub fn lower(allocator: std.mem.Allocator, spec: ModuleSpec) !LoweringResult {
    const analysis = try analyze(spec);

    var ir_buf = std.ArrayListUnmanaged(u8).empty;
    errdefer ir_buf.deinit(allocator);
    try ir_buf.print(
        allocator,
        "module @{s} attributes {{abi.dialect = \"{s}\", abi.ops = {d}, abi.checksum = \"{x}\"}} {{\n",
        .{ analysis.module_name, dialectName(analysis.dialect), analysis.operation_count, analysis.checksum },
    );
    if (spec.operations.len == 0) {
        try ir_buf.appendSlice(allocator, "  // no operations requested\n");
    } else {
        for (spec.operations, 0..) |op, i| {
            try ir_buf.print(allocator, "  abi.op @{s}_{d} attributes {{abi.name = ", .{ dialectName(spec.dialect), i });
            try appendMlirString(&ir_buf, allocator, op);
            try ir_buf.appendSlice(allocator, "}\n");
        }
    }
    try ir_buf.appendSlice(allocator, "}\n");

    return .{
        .module_name = analysis.module_name,
        .dialect = analysis.dialect,
        .target_backend = toolchainStatus().backend,
        .ir = try ir_buf.toOwnedSlice(allocator),
    };
}

pub fn analyze(spec: ModuleSpec) !ModuleAnalysis {
    validation.validateNonEmptySlice(spec.name) catch return error.InvalidValue;
    validation.validateNoNullBytes(spec.name) catch return error.InvalidValue;
    if (!isSymbolName(spec.name)) return error.InvalidMlirModuleName;

    var hash = std.hash.Wyhash.init(0);
    hash.update(spec.name);
    hash.update(dialectName(spec.dialect));
    for (spec.operations) |op| {
        validation.validateNonEmptySlice(op) catch return error.InvalidMlirOperation;
        validation.validateNoNullBytes(op) catch return error.InvalidMlirOperation;
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

fn appendMlirString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\22"),
            '\\' => try out.appendSlice(allocator, "\\5C"),
            '\n' => try out.appendSlice(allocator, "\\0A"),
            '\r' => try out.appendSlice(allocator, "\\0D"),
            '\t' => try out.appendSlice(allocator, "\\09"),
            0x00...0x08, 0x0b...0x0c, 0x0e...0x1f => try out.print(allocator, "\\{X:0>2}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

test "mlir lowering emits structured textual IR" {
    const result = try lower(std.testing.allocator, .{ .name = "train", .operations = &.{"matmul"} });
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "module @train") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "abi.name = \"matmul\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "abi.ops = 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "abi.checksum = \"") != null);
    try std.testing.expect(!toolchainStatus().available);
}

test "mlir lowering validates symbols and escapes operation attributes" {
    try std.testing.expectError(error.InvalidMlirModuleName, analyze(.{ .name = "bad name" }));
    try std.testing.expectError(error.InvalidMlirOperation, analyze(.{ .name = "train", .operations = &.{""} }));

    const result = try lower(std.testing.allocator, .{ .name = "train", .operations = &.{"quote \" op"} });
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "quote \\22 op") != null);
    const analysis = try analyze(.{ .name = "train", .operations = &.{"quote \" op"} });
    try std.testing.expectEqual(@as(usize, 1), analysis.operation_count);
    try std.testing.expect(analysis.checksum != 0);
}
