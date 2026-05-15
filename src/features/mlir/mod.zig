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

pub fn dialectName(dialect: Dialect) []const u8 {
    return switch (dialect) {
        .affine => "affine",
        .linalg => "linalg",
        .tensor => "tensor",
        .gpu => "gpu",
    };
}

pub fn lower(allocator: std.mem.Allocator, spec: ModuleSpec) !LoweringResult {
    if (spec.name.len == 0) return error.InvalidMlirModule;

    const ir = try std.fmt.allocPrint(
        allocator,
        "module @{s} {{ // dialect={s}, operations={d}, backend=cpu-simulated }}",
        .{ spec.name, dialectName(spec.dialect), spec.operations.len },
    );

    return .{
        .module_name = spec.name,
        .dialect = spec.dialect,
        .target_backend = "cpu-simulated",
        .ir = ir,
    };
}

test "mlir lowering emits scaffold IR" {
    const result = try lower(std.testing.allocator, .{ .name = "train", .operations = &.{"matmul"} });
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(std.mem.indexOf(u8, result.ir, "module @train") != null);
}
