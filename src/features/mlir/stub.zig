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
    return .{
        .module_name = spec.name,
        .dialect = spec.dialect,
        .target_backend = "disabled",
        .ir = try allocator.dupe(u8, "mlir feature is disabled"),
    };
}
