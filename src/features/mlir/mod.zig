pub const Dialect = enum { llvm, linalg };
pub const ModuleSpec = struct { name: []const u8 };
pub const LoweringResult = struct { output: []const u8 };

pub const dialectName = "abi-mlir";
pub fn lower(spec: ModuleSpec) !LoweringResult {
    _ = spec;
    return .{ .output = "" };
}
