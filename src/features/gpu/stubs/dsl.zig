const std = @import("std");

pub const dsl = struct {};
pub const ScalarType = enum { f32, f64, i32, i64, u32, u64 };
pub const VectorType = struct {};
pub const MatrixType = struct {};
pub const AddressSpace = enum { global, local, private };
pub const DslType = struct {};
pub const AccessMode = enum { read, write, read_write };
pub const Expr = struct {};
pub const BinaryOp = enum { add, sub, mul, div };
pub const UnaryOp = enum { neg, abs };
pub const BuiltinFn = enum {};
pub const BuiltinVar = enum {};
pub const Stmt = struct {};
pub const GeneratedSource = struct {};
pub const CompileOptions = struct {};

test {
    std.testing.refAllDecls(@This());
}
