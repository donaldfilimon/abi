const dsl = @import("../dsl/mod.zig");
const std = @import("std");

pub const KernelBuilder = dsl.KernelBuilder;
pub const KernelIR = dsl.KernelIR;
pub const PortableKernelSource = dsl.PortableKernelSource;
pub const compile = dsl.compile;
pub const CompileOptions = dsl.CompileOptions;
pub const CompileError = dsl.CompileError;

test {
    std.testing.refAllDecls(@This());
}
