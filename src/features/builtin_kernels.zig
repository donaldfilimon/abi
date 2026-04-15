//! Re-export from gpu/builtin_kernels

pub const builtin_kernels = @import("gpu/builtin_kernels.zig").builtin_kernels;
pub const buildKernelIR = @import("gpu/builtin_kernels.zig").buildKernelIR;
pub const BuiltinKernel = @import("gpu/builtin_kernels.zig").BuiltinKernel;
