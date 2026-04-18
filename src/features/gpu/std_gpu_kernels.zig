pub const vectorAddCpu = @import("std_gpu_kernels.zig").vectorAddCpu;
pub const vectorAddGpu = @import("std_gpu_kernels.zig").vectorAddGpu;
pub const vectorMulCpu = @import("std_gpu_kernels.zig").vectorMulCpu;
pub const vectorMulGpu = @import("std_gpu_kernels.zig").vectorMulGpu;
pub const matrixMultiplyCpu = @import("std_gpu_kernels.zig").matrixMultiplyCpu;
pub const matrixMultiplyGpu = @import("std_gpu_kernels.zig").matrixMultiplyGpu;
pub const matrixMulCpu = @import("std_gpu_kernels.zig").matrixMulCpu;
pub const StdGpuKernelRegistry = @import("std_gpu_kernels.zig").StdGpuKernelRegistry;
