pub const vectorAddCpu = @import("internal/std_gpu_kernels.zig").vectorAddCpu;
pub const vectorAddGpu = @import("internal/std_gpu_kernels.zig").vectorAddGpu;
pub const vectorMulCpu = @import("internal/std_gpu_kernels.zig").vectorMulCpu;
pub const vectorMulGpu = @import("internal/std_gpu_kernels.zig").vectorMulGpu;
pub const matrixMultiplyCpu = @import("internal/std_gpu_kernels.zig").matrixMultiplyCpu;
pub const matrixMultiplyGpu = @import("internal/std_gpu_kernels.zig").matrixMultiplyGpu;
pub const matrixMulCpu = @import("internal/std_gpu_kernels.zig").matrixMulCpu;
pub const StdGpuKernelRegistry = @import("internal/std_gpu_kernels.zig").StdGpuKernelRegistry;
