//! Re-export from internal/std_gpu

pub const is_gpu_target = @import("internal/std_gpu.zig").is_gpu_target;
pub const std_gpu_available = @import("internal/std_gpu.zig").std_gpu_available;
pub const AddressSpace = @import("internal/std_gpu.zig").AddressSpace;
pub const GlobalPtr = @import("internal/std_gpu.zig").GlobalPtr;
pub const SharedPtr = @import("internal/std_gpu.zig").SharedPtr;
pub const ConstantPtr = @import("internal/std_gpu.zig").ConstantPtr;
pub const UniformPtr = @import("internal/std_gpu.zig").UniformPtr;
