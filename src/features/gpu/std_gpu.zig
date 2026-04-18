//! Re-export from internal/std_gpu

pub const is_gpu_target = @import("std_gpu.zig").is_gpu_target;
pub const std_gpu_available = @import("std_gpu.zig").std_gpu_available;
pub const AddressSpace = @import("std_gpu.zig").AddressSpace;
pub const GlobalPtr = @import("std_gpu.zig").GlobalPtr;
pub const SharedPtr = @import("std_gpu.zig").SharedPtr;
pub const ConstantPtr = @import("std_gpu.zig").ConstantPtr;
pub const UniformPtr = @import("std_gpu.zig").UniformPtr;
