const backend = @import("../backend.zig");

pub const backendName = backend.backendName;
pub const backendDisplayName = backend.backendDisplayName;
pub const backendDescription = backend.backendDescription;
pub const backendFlag = backend.backendFlag;
pub const backendFromString = backend.backendFromString;
pub const backendSupportsKernels = backend.backendSupportsKernels;
