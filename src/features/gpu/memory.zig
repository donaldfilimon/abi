const memory_base = @import("memory/base.zig");
const unified = @import("unified.zig");
const unified_buffer = @import("unified_buffer.zig");

pub const BufferFlags = memory_base.BufferFlags;
pub const GpuBuffer = memory_base.GpuBuffer;
pub const Buffer = GpuBuffer; // Alias for convenience

pub const MemoryError = memory_base.MemoryError;
pub const MemoryInfo = unified.MemoryInfo;

pub const UnifiedBuffer = unified_buffer.Buffer;
pub const BufferOptions = unified_buffer.BufferOptions;
