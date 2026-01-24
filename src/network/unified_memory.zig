//! Unified Memory Wrapper
//!
//! This file re-exports the original unified_memory module to provide a
//! flattened import path (`@import("unified_memory.zig")`). The full
//! implementation resides in `unified_memory/mod.zig` and its sub-modules.

pub const UnifiedMemoryManager = @import("unified_memory/mod.zig").UnifiedMemoryManager;
pub const UnifiedMemoryConfig = @import("unified_memory/mod.zig").UnifiedMemoryConfig;
pub const UnifiedMemoryError = @import("unified_memory/mod.zig").UnifiedMemoryError;
pub const MemoryRegion = @import("unified_memory/mod.zig").MemoryRegion;
pub const RegionId = @import("unified_memory/mod.zig").RegionId;
pub const RegionFlags = @import("unified_memory/mod.zig").RegionFlags;
pub const RegionState = @import("unified_memory/mod.zig").RegionState;
pub const CoherenceProtocol = @import("unified_memory/mod.zig").CoherenceProtocol;
pub const CoherenceState = @import("unified_memory/mod.zig").CoherenceState;
pub const RemotePtr = @import("unified_memory/mod.zig").RemotePtr;
