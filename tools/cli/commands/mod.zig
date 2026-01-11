//! CLI command modules.
//!
//! Each command is implemented in its own module for maintainability.

pub const db = @import("db.zig");
pub const agent = @import("agent.zig");
pub const config = @import("config.zig");
pub const explore = @import("explore.zig");
pub const gpu = @import("gpu.zig");
pub const network = @import("network.zig");
pub const simd = @import("simd.zig");
pub const system_info = @import("system_info.zig");
pub const llm = @import("llm.zig");
