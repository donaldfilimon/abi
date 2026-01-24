//! CLI command modules.
//!
//! Each command is implemented in its own module for maintainability.

pub const db = @import("db.zig");
pub const agent = @import("agent.zig");
pub const bench = @import("bench.zig");
pub const config = @import("config.zig");
pub const convert = @import("convert.zig");
pub const discord = @import("discord.zig");
pub const embed = @import("embed.zig");
pub const explore = @import("explore.zig");
pub const gpu = @import("gpu.zig");
pub const gpu_dashboard = @import("gpu_dashboard.zig");
pub const llm = @import("llm.zig");
pub const network = @import("network.zig");
pub const simd = @import("simd.zig");
pub const system_info = @import("system_info.zig");
pub const task = @import("task.zig");
pub const tui = @import("tui.zig");
pub const train = @import("train.zig");
pub const completions = @import("completions.zig");
pub const multi_agent = @import("multi_agent.zig");
