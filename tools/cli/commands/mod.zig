//! CLI command modules.
//!
//! Each command declares `pub const meta: command_mod.Meta` and `pub fn run`.
//! The descriptors array is auto-derived at comptime from these declarations,
//! eliminating the manual three-layer bridge (catalog → descriptor → wiring).

const std = @import("std");
const command_mod = @import("../command.zig");
const CommandDescriptor = command_mod.CommandDescriptor;

// ─── Command module imports (pub for test discovery) ─────────────────────────

pub const db = @import("db.zig");
pub const agent = @import("agent.zig");
pub const bench = @import("bench/mod.zig");
pub const gpu = @import("gpu.zig");
pub const network = @import("network.zig");
pub const system_info = @import("system_info.zig");
pub const multi_agent = @import("multi_agent.zig");
pub const explore = @import("explore.zig");
pub const simd = @import("simd.zig");
pub const config = @import("config.zig");
pub const discord = @import("discord.zig");
pub const llm = @import("llm/mod.zig");
pub const model = @import("model.zig");
pub const embed = @import("embed.zig");
pub const train = @import("train/mod.zig");
pub const convert = @import("convert.zig");
pub const task = @import("task.zig");
pub const editor = @import("editor.zig");
pub const ui = @import("ui/mod.zig");
pub const plugins = @import("plugins.zig");
pub const profile = @import("profile.zig");
pub const completions = @import("completions.zig");
pub const status = @import("status.zig");
pub const toolchain = @import("toolchain.zig");
pub const lsp = @import("lsp.zig");
pub const mcp = @import("mcp.zig");
pub const acp = @import("acp.zig");
pub const ralph = @import("ralph/mod.zig");
pub const gendocs = @import("gendocs.zig");
pub const os_agent = @import("os_agent.zig");
pub const brain = @import("brain.zig");
pub const doctor = @import("doctor.zig");
pub const clean = @import("clean.zig");
pub const env = @import("env.zig");
pub const init = @import("init.zig");

// ─── Comptime-derived command registry ───────────────────────────────────────

/// Tuple of all registered command modules, in display order.
/// Each module must export `pub const meta: command_mod.Meta` and `pub fn run`.
const command_modules = .{
    db,          agent,    bench,   gpu,     network,     system_info,
    multi_agent, os_agent, explore, simd,    config,      discord,
    llm,         model,    embed,   train,   convert,     task,
    editor,      ui,       plugins, profile, completions, status,
    toolchain,   lsp,      mcp,     acp,     ralph,       gendocs,
    brain,       doctor,   clean,   env,     init,
};

/// Command descriptors auto-derived from command module metadata.
pub const descriptors: [std.meta.fields(@TypeOf(command_modules)).len]CommandDescriptor = blk: {
    const fields = std.meta.fields(@TypeOf(command_modules));
    var result: [fields.len]CommandDescriptor = undefined;
    for (fields, 0..) |field, i| {
        const mod = @field(command_modules, field.name);
        result[i] = command_mod.toDescriptor(mod);
    }
    break :blk result;
};

pub fn findDescriptor(raw_name: []const u8) ?*const CommandDescriptor {
    for (&descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw_name, descriptor.name)) return descriptor;
        for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw_name, alias)) return descriptor;
        }
    }
    return null;
}

test {
    std.testing.refAllDecls(@This());
}
