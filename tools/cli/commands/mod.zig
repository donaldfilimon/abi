//! CLI command modules.
//!
//! Command definitions are sourced from the generated registry snapshot and
//! normalized into descriptors at comptime.

const std = @import("std");
const command_mod = @import("../command.zig");
const CommandDescriptor = command_mod.CommandDescriptor;
const generated = @import("../generated/cli_registry_snapshot.zig");
const registry_overrides = @import("../registry/overrides.zig");

// ─── Command module re-exports (pub for test discovery) ──────────────────────

pub const db = generated.db;
pub const agent = generated.agent;
pub const bench = generated.bench;
pub const gpu = generated.gpu;
pub const network = generated.network;
pub const system_info = generated.system_info;
pub const multi_agent = generated.multi_agent;
pub const os_agent = generated.os_agent;
pub const explore = generated.explore;
pub const simd = generated.simd;
pub const config = generated.config;
pub const discord = generated.discord;
pub const llm = generated.llm;
pub const model = generated.model;
pub const embed = generated.embed;
pub const train = generated.train;
pub const convert = generated.convert;
pub const task = generated.task;
pub const ui = generated.ui;
pub const plugins = generated.plugins;
pub const profile = generated.profile;
pub const completions = generated.completions;
pub const status = generated.status;
pub const toolchain = generated.toolchain;
pub const lsp = generated.lsp;
pub const mcp = generated.mcp;
pub const acp = generated.acp;
pub const ralph = generated.ralph;
pub const gendocs = generated.gendocs;
pub const brain = generated.brain;
pub const doctor = generated.doctor;
pub const clean = generated.clean;
pub const env = generated.env;
pub const init = generated.init;

// ─── Comptime-derived command registry ────────────────────────────────────────

const command_modules = generated.command_modules;

fn applyOverride(desc: *CommandDescriptor, comptime ov: registry_overrides.CommandOverride) void {
    if (ov.source_id) |source_id| desc.source_id = source_id;
    if (ov.default_subcommand) |default_subcommand| desc.default_subcommand = default_subcommand;
    if (ov.visibility) |visibility| desc.visibility = visibility;
    if (ov.risk) |risk| desc.risk = risk;
    if (ov.ui) |ui| desc.ui = ui;
    if (ov.options) |options| desc.options = options;
    if (ov.middleware_tags) |middleware_tags| desc.middleware_tags = middleware_tags;
}

fn validateRegistry(comptime descriptors: []const CommandDescriptor) void {
    inline for (descriptors, 0..) |desc, i| {
        if (desc.name.len == 0) {
            @compileError(std.fmt.comptimePrint("Empty command name at index {d}", .{i}));
        }
        inline for (descriptors[0..i]) |prev| {
            if (std.mem.eql(u8, prev.name, desc.name)) {
                @compileError(std.fmt.comptimePrint("Duplicate command name '{s}'", .{desc.name}));
            }
        }

        inline for (desc.aliases) |alias| {
            if (std.mem.eql(u8, alias, desc.name)) {
                @compileError(std.fmt.comptimePrint(
                    "Alias '{s}' duplicates command name '{s}'",
                    .{ alias, desc.name },
                ));
            }
        }
    }

    inline for (descriptors) |lhs| {
        inline for (lhs.aliases) |alias| {
            inline for (descriptors) |rhs| {
                if (std.mem.eql(u8, rhs.name, alias)) {
                    @compileError(std.fmt.comptimePrint(
                        "Alias '{s}' on '{s}' collides with command name '{s}'",
                        .{ alias, lhs.name, rhs.name },
                    ));
                }
                if (!std.mem.eql(u8, rhs.name, lhs.name)) {
                    inline for (rhs.aliases) |rhs_alias| {
                        if (std.mem.eql(u8, rhs_alias, alias)) {
                            @compileError(std.fmt.comptimePrint(
                                "Alias collision: '{s}' used by '{s}' and '{s}'",
                                .{ alias, lhs.name, rhs.name },
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Command descriptors auto-derived from command module metadata + overrides.
pub const descriptors: [std.meta.fields(@TypeOf(command_modules)).len]CommandDescriptor = blk: {
    const fields = std.meta.fields(@TypeOf(command_modules));
    var result: [fields.len]CommandDescriptor = undefined;

    for (fields, 0..) |field, i| {
        const mod = @field(command_modules, field.name);
        result[i] = command_mod.toDescriptor(mod);
    }

    inline for (registry_overrides.command_overrides) |override| {
        var matched = false;
        inline for (&result) |*desc| {
            if (std.mem.eql(u8, desc.name, override.name)) {
                applyOverride(desc, override);
                matched = true;
                break;
            }
        }
        if (!matched) {
            @compileError(std.fmt.comptimePrint(
                "Unknown command override target: '{s}'",
                .{override.name},
            ));
        }
    }

    comptime validateRegistry(&result);
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
