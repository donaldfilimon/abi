//! TUI smart-default mapping helpers.

const std = @import("std");

pub const DefaultKind = enum {
    safe,
    interactive,
    none,
};

const empty_args = &[_][:0]const u8{};
const args_stats = [_][:0]const u8{"stats"};
const args_quick = [_][:0]const u8{"quick"};
const args_show = [_][:0]const u8{"show"};
const args_status = [_][:0]const u8{"status"};
const args_summary = [_][:0]const u8{"summary"};
const args_list = [_][:0]const u8{"list"};
const args_info = [_][:0]const u8{"info"};
const args_monitor = [_][:0]const u8{"monitor"};

pub fn commandDefaultArgs(cmd: anytype) []const [:0]const u8 {
    const tag = std.mem.sliceTo(@tagName(cmd), 0);

    if (std.mem.eql(u8, tag, "db")) return &args_stats;
    if (std.mem.eql(u8, tag, "bench")) return &args_quick;
    if (std.mem.eql(u8, tag, "config")) return &args_show;
    if (std.mem.eql(u8, tag, "discord")) return &args_status;
    if (std.mem.eql(u8, tag, "gpu")) return &args_summary;
    if (std.mem.eql(u8, tag, "llm")) return &args_list;
    if (std.mem.eql(u8, tag, "model")) return &args_list;
    if (std.mem.eql(u8, tag, "network")) return &args_status;
    if (std.mem.eql(u8, tag, "ralph")) return &args_status;
    if (std.mem.eql(u8, tag, "task")) return &args_list;
    if (std.mem.eql(u8, tag, "train")) return &args_info;
    if (std.mem.eql(u8, tag, "train_monitor")) return &args_monitor;
    return empty_args;
}
