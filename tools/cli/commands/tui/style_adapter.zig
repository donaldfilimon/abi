//! Cyber-tech styling adapter for launcher and GPU dashboard chrome.
//!
//! Hybrid scope:
//! - Inherit shared readability tokens from the active Theme.
//! - Layer launcher/GPU-specific accent/status tokens locally.

const tui = @import("../../tui/mod.zig");

pub const ChromeStyle = struct {
    frame: []const u8,
    title: []const u8,
    subtitle: []const u8,
    keycap_fg: []const u8,
    keycap_bg: []const u8,
    selection_rail: []const u8,
    selection_bg: []const u8,
    selection_fg: []const u8,
    chip_fg: []const u8,
    chip_bg: []const u8,
    live: []const u8,
    paused: []const u8,
    info: []const u8,
    warning: []const u8,
    @"error": []const u8,
    success: []const u8,
};

pub fn launcher(theme: *const tui.Theme) ChromeStyle {
    return .{
        .frame = theme.border,
        .title = theme.primary,
        .subtitle = theme.text_muted,
        .keycap_fg = "\x1b[38;5;153m",
        .keycap_bg = "\x1b[48;5;23m",
        .selection_rail = "\x1b[38;5;51m",
        .selection_bg = "\x1b[48;5;24m",
        .selection_fg = "\x1b[38;5;230m",
        .chip_fg = "\x1b[38;5;51m",
        .chip_bg = "\x1b[48;5;23m",
        .live = "\x1b[38;5;48m",
        .paused = "\x1b[38;5;220m",
        .info = theme.info,
        .warning = theme.warning,
        .@"error" = theme.@"error",
        .success = theme.success,
    };
}

pub fn gpu(theme: *const tui.Theme) ChromeStyle {
    return .{
        .frame = theme.border,
        .title = theme.primary,
        .subtitle = theme.text_muted,
        .keycap_fg = "\x1b[38;5;189m",
        .keycap_bg = "\x1b[48;5;24m",
        .selection_rail = "\x1b[38;5;45m",
        .selection_bg = "\x1b[48;5;17m",
        .selection_fg = "\x1b[38;5;230m",
        .chip_fg = "\x1b[38;5;45m",
        .chip_bg = "\x1b[48;5;24m",
        .live = "\x1b[38;5;48m",
        .paused = "\x1b[38;5;220m",
        .info = theme.info,
        .warning = theme.warning,
        .@"error" = theme.@"error",
        .success = theme.success,
    };
}

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
