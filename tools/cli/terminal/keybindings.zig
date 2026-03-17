//! Centralized keybinding registry for TUI dashboards.
//!
//! Maps raw key events to semantic actions, providing a single source
//! of truth for common keyboard shortcuts across all dashboard views.

const std = @import("std");
const events = @import("events.zig");

pub const KeyAction = union(enum) {
    quit,
    pause,
    theme_next,
    theme_prev,
    help_toggle,
    focus_next,
    focus_prev,
    refresh,
    settings_toggle,
    find,
    primary_action,
    density_toggle,
    tab_jump: u4, // 0-9 direct tab selection
    none,
};

/// Resolve a key event to a dashboard action.
pub fn resolve(key: events.Key) KeyAction {
    switch (key.code) {
        .ctrl_c, .escape => return .quit,
        .tab => return if (key.mods.shift) .focus_prev else .focus_next,
        .enter => return .primary_action,
        .character => {
            if (key.char) |ch| {
                // Ctrl+R for refresh
                if (ch == 'r' and key.mods.ctrl) return .refresh;

                return switch (ch) {
                    'q' => .quit,
                    'p' => .pause,
                    't' => .theme_next,
                    'T' => .theme_prev,
                    'h', '?' => .help_toggle,
                    'r' => .refresh,
                    's' => .settings_toggle,
                    'f', '/' => .find,
                    'd' => .density_toggle,
                    '1'...'9' => .{ .tab_jump = @intCast(ch - '0') },
                    '0' => .{ .tab_jump = 10 },
                    else => .none,
                };
            }
        },
        else => {},
    }
    return .none;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "resolve quit keys" {
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .ctrl_c }));
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .escape }));
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .character, .char = 'q' }));
}

test "resolve action keys" {
    try std.testing.expectEqual(KeyAction.pause, resolve(.{ .code = .character, .char = 'p' }));
    try std.testing.expectEqual(KeyAction.theme_next, resolve(.{ .code = .character, .char = 't' }));
    try std.testing.expectEqual(KeyAction.theme_prev, resolve(.{ .code = .character, .char = 'T' }));
    try std.testing.expectEqual(KeyAction.help_toggle, resolve(.{ .code = .character, .char = 'h' }));
    try std.testing.expectEqual(KeyAction.help_toggle, resolve(.{ .code = .character, .char = '?' }));
}

test "resolve new action keys" {
    try std.testing.expectEqual(KeyAction.refresh, resolve(.{ .code = .character, .char = 'r' }));
    try std.testing.expectEqual(KeyAction.settings_toggle, resolve(.{ .code = .character, .char = 's' }));
    try std.testing.expectEqual(KeyAction.find, resolve(.{ .code = .character, .char = 'f' }));
    try std.testing.expectEqual(KeyAction.find, resolve(.{ .code = .character, .char = '/' }));
    try std.testing.expectEqual(KeyAction.density_toggle, resolve(.{ .code = .character, .char = 'd' }));
    try std.testing.expectEqual(KeyAction.primary_action, resolve(.{ .code = .enter }));
}

test "resolve tab jump keys" {
    const r1 = resolve(.{ .code = .character, .char = '1' });
    try std.testing.expectEqual(KeyAction{ .tab_jump = 1 }, r1);
    const r9 = resolve(.{ .code = .character, .char = '9' });
    try std.testing.expectEqual(KeyAction{ .tab_jump = 9 }, r9);
    const r0 = resolve(.{ .code = .character, .char = '0' });
    try std.testing.expectEqual(KeyAction{ .tab_jump = 10 }, r0);
}

test "resolve unknown keys return none" {
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'x' }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .up }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'z' }));
}

test "resolve tab maps to focus_next" {
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab }));
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab, .mods = .{} }));
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab, .mods = .{ .ctrl = true } }));
}

test "resolve shift-tab maps to focus_prev" {
    try std.testing.expectEqual(KeyAction.focus_prev, resolve(.{ .code = .tab, .mods = .{ .shift = true } }));
    try std.testing.expectEqual(KeyAction.focus_prev, resolve(.{ .code = .tab, .mods = .{ .shift = true, .ctrl = true } }));
}

test {
    std.testing.refAllDecls(@This());
}
