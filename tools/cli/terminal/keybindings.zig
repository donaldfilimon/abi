//! Centralized keybinding registry for TUI dashboards.
//!
//! Maps raw key events to semantic actions, providing a single source
//! of truth for common keyboard shortcuts across all dashboard views.

const std = @import("std");
const events = @import("events.zig");

pub const KeyAction = enum {
    quit,
    pause,
    theme_next,
    theme_prev,
    help_toggle,
    focus_next,
    focus_prev,
    none,
};

/// Resolve a key event to a dashboard action.
pub fn resolve(key: events.Key) KeyAction {
    switch (key.code) {
        .ctrl_c, .escape => return .quit,
        .tab => return if (key.mods.shift) .focus_prev else .focus_next,
        .character => {
            if (key.char) |ch| {
                return switch (ch) {
                    'q' => .quit,
                    'p' => .pause,
                    't' => .theme_next,
                    'T' => .theme_prev,
                    'h', '?' => .help_toggle,
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

test "resolve unknown keys return none" {
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'x' }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .up }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'z' }));
}

test "resolve tab maps to focus_next" {
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab }));
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab, .mods = .{} }));
    // Tab with ctrl or alt but not shift is still focus_next
    try std.testing.expectEqual(KeyAction.focus_next, resolve(.{ .code = .tab, .mods = .{ .ctrl = true } }));
}

test "resolve shift-tab maps to focus_prev" {
    try std.testing.expectEqual(KeyAction.focus_prev, resolve(.{ .code = .tab, .mods = .{ .shift = true } }));
    try std.testing.expectEqual(KeyAction.focus_prev, resolve(.{ .code = .tab, .mods = .{ .shift = true, .ctrl = true } }));
}

test "no action maps to two different results" {
    // Verify each character key maps to exactly one action and there
    // are no accidental overlaps in the character switch.
    const char_actions = [_]struct { char: u8, expected: KeyAction }{
        .{ .char = 'q', .expected = .quit },
        .{ .char = 'p', .expected = .pause },
        .{ .char = 't', .expected = .theme_next },
        .{ .char = 'T', .expected = .theme_prev },
        .{ .char = 'h', .expected = .help_toggle },
        .{ .char = '?', .expected = .help_toggle },
    };
    for (char_actions) |entry| {
        try std.testing.expectEqual(entry.expected, resolve(.{ .code = .character, .char = entry.char }));
    }
}

test {
    std.testing.refAllDecls(@This());
}
