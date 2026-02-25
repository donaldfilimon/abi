//! Centralized keybinding registry for TUI dashboards.
//!
//! Maps raw key events to semantic actions, providing a single source
//! of truth for common keyboard shortcuts across all dashboard views.

const events = @import("events.zig");

pub const KeyAction = enum {
    quit,
    pause,
    theme_next,
    theme_prev,
    help_toggle,
    none,
};

/// Resolve a key event to a dashboard action.
pub fn resolve(key: events.Key) KeyAction {
    switch (key.code) {
        .ctrl_c, .escape => return .quit,
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
    const std = @import("std");
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .ctrl_c }));
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .escape }));
    try std.testing.expectEqual(KeyAction.quit, resolve(.{ .code = .character, .char = 'q' }));
}

test "resolve action keys" {
    const std = @import("std");
    try std.testing.expectEqual(KeyAction.pause, resolve(.{ .code = .character, .char = 'p' }));
    try std.testing.expectEqual(KeyAction.theme_next, resolve(.{ .code = .character, .char = 't' }));
    try std.testing.expectEqual(KeyAction.theme_prev, resolve(.{ .code = .character, .char = 'T' }));
    try std.testing.expectEqual(KeyAction.help_toggle, resolve(.{ .code = .character, .char = 'h' }));
    try std.testing.expectEqual(KeyAction.help_toggle, resolve(.{ .code = .character, .char = '?' }));
}

test "resolve unknown keys return none" {
    const std = @import("std");
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'x' }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .up }));
    try std.testing.expectEqual(KeyAction.none, resolve(.{ .code = .character, .char = 'z' }));
}

test {
    std.testing.refAllDecls(@This());
}
