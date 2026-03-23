//! Integration Tests: TUI
//!
//! Tests the TUI module's non-interactive components through
//! the abi public API.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// === Module Availability ===

test "tui: isEnabled returns bool" {
    const enabled = abi.tui.isEnabled();
    try std.testing.expect(enabled == true or enabled == false);
}

test "tui: types are accessible" {
    const Rect = abi.tui.types.Rect;
    const r = Rect{ .x = 0, .y = 0, .width = 80, .height = 24 };
    try std.testing.expectEqual(@as(u32, 1920), r.area());
}

test "tui: Color type constructs" {
    const Color = abi.tui.types.Color;
    const red: Color = .red;
    try std.testing.expectEqual(@as(?u8, 31), red.fgCode());
}

test "tui: Style type has defaults" {
    const Style = abi.tui.types.Style;
    const s = Style{};
    try std.testing.expect(!s.bold);
    try std.testing.expect(!s.italic);
}

test "tui: Rect splitting" {
    const Rect = abi.tui.types.Rect;
    const r = Rect{ .x = 0, .y = 0, .width = 100, .height = 50 };
    const split = r.splitHorizontal(10);
    try std.testing.expectEqual(@as(u16, 10), split.top.height);
    try std.testing.expectEqual(@as(u16, 40), split.bottom.height);
}

test "tui: Cell default is blank" {
    const Cell = abi.tui.types.Cell;
    const cell = Cell{};
    try std.testing.expectEqual(@as(u21, ' '), cell.char);
}

test "tui: Key type union" {
    const Key = abi.tui.types.Key;
    const k = Key{ .char = 'a' };
    try std.testing.expectEqual(@as(u21, 'a'), k.char);
}

test "tui: TuiError and Error types exist" {
    _ = abi.tui.TuiError;
    _ = abi.tui.Error;
}

test "tui: Context init and deinit" {
    var ctx = abi.tui.Context.init(std.testing.allocator);
    ctx.deinit();
}

test {
    std.testing.refAllDecls(@This());
}
