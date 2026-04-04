//! Integration Tests: TUI
//!
//! Tests the TUI module's non-interactive components through
//! the abi public API. Covers types, layout primitives, screen
//! rendering, and dashboard smoke tests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

/// Helper: skip test when TUI feature is disabled.
fn requireRealTui() !void {
    if (!build_options.feat_tui) return error.SkipZigTest;
}

/// Re-export dashboard's hasVisibleCell for use in tests.
const hasVisibleCell = abi.tui.dashboard.hasVisibleCell;
const containsText = abi.tui.dashboard.containsText;

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

test "tui: Rect hit testing contains method" {
    const Rect = abi.tui.types.Rect;
    const r = Rect{ .x = 10, .y = 10, .width = 50, .height = 20 };

    // Inside bounds
    try std.testing.expect(r.contains(10, 10));
    try std.testing.expect(r.contains(30, 20));
    try std.testing.expect(r.contains(59, 29));

    // Outside bounds
    try std.testing.expect(!r.contains(9, 10)); // left
    try std.testing.expect(!r.contains(10, 9)); // top
    try std.testing.expect(!r.contains(60, 20)); // right
    try std.testing.expect(!r.contains(30, 30)); // bottom
}

// === Layout Primitive Splitting ===

test "tui: Layout primitive splitting" {
    try requireRealTui();

    const Rect = abi.tui.types.Rect;
    const Constraint = abi.tui.types.Constraint;
    const layout = abi.tui.layout;

    // Vertical split with fixed, percentage, and min constraints
    const r = Rect{ .x = 0, .y = 0, .width = 100, .height = 40 };
    const constraints = [_]Constraint{
        .{ .fixed = 5 },
        .{ .percentage = 50 },
        .{ .min = 10 },
    };

    const split_res = layout.split(r, .vertical, &constraints);
    try std.testing.expectEqual(@as(usize, 3), split_res.len);

    const rects = split_res.slice();

    // Fixed region: exactly 5 columns wide
    try std.testing.expectEqual(@as(u16, 5), rects[0].width);
    try std.testing.expectEqual(@as(u16, 0), rects[0].x);
    try std.testing.expectEqual(@as(u16, 40), rects[0].height); // full height preserved

    // Percentage region: 50% of 100 = 50 columns
    try std.testing.expectEqual(@as(u16, 50), rects[1].width);
    try std.testing.expectEqual(@as(u16, 5), rects[1].x); // offset by previous

    // Min region: remaining space (45), satisfies min=10
    try std.testing.expectEqual(@as(u16, 45), rects[2].width);
    try std.testing.expectEqual(@as(u16, 55), rects[2].x);

    // Horizontal split sanity check
    const h_constraints = [_]Constraint{
        .{ .fixed = 3 },
        .{ .fixed = 18 },
        .{ .fixed = 3 },
    };
    const h_res = layout.split(r, .horizontal, &h_constraints);
    try std.testing.expectEqual(@as(usize, 3), h_res.len);
    const h_rects = h_res.slice();
    try std.testing.expectEqual(@as(u16, 3), h_rects[0].height);
    try std.testing.expectEqual(@as(u16, 18), h_rects[1].height);
    try std.testing.expectEqual(@as(u16, 3), h_rects[1].y);

    // Empty constraints yields zero rects
    const empty_res = layout.split(r, .vertical, &[_]Constraint{});
    try std.testing.expectEqual(@as(usize, 0), empty_res.len);
}

// === Screen Resize and Rect Contract ===

test "tui: Screen resize and rect contract" {
    try requireRealTui();

    const Screen = abi.tui.render.Screen;

    // Create an initial 80x24 screen
    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();

    try std.testing.expectEqual(@as(u16, 80), screen.width);
    try std.testing.expectEqual(@as(u16, 24), screen.height);

    // rect() must reflect current dimensions
    const r1 = screen.rect();
    try std.testing.expectEqual(@as(u16, 0), r1.x);
    try std.testing.expectEqual(@as(u16, 0), r1.y);
    try std.testing.expectEqual(@as(u16, 80), r1.width);
    try std.testing.expectEqual(@as(u16, 24), r1.height);

    // Write a cell and verify it lands in the back buffer
    screen.setCell(5, 3, .{ .char = 'Z', .style = .{} });
    const idx = @as(usize, 3) * @as(usize, 80) + 5;
    try std.testing.expectEqual(@as(u21, 'Z'), screen.back[idx].char);

    // Resize to 40x12
    try screen.resize(40, 12);
    try std.testing.expectEqual(@as(u16, 40), screen.width);
    try std.testing.expectEqual(@as(u16, 12), screen.height);

    // rect() must reflect new dimensions
    const r2 = screen.rect();
    try std.testing.expectEqual(@as(u16, 40), r2.width);
    try std.testing.expectEqual(@as(u16, 12), r2.height);

    // After resize, buffer should be cleared (all blanks)
    for (screen.back) |cell| {
        try std.testing.expectEqual(@as(u21, ' '), cell.char);
    }

    // Buffer length must match new dimensions
    try std.testing.expectEqual(@as(usize, 40 * 12), screen.back.len);
    try std.testing.expectEqual(@as(usize, 40 * 12), screen.front.len);

    // clear() resets back buffer to blanks
    screen.setCell(0, 0, .{ .char = 'X', .style = .{} });
    screen.clear();
    try std.testing.expectEqual(@as(u21, ' '), screen.back[0].char);
}

// === Dashboard Diagnostics Shell ===

test "tui: dashboard layout modes are resize aware" {
    try requireRealTui();

    const Rect = abi.tui.types.Rect;
    const dashboard = abi.tui.dashboard;

    try std.testing.expectEqual(dashboard.LayoutMode.wide, dashboard.computeLayout(Rect{ .x = 0, .y = 0, .width = 120, .height = 32 }).mode);
    try std.testing.expectEqual(dashboard.LayoutMode.medium, dashboard.computeLayout(Rect{ .x = 0, .y = 0, .width = 80, .height = 24 }).mode);
    try std.testing.expectEqual(dashboard.LayoutMode.compact, dashboard.computeLayout(Rect{ .x = 0, .y = 0, .width = 40, .height = 12 }).mode);
    try std.testing.expectEqual(dashboard.LayoutMode.compact, dashboard.computeLayout(Rect{ .x = 0, .y = 0, .width = 20, .height = 8 }).mode);
    try std.testing.expectEqual(dashboard.LayoutMode.minimal, dashboard.computeLayout(Rect{ .x = 0, .y = 0, .width = 17, .height = 6 }).mode);
}

test "tui: dashboard navigation and help overlay transitions" {
    try requireRealTui();

    const Key = abi.tui.types.Key;
    const dashboard = abi.tui.dashboard;

    var state: dashboard.AppState = .{};
    try std.testing.expectEqual(dashboard.FocusRegion.nav, state.focused_region);

    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, .tab));
    try std.testing.expectEqual(dashboard.FocusRegion.detail, state.focused_region);

    state.focused_region = .nav;
    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, .down));
    try std.testing.expectEqual(@as(usize, 1), state.nav_index);

    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, .enter));
    try std.testing.expectEqual(dashboard.View.features, state.current_view);
    try std.testing.expectEqual(dashboard.FocusRegion.detail, state.focused_region);

    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, Key{ .char = '?' }));
    try std.testing.expect(state.help_visible);

    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, .escape));
    try std.testing.expect(!state.help_visible);

    try std.testing.expectEqual(dashboard.DashboardAction.none, dashboard.handleKey(&state, Key{ .char = 'g' }));
    try std.testing.expectEqual(dashboard.View.overview, state.current_view);
    try std.testing.expectEqual(dashboard.FocusRegion.nav, state.focused_region);
}

test "tui: dashboard features view tracks canonical catalog" {
    try requireRealTui();

    const Screen = abi.tui.render.Screen;
    const dashboard = abi.tui.dashboard;

    var screen = try Screen.init(std.testing.allocator, 120, 40);
    defer screen.deinit();

    var state: dashboard.AppState = .{
        .current_view = .features,
        .focused_region = .detail,
        .nav_index = 1,
        .selected_row = abi.meta.features.feature_count - 1,
    };
    dashboard.renderDashboard(&screen, &state);

    try std.testing.expect(hasVisibleCell(screen.back));
    try std.testing.expect(containsText(screen.back, "ABI DIAGNOSTIC SHELL"));
    try std.testing.expect(containsText(screen.back, "feat_inference"));
    try std.testing.expect(containsText(screen.back, "inference"));
}

test "tui: dashboard compact and minimal modes render usable diagnostics" {
    try requireRealTui();

    const Screen = abi.tui.render.Screen;
    const dashboard = abi.tui.dashboard;

    {
        var screen = try Screen.init(std.testing.allocator, 20, 8);
        defer screen.deinit();

        var state: dashboard.AppState = .{
            .current_view = .features,
            .focused_region = .detail,
            .nav_index = 1,
        };
        dashboard.renderDashboard(&screen, &state);

        try std.testing.expect(hasVisibleCell(screen.back));
        try std.testing.expect(containsText(screen.back, "FEAT"));
    }

    {
        var screen = try Screen.init(std.testing.allocator, 17, 6);
        defer screen.deinit();

        var state: dashboard.AppState = .{};
        dashboard.renderDashboard(&screen, &state);

        try std.testing.expect(hasVisibleCell(screen.back));
        try std.testing.expect(containsText(screen.back, "grow terminal"));
    }
}

test "tui: dashboard resize preserves valid state and render" {
    try requireRealTui();

    const Screen = abi.tui.render.Screen;
    const dashboard = abi.tui.dashboard;

    var screen = try Screen.init(std.testing.allocator, 80, 24);
    defer screen.deinit();

    var state: dashboard.AppState = .{
        .current_view = .runtime,
        .focused_region = .detail,
        .nav_index = 2,
        .selected_row = 17,
        .detail_scroll = 4,
    };

    dashboard.renderDashboard(&screen, &state);
    try std.testing.expect(hasVisibleCell(screen.back));
    try std.testing.expect(containsText(screen.back, "Runtime"));

    try screen.resize(40, 12);
    dashboard.renderDashboard(&screen, &state);
    try std.testing.expect(hasVisibleCell(screen.back));
    try std.testing.expectEqual(dashboard.LayoutMode.compact, dashboard.computeLayout(screen.rect()).mode);

    try screen.resize(17, 6);
    dashboard.renderDashboard(&screen, &state);
    try std.testing.expect(hasVisibleCell(screen.back));
    try std.testing.expect(containsText(screen.back, "grow terminal"));
}

test {
    std.testing.refAllDecls(@This());
}
