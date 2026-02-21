//! Composable component system for TUI dashboards.
//!
//! Provides a SubPanel protocol and reusable components (Header,
//! StatusBar, Section, Spacer) that can be composed into layouts
//! using renderStack() and renderRow().

const std = @import("std");
const layout = @import("layout.zig");
const render_utils = @import("render_utils.zig");
const unicode = @import("unicode.zig");
const terminal_mod = @import("terminal.zig");
const themes = @import("themes.zig");

pub const Terminal = terminal_mod.Terminal;
pub const Rect = layout.Rect;
pub const Constraint = layout.Constraint;
pub const Theme = themes.Theme;

/// Maximum number of panels supported in a single stack or row.
const max_panels = 32;

/// A render function that draws content within a given Rect.
pub const RenderFn = *const fn (
    ctx: *anyopaque,
    term: *Terminal,
    rect: Rect,
    theme: *const Theme,
) anyerror!void;

/// A composable sub-panel that can be stacked in a layout.
pub const SubPanel = struct {
    /// Display label (for debugging/identification).
    label: []const u8,
    /// Height constraint for this panel.
    constraint: Constraint,
    /// Render callback.
    render_fn: RenderFn,
    /// Opaque context pointer passed to render_fn.
    ctx: *anyopaque,
};

/// Render a vertical stack of sub-panels within a rect.
/// Distributes available height among panels using their
/// constraints, then calls each panel's render_fn with its
/// allocated Rect.
pub fn renderStack(
    term: *Terminal,
    rect: Rect,
    theme: *const Theme,
    panels: []const SubPanel,
) !void {
    if (panels.len == 0) return;
    if (rect.isEmpty()) return;
    if (panels.len > max_panels) return error.TooManyPanels;

    var constraints: [max_panels]Constraint = undefined;
    for (panels, 0..) |p, i| {
        constraints[i] = p.constraint;
    }

    var sizes: [max_panels]u16 = undefined;
    layout.distribute(
        rect.height,
        constraints[0..panels.len],
        sizes[0..panels.len],
    );

    var y_offset: u16 = 0;
    for (panels, 0..) |panel, i| {
        const h = sizes[i];
        if (h == 0) continue;
        const panel_rect = Rect{
            .x = rect.x,
            .y = rect.y +| y_offset,
            .width = rect.width,
            .height = h,
        };
        try panel.render_fn(panel.ctx, term, panel_rect, theme);
        y_offset +|= h;
    }
}

/// Render a horizontal row of sub-panels within a rect.
/// Distributes available width among panels using their
/// constraints, then calls each panel's render_fn with its
/// allocated Rect.
pub fn renderRow(
    term: *Terminal,
    rect: Rect,
    theme: *const Theme,
    panels: []const SubPanel,
) !void {
    if (panels.len == 0) return;
    if (rect.isEmpty()) return;
    if (panels.len > max_panels) return error.TooManyPanels;

    var constraints: [max_panels]Constraint = undefined;
    for (panels, 0..) |p, i| {
        constraints[i] = p.constraint;
    }

    var sizes: [max_panels]u16 = undefined;
    layout.distribute(
        rect.width,
        constraints[0..panels.len],
        sizes[0..panels.len],
    );

    var x_offset: u16 = 0;
    for (panels, 0..) |panel, i| {
        const w = sizes[i];
        if (w == 0) continue;
        const panel_rect = Rect{
            .x = rect.x +| x_offset,
            .y = rect.y,
            .width = w,
            .height = rect.height,
        };
        try panel.render_fn(panel.ctx, term, panel_rect, theme);
        x_offset +|= w;
    }
}

/// A simple header component that renders a centered title
/// in a box.
pub const Header = struct {
    title: []const u8,
    style: render_utils.BoxStyle,

    pub fn subPanel(self: *Header) SubPanel {
        return .{
            .label = "header",
            .constraint = .{ .fixed = 3 },
            .render_fn = &render,
            .ctx = @ptrCast(@alignCast(self)),
        };
    }

    /// The render function (called via subPanel's render_fn).
    fn render(
        ctx: *anyopaque,
        term: *Terminal,
        rect: Rect,
        theme: *const Theme,
    ) anyerror!void {
        const self: *Header = @ptrCast(@alignCast(ctx));
        if (rect.isEmpty() or rect.height < 3 or rect.width < 2)
            return;

        // Draw box outline.
        try render_utils.drawBox(term, rect, self.style, theme);

        // Center the title on the middle row.
        const inner_w: usize = @as(usize, rect.width) -| 2;
        const clipped = unicode.truncateToWidth(
            self.title,
            inner_w,
        );
        const clipped_w = unicode.displayWidth(clipped);
        const pad_left: usize = if (inner_w > clipped_w)
            (inner_w - clipped_w) / 2
        else
            0;

        try render_utils.moveTo(term, rect.x +| 1, rect.y +| 1);
        try term.write(theme.header);
        try render_utils.writeRepeat(term, " ", pad_left);
        try term.write(clipped);
        const used = pad_left + clipped_w;
        if (used < inner_w) {
            try render_utils.writeRepeat(
                term,
                " ",
                inner_w - used,
            );
        }
        try term.write(theme.reset);
    }
};

/// A status bar component that renders key-value pairs in a row.
pub const StatusBar = struct {
    items: []const StatusItem,
    style: render_utils.BoxStyle,

    pub const StatusItem = struct {
        label: []const u8,
        value: []const u8,
    };

    pub fn subPanel(self: *StatusBar) SubPanel {
        return .{
            .label = "status_bar",
            .constraint = .{ .fixed = 1 },
            .render_fn = &render,
            .ctx = @ptrCast(@alignCast(self)),
        };
    }

    fn render(
        ctx: *anyopaque,
        term: *Terminal,
        rect: Rect,
        theme: *const Theme,
    ) anyerror!void {
        const self: *StatusBar = @ptrCast(@alignCast(ctx));
        if (rect.isEmpty() or rect.width == 0) return;

        try render_utils.moveTo(term, rect.x, rect.y);
        try term.write(theme.text_dim);

        var col: usize = 0;
        const max_col: usize = rect.width;

        for (self.items, 0..) |item, i| {
            if (col >= max_col) break;

            // Add separator between items.
            if (i > 0) {
                if (col + 3 > max_col) break;
                try term.write(" \u{2502} ");
                col += 3;
            }

            // Write "label: value".
            const remaining = max_col -| col;
            if (remaining == 0) break;

            try term.write(theme.text);
            const lw = try render_utils.writeClipped(
                term,
                item.label,
                remaining,
            );
            col += lw;

            if (col + 2 <= max_col) {
                try term.write(": ");
                col += 2;

                try term.write(theme.accent);
                const vr = max_col -| col;
                const vw = try render_utils.writeClipped(
                    term,
                    item.value,
                    vr,
                );
                col += vw;
            }
        }

        // Pad remaining space.
        if (col < max_col) {
            try render_utils.writeRepeat(
                term,
                " ",
                max_col - col,
            );
        }
        try term.write(theme.reset);
    }
};

/// A labeled section with a title line and content area.
/// The content is rendered by a nested render function.
pub const Section = struct {
    title: []const u8,
    content_fn: RenderFn,
    content_ctx: *anyopaque,
    style: render_utils.BoxStyle,

    pub fn subPanel(self: *Section) SubPanel {
        return .{
            .label = "section",
            .constraint = .{ .fill = {} },
            .render_fn = &render,
            .ctx = @ptrCast(@alignCast(self)),
        };
    }

    fn render(
        ctx: *anyopaque,
        term: *Terminal,
        rect: Rect,
        theme: *const Theme,
    ) anyerror!void {
        const self: *Section = @ptrCast(@alignCast(ctx));
        if (rect.isEmpty() or rect.height < 2) return;

        // Draw title row.
        try render_utils.moveTo(term, rect.x, rect.y);
        try term.write(theme.header);

        const title_clipped = unicode.truncateToWidth(
            self.title,
            @as(usize, rect.width),
        );
        try render_utils.writePadded(
            term,
            title_clipped,
            @as(usize, rect.width),
        );
        try term.write(theme.reset);

        // Delegate remaining area to content_fn.
        if (rect.height > 1) {
            const content_rect = Rect{
                .x = rect.x,
                .y = rect.y +| 1,
                .width = rect.width,
                .height = rect.height -| 1,
            };
            try self.content_fn(
                self.content_ctx,
                term,
                content_rect,
                theme,
            );
        }
    }
};

/// A spacer that takes up space but renders nothing
/// (or a blank area).
pub const Spacer = struct {
    pub fn subPanel(self: *Spacer) SubPanel {
        return .{
            .label = "spacer",
            .constraint = .{ .fixed = 1 },
            .render_fn = &render,
            .ctx = @ptrCast(@alignCast(self)),
        };
    }

    fn render(
        _: *anyopaque,
        _: *Terminal,
        _: Rect,
        _: *const Theme,
    ) anyerror!void {
        // Spacer renders nothing.
    }
};

// ── Tests ───────────────────────────────────────────────────────

test "Header.subPanel returns correct constraint and render_fn" {
    var header = Header{
        .title = "Test Dashboard",
        .style = .single,
    };
    const panel = header.subPanel();
    try std.testing.expectEqual(
        Constraint{ .fixed = 3 },
        panel.constraint,
    );
    try std.testing.expectEqualStrings("header", panel.label);
    try std.testing.expect(panel.render_fn == &Header.render);
}

test "StatusBar.subPanel returns correct constraint" {
    const items = [_]StatusBar.StatusItem{
        .{ .label = "Status", .value = "OK" },
    };
    var bar = StatusBar{
        .items = &items,
        .style = .single,
    };
    const panel = bar.subPanel();
    try std.testing.expectEqual(
        Constraint{ .fixed = 1 },
        panel.constraint,
    );
    try std.testing.expectEqualStrings("status_bar", panel.label);
}

test "Spacer.subPanel returns correct constraint" {
    var spacer = Spacer{};
    const panel = spacer.subPanel();
    try std.testing.expectEqual(
        Constraint{ .fixed = 1 },
        panel.constraint,
    );
    try std.testing.expectEqualStrings("spacer", panel.label);
}

test "renderStack with empty panels list" {
    // Should not crash with zero panels.
    var term = Terminal.init(std.testing.allocator);
    defer term.deinit();
    const rect = Rect.fromTerminalSize(80, 24);
    const theme = &themes.themes.default;
    try renderStack(&term, rect, theme, &.{});
}

test "renderRow with empty panels list" {
    var term = Terminal.init(std.testing.allocator);
    defer term.deinit();
    const rect = Rect.fromTerminalSize(80, 24);
    const theme = &themes.themes.default;
    try renderRow(&term, rect, theme, &.{});
}

test "SubPanel label assignment" {
    var header = Header{
        .title = "My Title",
        .style = .rounded,
    };
    const panel = header.subPanel();
    try std.testing.expectEqualStrings("header", panel.label);

    var spacer = Spacer{};
    const sp = spacer.subPanel();
    try std.testing.expectEqualStrings("spacer", sp.label);
}

test "Section.subPanel returns fill constraint" {
    const noop = struct {
        fn f(
            _: *anyopaque,
            _: *Terminal,
            _: Rect,
            _: *const Theme,
        ) anyerror!void {}
    }.f;

    var dummy: u8 = 0;
    var section = Section{
        .title = "Details",
        .content_fn = &noop,
        .content_ctx = @ptrCast(&dummy),
        .style = .single,
    };
    const panel = section.subPanel();
    try std.testing.expectEqual(
        Constraint{ .fill = {} },
        panel.constraint,
    );
    try std.testing.expectEqualStrings("section", panel.label);
}

test "renderStack constraint distribution" {
    // Verify that distribute is called with correct heights.
    // We test indirectly: fixed(3) + fill in 24 rows =>
    // first gets 3, second gets 21.
    var sizes: [2]u16 = undefined;
    layout.distribute(24, &.{
        .{ .fixed = 3 },
        .{ .fill = {} },
    }, &sizes);
    try std.testing.expectEqual(@as(u16, 3), sizes[0]);
    try std.testing.expectEqual(@as(u16, 21), sizes[1]);
}
