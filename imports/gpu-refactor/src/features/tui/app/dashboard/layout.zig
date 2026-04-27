const std = @import("std");
const types = @import("../../types.zig");
const Rect = types.Rect;
const state_mod = @import("state.zig");
const LayoutMode = state_mod.LayoutMode;

pub const DashboardLayout = struct {
    mode: LayoutMode,
    full: Rect,
    header: Rect,
    nav: Rect,
    summary: Rect,
    detail: Rect,
    footer: Rect,
    overlay: Rect,
};

pub fn emptyRect() Rect {
    return .{};
}

pub fn insetRect(area: Rect, pad_x: u16, pad_y: u16) Rect {
    return .{
        .x = area.x + @min(pad_x, area.width),
        .y = area.y + @min(pad_y, area.height),
        .width = area.width -| (@min(pad_x, area.width) * 2),
        .height = area.height -| (@min(pad_y, area.height) * 2),
    };
}

pub fn centeredRect(area: Rect, width: u16, height: u16) Rect {
    const w = @min(width, area.width);
    const h = @min(height, area.height);
    return .{
        .x = area.x + (area.width -| w) / 2,
        .y = area.y + (area.height -| h) / 2,
        .width = w,
        .height = h,
    };
}

pub fn rowRect(area: Rect, row: u16) Rect {
    if (row >= area.height) return emptyRect();
    return .{
        .x = area.x,
        .y = area.y + row,
        .width = area.width,
        .height = 1,
    };
}

pub fn classifyLayout(full: Rect) LayoutMode {
    if (full.width < 18 or full.height < 7) return .minimal;
    if (full.width < 52 or full.height < 14) return .compact;
    if (full.width < 96 or full.height < 24) return .medium;
    return .wide;
}

pub fn computeLayout(full: Rect) DashboardLayout {
    const mode = classifyLayout(full);
    if (mode == .minimal) {
        return .{
            .mode = mode,
            .full = full,
            .header = full,
            .nav = emptyRect(),
            .summary = emptyRect(),
            .detail = emptyRect(),
            .footer = emptyRect(),
            .overlay = full,
        };
    }

    const header_height: u16 = if (mode == .compact) @min(@as(u16, 3), full.height) else @min(@as(u16, 4), full.height);
    const top_split = full.splitHorizontal(header_height);
    const footer_height: u16 = if (top_split.bottom.height > 0) 1 else 0;
    const body_and_footer = top_split.bottom.splitHorizontal(top_split.bottom.height -| footer_height);
    const body = body_and_footer.top;
    const footer = if (footer_height == 0) emptyRect() else body_and_footer.bottom;

    return switch (mode) {
        .wide => blk: {
            const nav_width = @min(@as(u16, 24), @max(@as(u16, 18), body.width / 4));
            const nav_split = body.splitVertical(nav_width);
            const summary_height = @min(@as(u16, 6), @max(@as(u16, 5), body.height / 4));
            const summary_split = nav_split.right.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.left,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = centeredRect(full, full.width -| 10, full.height -| 6),
            };
        },
        .medium => blk: {
            const nav_width = @min(@as(u16, 20), @max(@as(u16, 16), body.width / 4));
            const nav_split = body.splitVertical(nav_width);
            const summary_height = @min(@as(u16, 5), @max(@as(u16, 4), body.height / 4));
            const summary_split = nav_split.right.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.left,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = centeredRect(full, full.width -| 6, full.height -| 4),
            };
        },
        .compact => blk: {
            const nav_height: u16 = if (body.height >= 4) 2 else 1;
            const nav_split = body.splitHorizontal(nav_height);
            const summary_height: u16 = if (nav_split.bottom.height > 1) 1 else 0;
            const summary_split = nav_split.bottom.splitHorizontal(summary_height);
            break :blk .{
                .mode = mode,
                .full = full,
                .header = top_split.top,
                .nav = nav_split.top,
                .summary = summary_split.top,
                .detail = summary_split.bottom,
                .footer = footer,
                .overlay = insetRect(full, 1, 1),
            };
        },
        .minimal => unreachable,
    };
}
