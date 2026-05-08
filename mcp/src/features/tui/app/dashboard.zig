//! Dashboard application entry point.
//! Re-exports the decomposed dashboard logic.

const mod = @import("dashboard/mod.zig");
const state = @import("dashboard/state.zig");
const layout = @import("dashboard/layout.zig");
const types = @import("../types.zig");

pub const run = mod.run;
pub const renderDashboard = mod.renderDashboard;
pub const computeLayout = layout.computeLayout;
pub const handleKey = state.handleKey;

pub const View = state.View;
pub const FocusRegion = state.FocusRegion;
pub const LayoutMode = state.LayoutMode;
pub const AppState = state.AppState;
pub const DashboardAction = state.DashboardAction;
pub const DashboardLayout = layout.DashboardLayout;

pub fn hasVisibleCell(cells: []const types.Cell) bool {
    for (cells) |cell| {
        if (cell.char != ' ') return true;
    }
    return false;
}

pub fn containsText(cells: []const types.Cell, text: []const u8) bool {
    if (text.len == 0) return true;
    if (cells.len < text.len) return false;

    var start: usize = 0;
    while (start + text.len <= cells.len) : (start += 1) {
        var matched = true;
        for (text, 0..) |byte, idx| {
            if (cells[start + idx].char != byte) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}

test {
    @import("std").testing.refAllDecls(mod);
    @import("std").testing.refAllDecls(state);
    @import("std").testing.refAllDecls(layout);
    @import("std").testing.refAllDecls(@import("dashboard/widgets.zig"));
    @import("std").testing.refAllDecls(@import("dashboard/view_overview.zig"));
    @import("std").testing.refAllDecls(@import("dashboard/view_features.zig"));
    @import("std").testing.refAllDecls(@import("dashboard/view_runtime.zig"));
}
