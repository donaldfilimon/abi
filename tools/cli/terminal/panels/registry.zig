//! Unified dashboard panel registry.

pub const PanelSpec = struct {
    id: []const u8,
    label: []const u8,
    shortcut_hint: []const u8,
};

pub const panel_specs = [_]PanelSpec{
    .{ .id = "gpu", .label = "GPU", .shortcut_hint = "1" },
    .{ .id = "agent", .label = "Agent", .shortcut_hint = "2" },
    .{ .id = "train", .label = "Train", .shortcut_hint = "3" },
    .{ .id = "model", .label = "Model", .shortcut_hint = "4" },
    .{ .id = "stream", .label = "Stream", .shortcut_hint = "5" },
    .{ .id = "db", .label = "DB", .shortcut_hint = "6" },
    .{ .id = "net", .label = "Net", .shortcut_hint = "7" },
    .{ .id = "bench", .label = "Bench", .shortcut_hint = "8" },
    .{ .id = "brain", .label = "Brain", .shortcut_hint = "9" },
    .{ .id = "security", .label = "Security", .shortcut_hint = "F6" },
    .{ .id = "connectors", .label = "Connectors", .shortcut_hint = "F7" },
    .{ .id = "ralph", .label = "Ralph", .shortcut_hint = "F8" },
};

pub const tab_labels = [_][]const u8{
    "GPU",
    "Agent",
    "Train",
    "Model",
    "Stream",
    "DB",
    "Net",
    "Bench",
    "Brain",
    "Security",
    "Connectors",
    "Ralph",
};

test {
    @import("std").testing.refAllDecls(@This());
}
