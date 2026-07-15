const std = @import("std");

pub const Status = enum {
    ready,
    busy,
    warning,
    disabled,
};

pub const Item = struct {
    label: []const u8,
    value: []const u8,
};

pub const State = struct {
    title: []const u8,
    status: Status = .ready,
    items: []const Item = &.{},
};

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

pub const PaneKind = enum {
    system,
    plugins,
    storage,
    scheduler,
    memory,
    agent_output,
};

pub const FocusedPane = enum {
    left,
    right,
};

pub const DashboardPaneMeta = struct {
    kind: PaneKind,
    title: []const u8,
    hotkey: u8,
};

pub const DASHBOARD_PANES = [_]DashboardPaneMeta{
    .{ .kind = .system, .title = "System", .hotkey = '1' },
    .{ .kind = .plugins, .title = "Plugins", .hotkey = '2' },
    .{ .kind = .storage, .title = "WDBX Storage", .hotkey = '3' },
    .{ .kind = .scheduler, .title = "Scheduler", .hotkey = '4' },
    .{ .kind = .memory, .title = "Memory", .hotkey = '5' },
};

pub const DASHBOARD_PANE_COUNT: usize = DASHBOARD_PANES.len;

pub const DiagPane = struct {
    kind: PaneKind,
    title: []const u8,
    items: []const Item,
};

pub const DashboardState = struct {
    gpu_backend: []const u8 = "unknown",
    gpu_accelerated: bool = false,
    gpu_linked: bool = false,
    plugin_count: usize = 0,
    plugin_names: []const []const u8 = &.{},
    wdbx_blocks: usize = 0,
    wdbx_vectors: usize = 0,
    wdbx_entries: usize = 0,
    wdbx_spatial_records: usize = 0,
    scheduler_source: []const u8 = "not attached",
    scheduler_running: usize = 0,
    scheduler_pending: usize = 0,
    scheduler_completed: usize = 0,
    scheduler_failed: usize = 0,
    memory_source: []const u8 = "not attached",
    memory_peak: usize = 0,
    memory_current: usize = 0,
    memory_leaked: usize = 0,
    selected_pane: usize = 0,

    /// Split-pane agent output (owned buffer, last ~8KB of agent REPL output).
    agent_output_buffer: []const u8 = &.{},
    /// Scroll offset into agent_output_buffer lines.
    agent_output_scroll: usize = 0,
    /// Which pane currently receives keyboard input.
    focused_pane: FocusedPane = .left,
};

test {
    std.testing.refAllDecls(@This());
}
