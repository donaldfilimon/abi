const std = @import("std");

pub const Status = enum { ready, busy, warning, disabled };
pub const Item = struct { label: []const u8, value: []const u8 };
pub const State = struct { title: []const u8, status: Status = .disabled, items: []const Item = &.{} };

pub const PaneKind = enum {
    system,
    plugins,
    storage,
    scheduler,
};

pub const DiagPane = struct {
    kind: PaneKind,
    title: []const u8,
    items: []const Item,
};

pub const DashboardState = struct {
    gpu_backend: []const u8 = "disabled",
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
};

pub const InteractiveTerminal = struct {
    fd: std.posix.fd_t,
    original: std.posix.termios,
    is_tty: bool,

    pub fn init(fd: std.posix.fd_t) !InteractiveTerminal {
        _ = fd;
        return error.NotATerminal;
    }

    pub fn deinit(self: *InteractiveTerminal) void {
        _ = self;
    }

    pub fn readKey(self: *InteractiveTerminal) ?u8 {
        _ = self;
        return null;
    }
};

pub const ScreenState = struct {
    width: u16,
    height: u16,
};

pub fn initScreen() !void {}
pub fn initScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn clearScreen() !void {}
pub fn clearScreenWriter(writer: anytype) !void {
    _ = writer;
}
pub fn render(state: ScreenState) !void {
    _ = state;
}
pub fn renderWriter(writer: anytype, state: ScreenState) !void {
    _ = writer;
    _ = state;
}
pub fn deinitScreen() void {}
pub fn deinitScreenWriter(writer: anytype) !void {
    _ = writer;
}

pub fn renderDashboard(allocator: std.mem.Allocator, state: State) ![]u8 {
    if (state.title.len == 0) return error.InvalidTuiState;
    return try allocator.dupe(u8, "TUI feature is disabled");
}

pub fn renderDiagnostics(allocator: std.mem.Allocator, ds: DashboardState) ![]u8 {
    _ = ds;
    return try allocator.dupe(u8, "TUI diagnostics are disabled in this build");
}

pub fn isQuitKey(byte: u8) bool {
    return byte == 'q' or byte == 'Q' or byte == 0x1b;
}

pub fn isRefreshKey(byte: u8) bool {
    return byte == 'r' or byte == 'R';
}

test {
    std.testing.refAllDecls(@This());
}

pub fn statusText(status: Status) []const u8 {
    return switch (status) {
        .ready => "ready",
        .busy => "busy",
        .warning => "warning",
        .disabled => "disabled",
    };
}
