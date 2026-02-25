//! Ralph agent loop status panel for the unified TUI dashboard.
//!
//! Reads state from `.ralph/state.json` and `.ralph/skills.jsonl`
//! to display loop status, run history, skill count, and gate results.

const std = @import("std");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const render_utils = @import("../render_utils.zig");
const Panel = @import("../panel.zig");

pub const RalphPanel = struct {
    allocator: std.mem.Allocator,
    tick_count: u64,

    // State from filesystem
    total_runs: u64,
    last_run_ts: i64,
    last_gate_passed: bool,
    is_locked: bool,
    skill_count: u32,
    max_iterations: u32,
    state_loaded: bool,
    last_poll: u64,

    const POLL_INTERVAL = 50; // ~5 seconds at 10 FPS

    pub fn init(allocator: std.mem.Allocator) RalphPanel {
        return .{
            .allocator = allocator,
            .tick_count = 0,
            .total_runs = 0,
            .last_run_ts = 0,
            .last_gate_passed = false,
            .is_locked = false,
            .skill_count = 0,
            .max_iterations = 5,
            .state_loaded = false,
            .last_poll = 0,
        };
    }

    fn loadState(self: *RalphPanel) void {
        // Use C-level file I/O to avoid needing std.Io backend.
        // Reads .ralph/state.json for run count, gate status, and lock state.
        self.loadStateJson();
        self.is_locked = fileExists(".ralph/lock");
        self.skill_count = countLines(".ralph/skills.jsonl");
    }

    fn loadStateJson(self: *RalphPanel) void {
        const data = readSmallFile(".ralph/state.json", self.allocator, 4096) orelse return;
        defer self.allocator.free(data);
        if (std.json.parseFromSlice(StateJson, self.allocator, data, .{ .ignore_unknown_fields = true })) |parsed| {
            defer parsed.deinit();
            self.total_runs = parsed.value.runs;
            self.last_run_ts = parsed.value.last_run_ts;
            self.last_gate_passed = parsed.value.last_gate_passed;
            self.state_loaded = true;
        } else |_| {}
    }

    /// Read a small file using C fopen/fread (no Zig I/O backend needed).
    fn readSmallFile(path: [*:0]const u8, allocator: std.mem.Allocator, max_size: usize) ?[]u8 {
        const f = std.c.fopen(path, "r") orelse return null;
        defer _ = std.c.fclose(f);
        const buf = allocator.alloc(u8, max_size) catch return null;
        const n = std.c.fread(buf.ptr, 1, max_size, f);
        if (n == 0) {
            allocator.free(buf);
            return null;
        }
        // Shrink to actual size so caller can free the returned slice correctly
        return allocator.realloc(buf, n) catch buf;
    }

    fn fileExists(path: [*:0]const u8) bool {
        const f = std.c.fopen(path, "r") orelse return false;
        _ = std.c.fclose(f);
        return true;
    }

    fn countLines(path: [*:0]const u8) u32 {
        const f = std.c.fopen(path, "r") orelse return 0;
        defer _ = std.c.fclose(f);
        var buf: [4096]u8 = undefined;
        var count: u32 = 0;
        while (true) {
            const n = std.c.fread(&buf, 1, buf.len, f);
            if (n == 0) break;
            for (buf[0..n]) |c| {
                if (c == '\n') count += 1;
            }
        }
        return count;
    }

    const StateJson = struct {
        runs: u64 = 0,
        last_run_ts: i64 = 0,
        last_gate_passed: bool = false,
    };

    pub fn render(self: *RalphPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        var y = rect.y;
        const col: u16 = rect.x + 2;
        var buf: [128]u8 = undefined;

        // Title
        try term.moveTo(y, rect.x);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" Ralph Agent Loop");
        try term.write(theme.reset);
        y += 2;

        if (!self.state_loaded) {
            try term.moveTo(y, col);
            try term.write(theme.text_dim);
            try term.write("No .ralph/ workspace found. Run: abi ralph init");
            try term.write(theme.reset);
            return;
        }

        // Loop status
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Loop Status");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  Status: ");
        if (self.is_locked) {
            try term.write(theme.accent);
            try term.write("RUNNING");
        } else {
            try term.write(theme.text_dim);
            try term.write("IDLE");
        }
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const runs_str = std.fmt.bufPrint(&buf, "  Total Runs: {d}", .{self.total_runs}) catch "";
        try term.write(runs_str);
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const iter_str = std.fmt.bufPrint(&buf, "  Max Iterations: {d}", .{self.max_iterations}) catch "";
        try term.write(iter_str);
        try term.write(theme.reset);
        y += 2;

        // Quality gate
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Quality Gate");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  Last Gate: ");
        if (self.total_runs == 0) {
            try term.write(theme.text_muted);
            try term.write("no runs yet");
        } else if (self.last_gate_passed) {
            try term.write(theme.success);
            try term.write("PASSED");
        } else {
            try term.write(theme.@"error");
            try term.write("FAILED");
        }
        try term.write(theme.reset);
        y += 1;

        if (self.last_run_ts > 0) {
            try term.moveTo(y, col);
            try term.write(theme.text_dim);
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.REALTIME, &ts);
            const now: i64 = @intCast(ts.sec);
            const ago = @as(u64, @intCast(@max(0, now - self.last_run_ts)));
            if (ago < 60) {
                const ts_str = std.fmt.bufPrint(&buf, "  Last Run: {d}s ago", .{ago}) catch "";
                try term.write(ts_str);
            } else if (ago < 3600) {
                const ts_str = std.fmt.bufPrint(&buf, "  Last Run: {d}m ago", .{ago / 60}) catch "";
                try term.write(ts_str);
            } else {
                const ts_str = std.fmt.bufPrint(&buf, "  Last Run: {d}h ago", .{ago / 3600}) catch "";
                try term.write(ts_str);
            }
            try term.write(theme.reset);
        }
        y += 2;

        // Skills memory
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Skills Memory");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const skills_str = std.fmt.bufPrint(&buf, "  Stored Skills: {d}", .{self.skill_count}) catch "";
        try term.write(skills_str);
        try term.write(theme.reset);
    }

    pub fn tick(self: *RalphPanel) anyerror!void {
        self.tick_count += 1;

        // Poll filesystem periodically
        if (self.tick_count - self.last_poll >= POLL_INTERVAL or !self.state_loaded) {
            self.loadState();
            self.last_poll = self.tick_count;
        }
    }

    pub fn handleEvent(_: *RalphPanel, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *RalphPanel) []const u8 {
        return "Ralph";
    }

    pub fn shortcutHint(_: *RalphPanel) []const u8 {
        return "F8";
    }

    pub fn deinit(_: *RalphPanel) void {}

    pub fn asPanel(self: *RalphPanel) Panel {
        return Panel.from(RalphPanel, self);
    }
};

// Tests
test "ralph_panel init" {
    var panel = RalphPanel.init(std.testing.allocator);
    try std.testing.expectEqualStrings("Ralph", panel.name());
    try std.testing.expectEqualStrings("F8", panel.shortcutHint());
    try std.testing.expect(!panel.state_loaded);
}

test "ralph_panel tick triggers load" {
    var panel = RalphPanel.init(std.testing.allocator);
    try panel.tick();
    try std.testing.expectEqual(@as(u64, 1), panel.tick_count);
    try std.testing.expectEqual(@as(u64, 1), panel.last_poll);
}

test {
    std.testing.refAllDecls(@This());
}
