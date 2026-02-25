//! Vtable-based Panel interface for TUI dashboards.
//!
//! Provides runtime polymorphism over any type that implements
//! render, tick, handleEvent, name, shortcutHint, and deinit.

const std = @import("std");
const terminal = @import("terminal.zig");
const layout = @import("layout.zig");
const themes = @import("themes.zig");
const events = @import("events.zig");

const Panel = @This();

ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    render: *const fn (ptr: *anyopaque, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void,
    tick: *const fn (ptr: *anyopaque) anyerror!void,
    handleEvent: *const fn (ptr: *anyopaque, event: events.Event) anyerror!bool,
    name: *const fn (ptr: *anyopaque) []const u8,
    shortcutHint: *const fn (ptr: *anyopaque) []const u8,
    deinit: *const fn (ptr: *anyopaque) void,
};

pub fn render(self: Panel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
    return self.vtable.render(self.ptr, term, rect, theme);
}

pub fn tick(self: Panel) anyerror!void {
    return self.vtable.tick(self.ptr);
}

pub fn handleEvent(self: Panel, event: events.Event) anyerror!bool {
    return self.vtable.handleEvent(self.ptr, event);
}

pub fn getName(self: Panel) []const u8 {
    return self.vtable.name(self.ptr);
}

pub fn shortcutHint(self: Panel) []const u8 {
    return self.vtable.shortcutHint(self.ptr);
}

pub fn deinit(self: Panel) void {
    self.vtable.deinit(self.ptr);
}

/// Create a Panel from any pointer type that implements the required methods.
pub fn from(comptime T: type, obj: *T) Panel {
    return .{
        .ptr = @ptrCast(@alignCast(obj)),
        .vtable = &.{
            .render = &typeErasedRender(T),
            .tick = &typeErasedTick(T),
            .handleEvent = &typeErasedHandleEvent(T),
            .name = &typeErasedName(T),
            .shortcutHint = &typeErasedShortcutHint(T),
            .deinit = &typeErasedDeinit(T),
        },
    };
}

fn typeErasedRender(comptime T: type) fn (*anyopaque, *terminal.Terminal, layout.Rect, *const themes.Theme) anyerror!void {
    return struct {
        fn call(ptr: *anyopaque, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.render(term, rect, theme);
        }
    }.call;
}

fn typeErasedTick(comptime T: type) fn (*anyopaque) anyerror!void {
    return struct {
        fn call(ptr: *anyopaque) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.tick();
        }
    }.call;
}

fn typeErasedHandleEvent(comptime T: type) fn (*anyopaque, events.Event) anyerror!bool {
    return struct {
        fn call(ptr: *anyopaque, event: events.Event) anyerror!bool {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.handleEvent(event);
        }
    }.call;
}

fn typeErasedName(comptime T: type) fn (*anyopaque) []const u8 {
    return struct {
        fn call(ptr: *anyopaque) []const u8 {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.name();
        }
    }.call;
}

fn typeErasedShortcutHint(comptime T: type) fn (*anyopaque) []const u8 {
    return struct {
        fn call(ptr: *anyopaque) []const u8 {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.shortcutHint();
        }
    }.call;
}

fn typeErasedDeinit(comptime T: type) fn (*anyopaque) void {
    return struct {
        fn call(ptr: *anyopaque) void {
            const self: *T = @ptrCast(@alignCast(ptr));
            self.deinit();
        }
    }.call;
}

// ── Error Boundary Panel ────────────────────────────────────────
//
// Wraps any Panel with error-catching logic so that a single panel
// crashing does not kill the entire dashboard. On render error it
// displays an error message in the content area. On tick error it
// silently skips the update. On handleEvent error it returns false.

pub const ErrorBoundaryPanel = struct {
    inner: Panel,
    last_error: ?anyerror = null,
    error_count: u32 = 0,

    /// Maximum consecutive errors before the panel stops retrying tick/render
    /// and stays in degraded mode until the next successful call.
    const max_consecutive_errors: u32 = 50;

    // -- Panel vtable methods --

    pub fn render(self: *ErrorBoundaryPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        if (self.error_count >= max_consecutive_errors) {
            // In degraded mode -- show persistent error without retrying
            renderErrorFallback(self, term, rect, theme);
            return;
        }

        self.inner.render(term, rect, theme) catch |err| {
            self.last_error = err;
            self.error_count += 1;
            renderErrorFallback(self, term, rect, theme);
            return;
        };
        // Successful render clears consecutive error counter
        self.error_count = 0;
    }

    pub fn tick(self: *ErrorBoundaryPanel) anyerror!void {
        if (self.error_count >= max_consecutive_errors) return;

        self.inner.tick() catch |err| {
            self.last_error = err;
            self.error_count += 1;
            return; // Silently skip tick
        };
    }

    pub fn handleEvent(self: *ErrorBoundaryPanel, event: events.Event) anyerror!bool {
        return self.inner.handleEvent(event) catch |err| {
            self.last_error = err;
            self.error_count += 1;
            return false; // Treat as not consumed
        };
    }

    pub fn name(self: *ErrorBoundaryPanel) []const u8 {
        return self.inner.getName();
    }

    pub fn shortcutHint(self: *ErrorBoundaryPanel) []const u8 {
        return self.inner.shortcutHint();
    }

    pub fn deinit(self: *ErrorBoundaryPanel) void {
        self.inner.deinit();
    }

    /// Render a fallback error message in the panel content area.
    fn renderErrorFallback(self: *ErrorBoundaryPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) void {
        // Best-effort rendering -- if even the fallback fails, silently ignore
        const panel_name = self.inner.getName();
        const y_center = rect.y +| (rect.height / 2);
        const col: u16 = rect.x + 2;

        // Title line
        term.moveTo(y_center -| 1, col) catch return;
        term.write(theme.@"error") catch return;
        term.write(theme.bold) catch return;
        term.write("Panel Error") catch return;
        term.write(theme.reset) catch return;

        // Error detail line
        term.moveTo(y_center, col) catch return;
        term.write(theme.text) catch return;

        var buf: [128]u8 = undefined;
        if (self.last_error) |err| {
            const err_name = std.mem.sliceTo(@errorName(err), 0);
            const msg = std.fmt.bufPrint(&buf, "{s}: {s}", .{ panel_name, err_name }) catch panel_name;
            term.write(msg) catch return;
        } else {
            term.write(panel_name) catch return;
            term.write(": unknown error") catch return;
        }
        term.write(theme.reset) catch return;

        // Error count line
        term.moveTo(y_center +| 1, col) catch return;
        term.write(theme.text_dim) catch return;
        if (self.error_count >= max_consecutive_errors) {
            term.write("(panel suspended -- too many errors)") catch return;
        } else {
            const count_msg = std.fmt.bufPrint(&buf, "(errors: {d})", .{self.error_count}) catch "(errors: ?)";
            term.write(count_msg) catch return;
        }
        term.write(theme.reset) catch return;
    }

    /// Convert this ErrorBoundaryPanel into a type-erased Panel.
    pub fn asPanel(self: *ErrorBoundaryPanel) Panel {
        return Panel.from(ErrorBoundaryPanel, self);
    }

    /// Reset error state, allowing a suspended panel to retry.
    pub fn resetErrors(self: *ErrorBoundaryPanel) void {
        self.last_error = null;
        self.error_count = 0;
    }
};

/// Wrap a Panel in an ErrorBoundaryPanel for fault isolation.
/// The returned struct must be stored with stable memory (e.g., in an array
/// or heap allocation) before calling `.asPanel()`.
///
/// Usage:
///   var boundaries: [N]ErrorBoundaryPanel = undefined;
///   boundaries[i] = Panel.withErrorBoundary(some_panel);
///   panels[i] = boundaries[i].asPanel();
pub fn withErrorBoundary(inner: Panel) ErrorBoundaryPanel {
    return .{ .inner = inner };
}

// ── Noop Panel ──────────────────────────────────────────────────

const noop_vtable: VTable = .{
    .render = &struct {
        fn f(_: *anyopaque, _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {}
    }.f,
    .tick = &struct {
        fn f(_: *anyopaque) anyerror!void {}
    }.f,
    .handleEvent = &struct {
        fn f(_: *anyopaque, _: events.Event) anyerror!bool {
            return false;
        }
    }.f,
    .name = &struct {
        fn f(_: *anyopaque) []const u8 {
            return "noop";
        }
    }.f,
    .shortcutHint = &struct {
        fn f(_: *anyopaque) []const u8 {
            return "";
        }
    }.f,
    .deinit = &struct {
        fn f(_: *anyopaque) void {}
    }.f,
};

var noop_sentinel: u8 = 0;

pub const noop_panel: Panel = .{
    .ptr = @ptrCast(&noop_sentinel),
    .vtable = &noop_vtable,
};

// ── Tests ───────────────────────────────────────────────────────

test "noop_panel render and tick do not crash" {
    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();
    const rect = layout.Rect.fromTerminalSize(80, 24);
    const theme = &themes.themes.default;
    try noop_panel.render(&term, rect, theme);
    try noop_panel.tick();
}

test "noop_panel handleEvent returns false" {
    const consumed = try noop_panel.handleEvent(.{ .key = .{ .code = .escape } });
    try std.testing.expect(!consumed);
}

test "noop_panel name returns noop" {
    try std.testing.expectEqualStrings("noop", noop_panel.getName());
}

test "noop_panel shortcutHint returns empty" {
    try std.testing.expectEqualStrings("", noop_panel.shortcutHint());
}

test "from creates panel from concrete type" {
    const Mock = struct {
        rendered: bool = false,
        ticked: bool = false,

        pub fn render(self: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            self.rendered = true;
        }
        pub fn tick(self: *@This()) anyerror!void {
            self.ticked = true;
        }
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return true;
        }
        pub fn name(_: *@This()) []const u8 {
            return "mock";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "F1";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var mock: Mock = .{};
    const p = Panel.from(Mock, &mock);

    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();
    try p.render(&term, layout.Rect.fromTerminalSize(80, 24), &themes.themes.default);
    try p.tick();
    try std.testing.expect(mock.rendered);
    try std.testing.expect(mock.ticked);
    try std.testing.expectEqualStrings("mock", p.getName());
    try std.testing.expectEqualStrings("F1", p.shortcutHint());
    try std.testing.expect(try p.handleEvent(.{ .key = .{ .code = .enter } }));
}

// ── Error Boundary Tests ────────────────────────────────────────

test "error_boundary passes through successful render" {
    const Mock = struct {
        rendered: bool = false,

        pub fn render(self: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            self.rendered = true;
        }
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return true;
        }
        pub fn name(_: *@This()) []const u8 {
            return "ok-panel";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "F1";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var mock: Mock = .{};
    const inner = Panel.from(Mock, &mock);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();
    try p.render(&term, layout.Rect.fromTerminalSize(80, 24), &themes.themes.default);
    try std.testing.expect(mock.rendered);
    try std.testing.expectEqual(@as(u32, 0), boundary.error_count);
    try std.testing.expect(boundary.last_error == null);
}

test "error_boundary catches render error without crashing" {
    const Failing = struct {
        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            return error.OutOfMemory;
        }
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "fail-render";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var failing: Failing = .{};
    const inner = Panel.from(Failing, &failing);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();

    // Should NOT propagate the error -- catches it internally
    try p.render(&term, layout.Rect.fromTerminalSize(80, 24), &themes.themes.default);
    try std.testing.expectEqual(@as(u32, 1), boundary.error_count);
    try std.testing.expectEqual(@as(anyerror, error.OutOfMemory), boundary.last_error.?);
}

test "error_boundary catches tick error silently" {
    const Failing = struct {
        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {}
        pub fn tick(_: *@This()) anyerror!void {
            return error.ConnectionRefused;
        }
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "fail-tick";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var failing: Failing = .{};
    const inner = Panel.from(Failing, &failing);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    // Should NOT propagate the error
    try p.tick();
    try std.testing.expectEqual(@as(u32, 1), boundary.error_count);
    try std.testing.expectEqual(@as(anyerror, error.ConnectionRefused), boundary.last_error.?);
}

test "error_boundary catches handleEvent error and returns false" {
    const Failing = struct {
        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {}
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return error.Unexpected;
        }
        pub fn name(_: *@This()) []const u8 {
            return "fail-event";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var failing: Failing = .{};
    const inner = Panel.from(Failing, &failing);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    const consumed = try p.handleEvent(.{ .key = .{ .code = .enter } });
    try std.testing.expect(!consumed);
    try std.testing.expectEqual(@as(u32, 1), boundary.error_count);
}

test "error_boundary delegates name and shortcutHint" {
    const Mock = struct {
        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {}
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "test-panel";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "F5";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var mock: Mock = .{};
    const inner = Panel.from(Mock, &mock);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    try std.testing.expectEqualStrings("test-panel", p.getName());
    try std.testing.expectEqualStrings("F5", p.shortcutHint());
}

test "error_boundary resets error state" {
    const Failing = struct {
        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            return error.OutOfMemory;
        }
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "resettable";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var failing: Failing = .{};
    const inner = Panel.from(Failing, &failing);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();

    try p.render(&term, layout.Rect.fromTerminalSize(80, 24), &themes.themes.default);
    try std.testing.expectEqual(@as(u32, 1), boundary.error_count);

    boundary.resetErrors();
    try std.testing.expectEqual(@as(u32, 0), boundary.error_count);
    try std.testing.expect(boundary.last_error == null);
}

test "error_boundary suspends after max consecutive errors" {
    const Failing = struct {
        tick_calls: u32 = 0,

        pub fn render(_: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            return error.Broken;
        }
        pub fn tick(self: *@This()) anyerror!void {
            self.tick_calls += 1;
            return error.Broken;
        }
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "broken";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var failing: Failing = .{};
    const inner = Panel.from(Failing, &failing);
    var boundary = Panel.withErrorBoundary(inner);

    // Simulate reaching the max error threshold
    boundary.error_count = ErrorBoundaryPanel.max_consecutive_errors;

    const p = boundary.asPanel();

    // tick should not call inner when suspended
    try p.tick();
    try std.testing.expectEqual(@as(u32, 0), failing.tick_calls);

    // render should show error fallback without calling inner
    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();
    try p.render(&term, layout.Rect.fromTerminalSize(80, 24), &themes.themes.default);
    // error_count should stay the same (not increment further)
    try std.testing.expectEqual(ErrorBoundaryPanel.max_consecutive_errors, boundary.error_count);
}

test "error_boundary successful render resets error count" {
    const Flaky = struct {
        should_fail: bool = true,

        pub fn render(self: *@This(), _: *terminal.Terminal, _: layout.Rect, _: *const themes.Theme) anyerror!void {
            if (self.should_fail) return error.Temporary;
        }
        pub fn tick(_: *@This()) anyerror!void {}
        pub fn handleEvent(_: *@This(), _: events.Event) anyerror!bool {
            return false;
        }
        pub fn name(_: *@This()) []const u8 {
            return "flaky";
        }
        pub fn shortcutHint(_: *@This()) []const u8 {
            return "";
        }
        pub fn deinit(_: *@This()) void {}
    };

    var flaky: Flaky = .{};
    const inner = Panel.from(Flaky, &flaky);
    var boundary = Panel.withErrorBoundary(inner);
    const p = boundary.asPanel();

    var term = terminal.Terminal.init(std.testing.allocator);
    defer term.deinit();
    const rect = layout.Rect.fromTerminalSize(80, 24);
    const theme = &themes.themes.default;

    // First render fails
    try p.render(&term, rect, theme);
    try std.testing.expectEqual(@as(u32, 1), boundary.error_count);

    // Panel recovers
    flaky.should_fail = false;
    try p.render(&term, rect, theme);
    try std.testing.expectEqual(@as(u32, 0), boundary.error_count);
}

test {
    std.testing.refAllDecls(@This());
}
