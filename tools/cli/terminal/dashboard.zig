//! Generic TUI Dashboard
//!
//! Comptime-generic `Dashboard(PanelType)` that eliminates copy-pasted
//! DashboardState structs across dashboard command files. The generic owns
//! state fields, init/deinit, theme management, notification system,
//! common key handling, render chrome, and the full AsyncLoop lifecycle.

const std = @import("std");
const abi = @import("abi");
const events = @import("events.zig");
const terminal_mod = @import("terminal.zig");
const themes_mod = @import("themes.zig");
const async_loop_mod = @import("async_loop.zig");
const keybindings = @import("keybindings.zig");

/// Create a generic dashboard wrapper around any panel type.
///
/// `PanelType` must provide (duck-typed at comptime):
///   - `fn deinit(*PanelType) void`
///   - `fn update(*PanelType) !void`
///   - `fn render(*PanelType, usize, usize, usize, usize) !void`
///   - field `theme: *const themes_mod.Theme`
pub fn Dashboard(comptime PanelType: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        terminal: *terminal_mod.Terminal,
        theme_manager: themes_mod.ThemeManager,
        panel: PanelType,
        term_size: terminal_mod.TerminalSize,
        paused: bool,
        show_help: bool,
        frame_count: u64,
        notification: ?[]const u8,
        notification_time: i64,
        config: Config,
        extra_key_handler: ?*const fn (*Self, events.Key) bool,

        pub const Config = struct {
            title: []const u8,
            refresh_rate_ms: u32 = 250,
            help_keys: []const u8 = " [q]uit  [p]ause  [t]heme  [?]help",
            min_width: u16 = 40,
            min_height: u16 = 10,
        };

        pub fn init(
            allocator: std.mem.Allocator,
            terminal: *terminal_mod.Terminal,
            initial_theme: *const themes_mod.Theme,
            panel: PanelType,
            config: Config,
        ) Self {
            var tm = themes_mod.ThemeManager.init();
            tm.current = initial_theme;
            return .{
                .allocator = allocator,
                .terminal = terminal,
                .theme_manager = tm,
                .panel = panel,
                .term_size = terminal.size(),
                .paused = false,
                .show_help = false,
                .frame_count = 0,
                .notification = null,
                .notification_time = 0,
                .config = config,
                .extra_key_handler = null,
            };
        }

        pub fn deinit(self: *Self) void {
            self.panel.deinit();
        }

        pub fn theme(self: *const Self) *const themes_mod.Theme {
            return self.theme_manager.current;
        }

        pub fn showNotification(self: *Self, msg: []const u8) void {
            self.notification = msg;
            self.notification_time = abi.services.shared.utils.unixMs();
        }

        fn clearExpiredNotification(self: *Self) void {
            if (self.notification_time > 0) {
                const elapsed = abi.services.shared.utils.unixMs() - self.notification_time;
                if (elapsed > 3000) {
                    self.notification = null;
                    self.notification_time = 0;
                }
            }
        }

        fn updateTheme(self: *Self) void {
            self.panel.theme = self.theme_manager.current;
        }

        /// Run the dashboard event loop. Blocks until the user quits.
        pub fn run(self: *Self) !void {
            var loop = async_loop_mod.AsyncLoop.init(self.allocator, self.terminal, .{
                .refresh_rate_ms = self.config.refresh_rate_ms,
                .input_poll_ms = 16,
                .auto_resize = true,
            });
            defer loop.deinit();

            loop.setRenderCallback(&renderCallback);
            loop.setTickCallback(&tickCallback);
            loop.setUpdateCallback(&updateCallback);
            loop.setUserData(@ptrCast(self));

            try loop.run();
        }

        fn renderCallback(loop: *async_loop_mod.AsyncLoop) anyerror!void {
            const self: *Self = loop.getUserData(Self) orelse return error.UserDataNotSet;
            try self.terminal.clear();
            try self.renderChrome();
            self.frame_count = loop.getFrameCount();
        }

        fn tickCallback(loop: *async_loop_mod.AsyncLoop) anyerror!void {
            const self: *Self = loop.getUserData(Self) orelse return error.UserDataNotSet;
            self.term_size = self.terminal.size();
            self.clearExpiredNotification();
            if (!self.paused) {
                try self.panel.update();
            }
        }

        fn updateCallback(loop: *async_loop_mod.AsyncLoop, event: async_loop_mod.AsyncEvent) anyerror!bool {
            const self: *Self = loop.getUserData(Self) orelse return error.UserDataNotSet;
            return switch (event) {
                .input => |ev| switch (ev) {
                    .key => |key| self.handleKey(key),
                    .mouse => false,
                },
                .resize => |sz| blk: {
                    self.term_size = .{ .rows = sz.rows, .cols = sz.cols };
                    break :blk false;
                },
                .quit => true,
                else => false,
            };
        }

        fn handleKey(self: *Self, key: events.Key) bool {
            // Help overlay intercepts all keys
            if (self.show_help) {
                switch (key.code) {
                    .escape, .enter => self.show_help = false,
                    .character => {
                        if (key.char) |ch| {
                            if (ch == 'h' or ch == 'q') self.show_help = false;
                        }
                    },
                    else => {},
                }
                return false;
            }

            // Common keybindings
            const action = keybindings.resolve(key);
            switch (action) {
                .quit => return true,
                .pause => {
                    self.paused = !self.paused;
                    self.showNotification(if (self.paused) "Paused" else "Resumed");
                },
                .theme_next => {
                    self.theme_manager.nextTheme();
                    self.updateTheme();
                    self.showNotification("Theme changed");
                },
                .theme_prev => {
                    self.theme_manager.prevTheme();
                    self.updateTheme();
                    self.showNotification("Theme changed");
                },
                .help_toggle => self.show_help = true,
                .none => {
                    // Delegate to panel-specific key handler
                    if (self.extra_key_handler) |handler| {
                        return handler(self, key);
                    }
                },
            }
            return false;
        }

        fn renderChrome(self: *Self) !void {
            const term = self.terminal;
            const th = self.theme();
            const width = self.term_size.cols;
            const height = self.term_size.rows;

            if (width < self.config.min_width or height < self.config.min_height) {
                try term.moveTo(height / 2, 0);
                try term.write(th.warning);
                var buf: [64]u8 = undefined;
                const msg = std.fmt.bufPrint(&buf, "Resize terminal to at least {d}x{d}", .{
                    self.config.min_width, self.config.min_height,
                }) catch "Resize terminal";
                try term.write(msg);
                try term.write(th.reset);
                return;
            }

            // Title bar
            try term.moveTo(0, 0);
            try term.write(th.bold);
            try term.write(th.primary);
            try term.write(" ");
            try term.write(self.config.title);
            try term.write(" ");
            try term.write(th.reset);

            // Panel content area — u16 widens to usize implicitly
            try self.panel.render(2, 0, width, height -| 4);

            // Notification line
            if (self.notification) |msg| {
                try term.moveTo(height -| 2, 2);
                try term.write(th.info);
                try term.write(msg);
                try term.write(th.reset);
            }

            // Help/status bar
            try term.moveTo(height -| 1, 0);
            try term.write(th.text_dim);
            try term.write(self.config.help_keys);
            try term.write(th.reset);
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "Dashboard type instantiation" {
    // Verify that Dashboard can be instantiated with a mock panel type
    const MockPanel = struct {
        theme: *const themes_mod.Theme,
        updated: bool = false,

        fn deinit(_: *@This()) void {}
        fn update(self: *@This()) !void {
            self.updated = true;
        }
        fn render(_: *@This(), _: usize, _: usize, _: usize, _: usize) !void {}
    };

    const DashType = Dashboard(MockPanel);
    // Verify the type has the expected fields
    try std.testing.expect(@hasField(DashType, "terminal"));
    try std.testing.expect(@hasField(DashType, "paused"));
    try std.testing.expect(@hasField(DashType, "show_help"));
    try std.testing.expect(@hasField(DashType, "notification"));
    try std.testing.expect(@hasField(DashType, "frame_count"));
    try std.testing.expect(@hasField(DashType, "config"));
}

test "Dashboard.Config defaults" {
    const MockPanel = struct {
        theme: *const themes_mod.Theme,
        fn deinit(_: *@This()) void {}
        fn update(_: *@This()) !void {}
        fn render(_: *@This(), _: usize, _: usize, _: usize, _: usize) !void {}
    };

    const DashType = Dashboard(MockPanel);
    const cfg = DashType.Config{ .title = "Test" };
    try std.testing.expectEqual(@as(u32, 250), cfg.refresh_rate_ms);
    try std.testing.expectEqual(@as(u16, 40), cfg.min_width);
    try std.testing.expectEqual(@as(u16, 10), cfg.min_height);
}

test "keybindings integration" {
    // Verify that keybindings module is accessible from dashboard context
    const action = keybindings.resolve(.{ .code = .character, .char = 'q' });
    try std.testing.expectEqual(keybindings.KeyAction.quit, action);
}

test {
    std.testing.refAllDecls(@This());
}
