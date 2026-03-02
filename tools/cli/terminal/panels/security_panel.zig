//! Security infrastructure status panel for the unified TUI dashboard.
//!
//! Displays simulated security metrics: rate limiting, audit events,
//! session management, and threat indicators. In a full deployment,
//! these would connect to the security managers in services/shared/security/.

const std = @import("std");
const terminal = @import("../terminal.zig");
const layout = @import("../layout.zig");
const themes = @import("../themes.zig");
const events = @import("../events.zig");
const render_utils = @import("../render_utils.zig");
const Panel = @import("../panel.zig");

pub const SecurityPanel = struct {
    allocator: std.mem.Allocator,
    tick_count: u64,

    // Simulated metrics
    total_requests: u64,
    blocked_requests: u64,
    active_sessions: u32,
    audit_events: u32,
    threat_level: ThreatLevel,

    // Rate limit sparkline (last 20 ticks)
    rate_history: [20]u8,
    rate_idx: usize,

    const ThreatLevel = enum {
        low,
        moderate,
        elevated,
        high,
        critical,

        fn label(self: ThreatLevel) []const u8 {
            return switch (self) {
                .low => "LOW",
                .moderate => "MODERATE",
                .elevated => "ELEVATED",
                .high => "HIGH",
                .critical => "CRITICAL",
            };
        }
    };

    pub fn init(allocator: std.mem.Allocator) SecurityPanel {
        return .{
            .allocator = allocator,
            .tick_count = 0,
            .total_requests = 14523,
            .blocked_requests = 37,
            .active_sessions = 8,
            .audit_events = 156,
            .threat_level = .low,
            .rate_history = [_]u8{20} ** 20,
            .rate_idx = 0,
        };
    }

    pub fn render(self: *SecurityPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        var y = rect.y;
        const col: u16 = rect.x + 2;
        var buf: [128]u8 = undefined;

        // Title
        try term.moveTo(y, rect.x);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" Security Infrastructure");
        try term.write(theme.reset);
        y += 2;

        // Threat level indicator
        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("Threat Level: ");
        const threat_color = switch (self.threat_level) {
            .low => theme.success,
            .moderate => theme.info,
            .elevated => theme.warning,
            .high, .critical => theme.@"error",
        };
        try term.write(threat_color);
        try term.write(theme.bold);
        try term.write(self.threat_level.label());
        try term.write(theme.reset);
        y += 2;

        // Rate limiting section
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Rate Limiting");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const req_str = std.fmt.bufPrint(&buf, "  Total Requests: {d}", .{self.total_requests}) catch "";
        try term.write(req_str);
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.@"error");
        const blk_str = std.fmt.bufPrint(&buf, "  Blocked:        {d} ({d:.1}%)", .{
            self.blocked_requests,
            if (self.total_requests > 0)
                @as(f64, @floatFromInt(self.blocked_requests)) / @as(f64, @floatFromInt(self.total_requests)) * 100.0
            else
                0.0,
        }) catch "";
        try term.write(blk_str);
        try term.write(theme.reset);
        y += 1;

        // Sparkline
        try term.moveTo(y, col);
        try term.write(theme.text_dim);
        try term.write("  Rate: ");
        const bars = [_][]const u8{ "\u{2581}", "\u{2582}", "\u{2583}", "\u{2584}", "\u{2585}", "\u{2586}", "\u{2587}", "\u{2588}" };
        for (0..20) |i| {
            const idx = (self.rate_idx + i) % 20;
            const bar_idx = @min(7, self.rate_history[idx] / 13);
            try term.write(theme.info);
            try term.write(bars[bar_idx]);
        }
        try term.write(theme.reset);
        y += 2;

        // Sessions section
        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Sessions & Auth");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const sess_str = std.fmt.bufPrint(&buf, "  Active Sessions: {d}", .{self.active_sessions}) catch "";
        try term.write(sess_str);
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        const audit_str = std.fmt.bufPrint(&buf, "  Audit Events:    {d}", .{self.audit_events}) catch "";
        try term.write(audit_str);
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.success);
        try term.write("  RBAC: active  JWT: valid  HMAC-chain: intact");
        try term.write(theme.reset);
    }

    pub fn tick(self: *SecurityPanel) anyerror!void {
        self.tick_count += 1;

        // Simulate traffic
        self.total_requests += 7 + (self.tick_count % 5);
        if (self.tick_count % 23 == 0) self.blocked_requests += 1;
        if (self.tick_count % 50 == 0 and self.active_sessions > 2) self.active_sessions -= 1;
        if (self.tick_count % 30 == 0) self.active_sessions += 1;
        if (self.tick_count % 10 == 0) self.audit_events += 1;

        // Update rate sparkline
        const rate: u8 = @intCast(20 + (self.tick_count * 7 + self.tick_count / 3) % 60);
        self.rate_history[self.rate_idx] = rate;
        self.rate_idx = (self.rate_idx + 1) % 20;

        // Threat level based on block rate
        self.threat_level = if (self.blocked_requests > self.total_requests / 20)
            .elevated
        else if (self.blocked_requests > self.total_requests / 50)
            .moderate
        else
            .low;
    }

    pub fn handleEvent(_: *SecurityPanel, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *SecurityPanel) []const u8 {
        return "Security";
    }

    pub fn shortcutHint(_: *SecurityPanel) []const u8 {
        return "F6";
    }

    pub fn deinit(_: *SecurityPanel) void {}

    pub fn asPanel(self: *SecurityPanel) Panel {
        return Panel.from(SecurityPanel, self);
    }
};

// Tests
test "security_panel init and tick" {
    var panel = SecurityPanel.init(std.testing.allocator);
    try panel.tick();
    try std.testing.expect(panel.tick_count == 1);
    try std.testing.expect(panel.total_requests > 14523);
    try std.testing.expectEqualStrings("Security", panel.name());
}

test "security_panel threat level" {
    var panel = SecurityPanel.init(std.testing.allocator);
    panel.total_requests = 100;
    panel.blocked_requests = 0;
    try panel.tick();
    try std.testing.expect(panel.threat_level == .low);

    panel.blocked_requests = 10;
    try panel.tick();
    try std.testing.expect(panel.threat_level == .elevated);
}

test {
    std.testing.refAllDecls(@This());
}
