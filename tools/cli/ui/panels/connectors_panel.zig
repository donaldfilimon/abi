//! LLM connector availability panel for the unified TUI dashboard.
//!
//! Displays the status of all 16 LLM provider connectors by checking
//! environment variable configuration. Each connector follows the
//! ABI_<PROVIDER>_API_KEY pattern (with documented exceptions).

const std = @import("std");
const terminal = @import("../core/terminal.zig");
const layout = @import("../core/layout.zig");
const themes = @import("../core/themes.zig");
const events = @import("../core/events.zig");
const render_utils = @import("../core/render_utils.zig");
const Panel = @import("../core/panel.zig");

pub const ConnectorsPanel = struct {
    allocator: std.mem.Allocator,
    tick_count: u64,
    scroll_offset: usize,

    const Provider = struct {
        name: []const u8,
        env_key: [:0]const u8,
        note: []const u8,
    };

    const providers = [_]Provider{
        .{ .name = "OpenAI", .env_key = "ABI_OPENAI_API_KEY", .note = "" },
        .{ .name = "Anthropic", .env_key = "ABI_ANTHROPIC_API_KEY", .note = "" },
        .{ .name = "Gemini", .env_key = "ABI_GEMINI_API_KEY", .note = "" },
        .{ .name = "HuggingFace", .env_key = "ABI_HF_API_TOKEN", .note = "NOT _API_KEY" },
        .{ .name = "Ollama", .env_key = "ABI_OLLAMA_HOST", .note = "local, no key" },
        .{ .name = "LM Studio", .env_key = "ABI_LM_STUDIO_HOST", .note = "local, no key" },
        .{ .name = "vLLM", .env_key = "ABI_VLLM_HOST", .note = "local, no key" },
        .{ .name = "MLX", .env_key = "ABI_MLX_HOST", .note = "local, no key" },
        .{ .name = "Llama.cpp", .env_key = "ABI_LLAMA_CPP_HOST", .note = "local, no key" },
        .{ .name = "Mistral", .env_key = "ABI_MISTRAL_API_KEY", .note = "" },
        .{ .name = "Cohere", .env_key = "ABI_COHERE_API_KEY", .note = "" },
        .{ .name = "Discord", .env_key = "DISCORD_BOT_TOKEN", .note = "no ABI_ prefix" },
        .{ .name = "Codex", .env_key = "ABI_OPENAI_API_KEY", .note = "uses OpenAI" },
        .{ .name = "OpenCode", .env_key = "ABI_OPENAI_API_KEY", .note = "uses OpenAI" },
        .{ .name = "Ollama PT", .env_key = "ABI_OLLAMA_PASSTHROUGH_URL", .note = "uses _URL" },
    };

    // Scheduler is always available (no env check needed)
    const total_providers = providers.len + 1;

    pub fn init(allocator: std.mem.Allocator) ConnectorsPanel {
        return .{
            .allocator = allocator,
            .tick_count = 0,
            .scroll_offset = 0,
        };
    }

    fn checkEnv(key: [:0]const u8) bool {
        const ptr = std.c.getenv(key.ptr) orelse return false;
        const val = std.mem.sliceTo(ptr, 0);
        return val.len > 0;
    }

    pub fn render(self: *ConnectorsPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        var y = rect.y;
        const col: u16 = rect.x + 2;
        var buf: [64]u8 = undefined;

        // Title
        try term.moveTo(y, rect.x);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" LLM Connectors (16 providers)");
        try term.write(theme.reset);
        y += 2;

        // Count configured
        var configured: usize = 1; // Scheduler always available
        for (&providers) |p| {
            if (checkEnv(p.env_key)) configured += 1;
        }

        try term.moveTo(y, col);
        try term.write(theme.text);
        const summary = std.fmt.bufPrint(&buf, "Configured: {d}/{d}", .{ configured, total_providers }) catch "";
        try term.write(summary);
        try term.write(theme.reset);
        y += 2;

        // Header
        try term.moveTo(y, col);
        try term.write(theme.text_dim);
        try term.write("Provider       Status    Note");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text_muted);
        try render_utils.writeRepeat(term, "\u{2500}", 40);
        try term.write(theme.reset);
        y += 1;

        // Provider list
        const max_visible = @as(usize, rect.height) -| 8;
        const visible = @min(providers.len - self.scroll_offset, max_visible);
        for (providers[self.scroll_offset..][0..visible]) |p| {
            const has_key = checkEnv(p.env_key);

            try term.moveTo(y, col);

            if (has_key) {
                try term.write(theme.success);
                try term.write("\u{25cf}");
            } else {
                try term.write(theme.text_muted);
                try term.write("\u{25cb}");
            }
            try term.write(" ");

            // Name (padded to 14 chars)
            try term.write(theme.text);
            try term.write(p.name);
            const pad = if (p.name.len < 14) 14 - p.name.len else 1;
            try render_utils.writeRepeat(term, " ", pad);

            if (has_key) {
                try term.write(theme.success);
                try term.write("ready   ");
            } else {
                try term.write(theme.text_muted);
                try term.write("not set ");
            }

            if (p.note.len > 0) {
                try term.write(theme.text_dim);
                try term.write(p.note);
            }
            try term.write(theme.reset);
            y += 1;
        }

        // Scheduler (always available)
        if (self.scroll_offset + visible >= providers.len) {
            try term.moveTo(y, col);
            try term.write(theme.success);
            try term.write("\u{25cf} ");
            try term.write(theme.text);
            try term.write("Scheduler     ");
            try term.write(theme.success);
            try term.write("ready   ");
            try term.write(theme.text_dim);
            try term.write("always available");
            try term.write(theme.reset);
            y += 1;
        }

        // Scroll hint
        if (self.scroll_offset + visible < providers.len) {
            y += 1;
            try term.moveTo(y, col);
            try term.write(theme.text_dim);
            try term.write("\u{2193} scroll for more (j/k or arrows)");
            try term.write(theme.reset);
        }
    }

    pub fn tick(self: *ConnectorsPanel) anyerror!void {
        self.tick_count += 1;
    }

    pub fn handleEvent(self: *ConnectorsPanel, event: events.Event) anyerror!bool {
        switch (event) {
            .key => |key| switch (key.code) {
                .down => {
                    if (self.scroll_offset + 1 < providers.len) self.scroll_offset += 1;
                    return true;
                },
                .up => {
                    if (self.scroll_offset > 0) self.scroll_offset -= 1;
                    return true;
                },
                .character => {
                    if (key.char) |ch| {
                        switch (ch) {
                            'j' => {
                                if (self.scroll_offset + 1 < providers.len) self.scroll_offset += 1;
                                return true;
                            },
                            'k' => {
                                if (self.scroll_offset > 0) self.scroll_offset -= 1;
                                return true;
                            },
                            else => return false,
                        }
                    }
                    return false;
                },
                else => return false,
            },
            else => return false,
        }
    }

    pub fn name(_: *ConnectorsPanel) []const u8 {
        return "Connectors";
    }

    pub fn shortcutHint(_: *ConnectorsPanel) []const u8 {
        return "F7";
    }

    pub fn deinit(_: *ConnectorsPanel) void {}

    pub fn asPanel(self: *ConnectorsPanel) Panel {
        return Panel.from(ConnectorsPanel, self);
    }
};

// Tests
test "connectors_panel init" {
    var panel = ConnectorsPanel.init(std.testing.allocator);
    try std.testing.expectEqualStrings("Connectors", panel.name());
    try std.testing.expectEqualStrings("F7", panel.shortcutHint());
    try panel.tick();
    try std.testing.expectEqual(@as(u64, 1), panel.tick_count);
}

test "connectors_panel scroll" {
    var panel = ConnectorsPanel.init(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), panel.scroll_offset);

    _ = try panel.handleEvent(.{ .key = .{ .code = .down } });
    try std.testing.expectEqual(@as(usize, 1), panel.scroll_offset);

    _ = try panel.handleEvent(.{ .key = .{ .code = .up } });
    try std.testing.expectEqual(@as(usize, 0), panel.scroll_offset);

    _ = try panel.handleEvent(.{ .key = .{ .code = .up } });
    try std.testing.expectEqual(@as(usize, 0), panel.scroll_offset);
}

test {
    std.testing.refAllDecls(@This());
}
