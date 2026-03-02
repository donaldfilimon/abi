//! Terminal UI Chat Panel
//!
//! Provides a split-pane multi-persona chat interface.
//! Left pane: Switchable personas (custom local AIs)
//! Right pane: Interaction buffer (Voice/Text/Vision context aware)

const std = @import("std");
const tui = @import("../mod.zig");
const themes = @import("../themes.zig");

pub const ChatPanel = struct {
    allocator: std.mem.Allocator,
    term: ?*tui.Terminal = null,
    theme: *const themes.Theme,
    
    personas: std.ArrayListUnmanaged([]const u8) = .empty,
    active_persona_idx: usize = 0,
    
    messages: std.ArrayListUnmanaged([]const u8) = .empty,
    input_buffer: std.ArrayListUnmanaged(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator, term: *tui.Terminal, theme: *const themes.Theme) !ChatPanel {
        var out: ChatPanel = .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
        };
        try out.personas.append(allocator, "Default AI");
        try out.personas.append(allocator, "Code Expert");
        try out.personas.append(allocator, "Analyzer");
        return out;
    }

    pub fn deinit(self: *ChatPanel) void {
        self.personas.deinit(self.allocator);
        for (self.messages.items) |msg| {
            self.allocator.free(msg);
        }
        self.messages.deinit(self.allocator);
        self.input_buffer.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn update(self: *ChatPanel) !void {
        _ = self;
        // Background polling or message reception goes here
    }

    pub fn handleEvent(self: *ChatPanel, event: tui.Event) anyerror!bool {
        switch (event) {
            .key => |key| switch (key.code) {
                .tab => {
                    self.active_persona_idx = (self.active_persona_idx + 1) % self.personas.items.len;
                    return true;
                },
                .enter => {
                    if (self.input_buffer.items.len > 0) {
                        const msg = try self.allocator.dupe(u8, self.input_buffer.items);
                        try self.messages.append(self.allocator, msg);
                        self.input_buffer.clearRetainingCapacity();
                    }
                    return true;
                },
                .backspace => {
                    if (self.input_buffer.items.len > 0) {
                        _ = self.input_buffer.pop();
                    }
                    return true;
                },
                .character => {
                    if (key.char) |ch| {
                        if (ch >= 32 and ch <= 126) {
                            try self.input_buffer.append(self.allocator, ch);
                            return true;
                        }
                    }
                },
                else => {},
            },
            else => {},
        }
        return false;
    }

    pub fn render(self: *ChatPanel, y: usize, x: usize, width: usize, height: usize) anyerror!void {
        const term = self.term orelse return; // Assume we add this
        const rect = tui.Rect{
            .x = @intCast(x),
            .y = @intCast(y),
            .width = @intCast(width),
            .height = @intCast(height)
        };
        const theme = self.theme;

        if (rect.isEmpty() or rect.width < 10) return;

        // Split: 25% Personas, 75% Chat
        const left_w = @max(rect.width / 4, 15);
        const right_w = rect.width - left_w;

        // Render Left Pane (Personas)
        const left_rect = tui.Rect{ .x = rect.x, .y = rect.y, .width = left_w, .height = rect.height };
        try tui.render_utils.drawBox(term, left_rect, .rounded, theme);
        
        try term.moveTo(rect.y + 1, rect.x + 2);
        try term.write(theme.text_dim);
        try term.write("PERSONAS (Tab)");
        try term.write(theme.reset);

        var y_offset: u16 = 3;
        for (self.personas.items, 0..) |p, i| {
            try term.moveTo(rect.y + y_offset, rect.x + 2);
            if (i == self.active_persona_idx) {
                try term.write(theme.selection_bg);
                try term.write(theme.selection_fg);
                try term.write(" > ");
            } else {
                try term.write("   ");
            }
            try term.write(p);
            try term.write(theme.reset);
            y_offset += 2;
        }

        // Render Right Pane (Chat History + Input)
        const right_rect = tui.Rect{ .x = rect.x + left_w, .y = rect.y, .width = right_w, .height = rect.height };
        try tui.render_utils.drawBox(term, right_rect, .rounded, theme);
        
        // Chat History (from bottom up, leaving room for input box)
        const input_h = 3;
        
        var hist_y = rect.y + rect.height - input_h - 2;
        
        // Simple reversed iteration for drawing
        var i: usize = self.messages.items.len;
        while (i > 0 and (hist_y > rect.y + 1)) : (i -= 1) {
            const msg = self.messages.items[i - 1];
            try term.moveTo(hist_y, right_rect.x + 2);
            try term.write(theme.info);
            try term.write("User: ");
            try term.write(theme.reset);
            try term.write(msg);
            hist_y -|= 2;
        }

        // Input Box
        const input_rect = tui.Rect{
            .x = right_rect.x + 1,
            .y = rect.y + rect.height - input_h - 1,
            .width = right_rect.width - 2,
            .height = input_h
        };
        try tui.render_utils.drawBox(term, input_rect, .single, theme);
        try term.moveTo(input_rect.y + 1, input_rect.x + 1);
        try term.write("> ");
        try term.write(self.input_buffer.items);
    }
};

test {
    std.testing.refAllDecls(@This());
}
