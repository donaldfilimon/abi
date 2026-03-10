//! Create Subagent wizard panel for the unified TUI dashboard.
//!
//! Displays the Cursor subagent creation workflow and prompts.
//! Run the actual creation via CLI: abi create-subagent [options] [name].

const std = @import("std");
const terminal = @import("../terminal");
const layout = @import("../layout");
const themes = @import("../themes");
const events = @import("../events");
const Panel = @import("../panel");

pub const CreateSubagentPanel = struct {
    allocator: std.mem.Allocator,
    tick_count: u64,

    pub fn init(allocator: std.mem.Allocator) CreateSubagentPanel {
        return .{
            .allocator = allocator,
            .tick_count = 0,
        };
    }

    pub fn render(self: *CreateSubagentPanel, term: *terminal.Terminal, rect: layout.Rect, theme: *const themes.Theme) anyerror!void {
        _ = self;
        var y = rect.y;
        const col: u16 = rect.x + 2;

        try term.moveTo(y, rect.x);
        try term.write(theme.bold);
        try term.write(theme.primary);
        try term.write(" Create Cursor Subagent");
        try term.write(theme.reset);
        y += 2;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("Create a .md file in .cursor/agents/ (project) or ~/.cursor/agents/ (user).");
        try term.write(theme.reset);
        y += 2;

        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("CLI (recommended)");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.info);
        try term.write("  abi create-subagent --name <name> --description \"When to delegate...\"");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text_dim);
        try term.write("  abi create-subagent -u -n my-agent -d \"Use proactively for X.\"");
        try term.write(theme.reset);
        y += 2;

        try term.moveTo(y, col);
        try term.write(theme.bold);
        try term.write(theme.accent);
        try term.write("Options");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  -p, --project   .cursor/agents/ (default)");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  -u, --user      ~/.cursor/agents/");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  -n, --name      Subagent id (lowercase, hyphens)");
        try term.write(theme.reset);
        y += 1;

        try term.moveTo(y, col);
        try term.write(theme.text);
        try term.write("  -d, --description  When to delegate (be specific)");
        try term.write(theme.reset);
        y += 2;

        try term.moveTo(y, col);
        try term.write(theme.success);
        try term.write("Run: abi create-subagent --help");
        try term.write(theme.reset);
    }

    pub fn tick(self: *CreateSubagentPanel) anyerror!void {
        _ = self;
    }

    pub fn handleEvent(_: *CreateSubagentPanel, _: events.Event) anyerror!bool {
        return false;
    }

    pub fn name(_: *CreateSubagentPanel) []const u8 {
        return "Create Subagent";
    }

    pub fn shortcutHint(_: *CreateSubagentPanel) []const u8 {
        return "F10";
    }

    pub fn deinit(_: *CreateSubagentPanel) void {}

    pub fn asPanel(self: *CreateSubagentPanel) Panel {
        return Panel.from(CreateSubagentPanel, self);
    }
};

test "create_subagent_panel init and name" {
    var panel = CreateSubagentPanel.init(std.testing.allocator);
    try std.testing.expectEqualStrings("Create Subagent", panel.name());
    try std.testing.expectEqualStrings("F10", panel.shortcutHint());
}

test {
    std.testing.refAllDecls(@This());
}
