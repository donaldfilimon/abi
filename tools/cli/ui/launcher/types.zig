//! TUI type definitions for the command launcher.
//!
//! Contains all core types shared across the TUI modules:
//! categories, actions, menu items, history, completion, and
//! box-drawing / color constants.

const std = @import("std");
const tui = @import("../core/mod.zig");

// ═══════════════════════════════════════════════════════════════════
// Categories
// ═══════════════════════════════════════════════════════════════════

pub const Category = enum {
    ai,
    data,
    system,
    tools,
    meta,

    pub fn icon(self: Category) []const u8 {
        return switch (self) {
            .ai => "\xF0\x9F\xA4\x96",
            .data => "\xF0\x9F\x92\xBE",
            .system => "\xE2\x9A\x99\xEF\xB8\x8F",
            .tools => "\xF0\x9F\x94\xA7",
            .meta => "\xF0\x9F\x93\x8B",
        };
    }

    pub fn name(self: Category) []const u8 {
        return switch (self) {
            .ai => "AI & ML",
            .data => "Data",
            .system => "System",
            .tools => "Tools",
            .meta => "Meta",
        };
    }

    pub fn color(self: Category) []const u8 {
        return switch (self) {
            .ai => "\x1b[35m", // Magenta
            .data => "\x1b[34m", // Blue
            .system => "\x1b[33m", // Yellow
            .tools => "\x1b[32m", // Green
            .meta => "\x1b[36m", // Cyan
        };
    }
};

// ═══════════════════════════════════════════════════════════════════
// Actions & Commands
// ═══════════════════════════════════════════════════════════════════

pub const CommandRef = struct {
    /// Stable id used by history/completion.
    id: []const u8,
    /// Top-level CLI command routed through descriptor dispatch.
    command: []const u8,
    /// Default argument vector for launcher execution.
    args: []const [:0]const u8,
};

pub const Action = union(enum) {
    command: CommandRef,
    version: void,
    help: void,
    quit: void,
};

// ═══════════════════════════════════════════════════════════════════
// Menu Items
// ═══════════════════════════════════════════════════════════════════

pub const MenuItem = struct {
    label: []const u8,
    description: []const u8,
    action: Action,
    category: Category,
    shortcut: ?u8 = null, // Quick launch key (1-9)
    usage: []const u8 = "", // Usage string for preview
    examples: []const []const u8 = &[_][]const u8{}, // Example commands
    related: []const []const u8 = &[_][]const u8{}, // Related commands

    pub fn categoryColor(self: *const MenuItem, theme: *const tui.Theme) []const u8 {
        return switch (self.category) {
            .ai => theme.category_ai,
            .data => theme.category_data,
            .system => theme.category_system,
            .tools => theme.category_tools,
            .meta => theme.category_meta,
        };
    }
};

// ═══════════════════════════════════════════════════════════════════
// History
// ═══════════════════════════════════════════════════════════════════

/// Command history entry
pub const HistoryEntry = struct {
    command_id: []const u8,
    timestamp: i64,
};

// ═══════════════════════════════════════════════════════════════════
// Completion
// ═══════════════════════════════════════════════════════════════════

/// Match type for completion scoring
pub const MatchType = enum {
    exact_prefix, // Exact prefix match (highest priority)
    fuzzy, // Fuzzy character match
    history_recent, // Recently used command
    substring, // Substring match (lowest priority)

    pub fn indicator(self: MatchType) []const u8 {
        return switch (self) {
            .exact_prefix => "\xE2\x89\xA1",
            .fuzzy => "\xE2\x89\x88",
            .history_recent => "\xE2\x86\xBA",
            .substring => "\xE2\x8A\x82",
        };
    }
};

/// Completion suggestion with ranking score
pub const CompletionSuggestion = struct {
    item_index: usize, // Index into MenuItems array
    score: u32, // Ranking score (higher = better)
    match_type: MatchType, // How the match was found
};

/// Completion state for the TUI
pub const CompletionState = struct {
    suggestions: std.ArrayListUnmanaged(CompletionSuggestion),
    selected_suggestion: usize, // Index into suggestions array
    active: bool, // Whether dropdown is shown
    max_visible: usize, // Max suggestions to show (default: 5)

    pub fn init() CompletionState {
        return .{
            .suggestions = .empty,
            .selected_suggestion = 0,
            .active = false,
            .max_visible = 5,
        };
    }

    pub fn deinit(self: *CompletionState, allocator: std.mem.Allocator) void {
        self.suggestions.deinit(allocator);
    }

    pub fn clear(self: *CompletionState) void {
        self.suggestions.clearRetainingCapacity();
        self.selected_suggestion = 0;
        self.active = false;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Box Drawing Characters
// ═══════════════════════════════════════════════════════════════════

pub const box = struct {
    pub const tl = "\u{256d}"; // Top-left
    pub const tr = "\u{256e}"; // Top-right
    pub const bl = "\u{2570}"; // Bottom-left
    pub const br = "\u{256f}"; // Bottom-right
    pub const h = "\u{2500}"; // Horizontal
    pub const v = "\u{2502}"; // Vertical
    pub const lsep = "\u{251c}"; // Left separator
    pub const rsep = "\u{2524}"; // Right separator
};

// ═══════════════════════════════════════════════════════════════════
// Colors
// ═══════════════════════════════════════════════════════════════════

pub const colors = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const underline = "\x1b[4m";

    pub const black = "\x1b[30m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";

    pub const bg_black = "\x1b[40m";
    pub const bg_blue = "\x1b[44m";
    pub const bg_cyan = "\x1b[46m";

    pub const bright_black = "\x1b[90m";
    pub const bright_white = "\x1b[97m";
};

test {
    std.testing.refAllDecls(@This());
}
