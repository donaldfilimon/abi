//! TUI Theme System
//!
//! Provides customizable color schemes for the TUI interface.
//! Includes built-in themes: default, monokai, solarized, nord, gruvbox.

const std = @import("std");

// ═══════════════════════════════════════════════════════════════════════════
// Theme Definition
// ═══════════════════════════════════════════════════════════════════════════

pub const Theme = struct {
    name: []const u8,

    // Primary colors
    primary: []const u8,
    secondary: []const u8,
    accent: []const u8,

    // UI elements
    border: []const u8,
    selection_bg: []const u8,
    selection_fg: []const u8,

    // Text colors
    text: []const u8,
    text_dim: []const u8,
    text_muted: []const u8,

    // Status colors
    success: []const u8,
    warning: []const u8,
    @"error": []const u8,
    info: []const u8,

    // Category colors
    category_ai: []const u8,
    category_data: []const u8,
    category_system: []const u8,
    category_tools: []const u8,
    category_meta: []const u8,

    // Common codes
    reset: []const u8 = "\x1b[0m",
    bold: []const u8 = "\x1b[1m",
    dim: []const u8 = "\x1b[2m",
    italic: []const u8 = "\x1b[3m",
    underline: []const u8 = "\x1b[4m",

    // Semantic styles
    header: []const u8 = "\x1b[1m", // Bold for headers by default
};

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Themes
// ═══════════════════════════════════════════════════════════════════════════

pub const themes = struct {
    /// Default theme - Cyan-focused with vibrant colors
    pub const default = Theme{
        .name = "default",
        .primary = "\x1b[36m", // Cyan
        .secondary = "\x1b[34m", // Blue
        .accent = "\x1b[33m", // Yellow
        .border = "\x1b[36m", // Cyan
        .selection_bg = "\x1b[44m", // Blue bg
        .selection_fg = "\x1b[97m", // Bright white
        .text = "\x1b[37m", // White
        .text_dim = "\x1b[2m", // Dim
        .text_muted = "\x1b[90m", // Bright black
        .success = "\x1b[32m", // Green
        .warning = "\x1b[33m", // Yellow
        .@"error" = "\x1b[31m", // Red
        .info = "\x1b[36m", // Cyan
        .category_ai = "\x1b[35m", // Magenta
        .category_data = "\x1b[34m", // Blue
        .category_system = "\x1b[33m", // Yellow
        .category_tools = "\x1b[32m", // Green
        .category_meta = "\x1b[36m", // Cyan
    };

    /// Monokai theme - Warm and cozy colors
    pub const monokai = Theme{
        .name = "monokai",
        .primary = "\x1b[38;5;81m", // Cyan
        .secondary = "\x1b[38;5;141m", // Purple
        .accent = "\x1b[38;5;221m", // Yellow
        .border = "\x1b[38;5;59m", // Gray
        .selection_bg = "\x1b[48;5;237m", // Dark gray bg
        .selection_fg = "\x1b[38;5;231m", // White
        .text = "\x1b[38;5;231m", // White
        .text_dim = "\x1b[38;5;242m", // Gray
        .text_muted = "\x1b[38;5;59m", // Dark gray
        .success = "\x1b[38;5;148m", // Green
        .warning = "\x1b[38;5;221m", // Yellow
        .@"error" = "\x1b[38;5;197m", // Pink/Red
        .info = "\x1b[38;5;81m", // Cyan
        .category_ai = "\x1b[38;5;141m", // Purple
        .category_data = "\x1b[38;5;81m", // Cyan
        .category_system = "\x1b[38;5;221m", // Yellow
        .category_tools = "\x1b[38;5;148m", // Green
        .category_meta = "\x1b[38;5;208m", // Orange
    };

    /// Solarized Dark theme
    pub const solarized = Theme{
        .name = "solarized",
        .primary = "\x1b[38;5;37m", // Cyan
        .secondary = "\x1b[38;5;33m", // Blue
        .accent = "\x1b[38;5;136m", // Yellow
        .border = "\x1b[38;5;240m", // Base01
        .selection_bg = "\x1b[48;5;236m", // Base02
        .selection_fg = "\x1b[38;5;230m", // Base3
        .text = "\x1b[38;5;187m", // Base1
        .text_dim = "\x1b[38;5;240m", // Base01
        .text_muted = "\x1b[38;5;239m", // Base00
        .success = "\x1b[38;5;64m", // Green
        .warning = "\x1b[38;5;136m", // Yellow
        .@"error" = "\x1b[38;5;160m", // Red
        .info = "\x1b[38;5;37m", // Cyan
        .category_ai = "\x1b[38;5;125m", // Magenta
        .category_data = "\x1b[38;5;33m", // Blue
        .category_system = "\x1b[38;5;136m", // Yellow
        .category_tools = "\x1b[38;5;64m", // Green
        .category_meta = "\x1b[38;5;37m", // Cyan
    };

    /// Nord theme - Arctic, north-bluish colors
    pub const nord = Theme{
        .name = "nord",
        .primary = "\x1b[38;5;110m", // Nord8 (cyan)
        .secondary = "\x1b[38;5;111m", // Nord9 (blue)
        .accent = "\x1b[38;5;222m", // Nord13 (yellow)
        .border = "\x1b[38;5;60m", // Nord3
        .selection_bg = "\x1b[48;5;60m", // Nord3
        .selection_fg = "\x1b[38;5;255m", // Snow
        .text = "\x1b[38;5;255m", // Snow Storm
        .text_dim = "\x1b[38;5;60m", // Nord3
        .text_muted = "\x1b[38;5;59m", // Nord2
        .success = "\x1b[38;5;108m", // Nord14 (green)
        .warning = "\x1b[38;5;222m", // Nord13 (yellow)
        .@"error" = "\x1b[38;5;174m", // Nord11 (red)
        .info = "\x1b[38;5;110m", // Nord8 (cyan)
        .category_ai = "\x1b[38;5;139m", // Nord15 (purple)
        .category_data = "\x1b[38;5;111m", // Nord9 (blue)
        .category_system = "\x1b[38;5;222m", // Nord13 (yellow)
        .category_tools = "\x1b[38;5;108m", // Nord14 (green)
        .category_meta = "\x1b[38;5;110m", // Nord8 (cyan)
    };

    /// Gruvbox Dark theme - Retro groove colors
    pub const gruvbox = Theme{
        .name = "gruvbox",
        .primary = "\x1b[38;5;108m", // Aqua
        .secondary = "\x1b[38;5;109m", // Blue
        .accent = "\x1b[38;5;214m", // Yellow
        .border = "\x1b[38;5;239m", // Gray
        .selection_bg = "\x1b[48;5;237m", // Bg1
        .selection_fg = "\x1b[38;5;223m", // Fg0
        .text = "\x1b[38;5;223m", // Fg1
        .text_dim = "\x1b[38;5;245m", // Gray
        .text_muted = "\x1b[38;5;239m", // Bg3
        .success = "\x1b[38;5;142m", // Green
        .warning = "\x1b[38;5;214m", // Yellow
        .@"error" = "\x1b[38;5;167m", // Red
        .info = "\x1b[38;5;108m", // Aqua
        .category_ai = "\x1b[38;5;175m", // Purple
        .category_data = "\x1b[38;5;109m", // Blue
        .category_system = "\x1b[38;5;214m", // Yellow
        .category_tools = "\x1b[38;5;142m", // Green
        .category_meta = "\x1b[38;5;208m", // Orange
    };

    /// High contrast theme - Accessibility focused
    pub const high_contrast = Theme{
        .name = "high_contrast",
        .primary = "\x1b[97m", // Bright white
        .secondary = "\x1b[96m", // Bright cyan
        .accent = "\x1b[93m", // Bright yellow
        .border = "\x1b[97m", // Bright white
        .selection_bg = "\x1b[107m", // Bright white bg
        .selection_fg = "\x1b[30m", // Black
        .text = "\x1b[97m", // Bright white
        .text_dim = "\x1b[37m", // White
        .text_muted = "\x1b[90m", // Bright black
        .success = "\x1b[92m", // Bright green
        .warning = "\x1b[93m", // Bright yellow
        .@"error" = "\x1b[91m", // Bright red
        .info = "\x1b[96m", // Bright cyan
        .category_ai = "\x1b[95m", // Bright magenta
        .category_data = "\x1b[96m", // Bright cyan
        .category_system = "\x1b[93m", // Bright yellow
        .category_tools = "\x1b[92m", // Bright green
        .category_meta = "\x1b[97m", // Bright white
    };

    /// Minimal theme - Low color, distraction-free
    pub const minimal = Theme{
        .name = "minimal",
        .primary = "\x1b[37m", // White
        .secondary = "\x1b[37m", // White
        .accent = "\x1b[1m", // Bold only
        .border = "\x1b[90m", // Gray
        .selection_bg = "\x1b[7m", // Inverse
        .selection_fg = "", // Default
        .text = "\x1b[37m", // White
        .text_dim = "\x1b[90m", // Gray
        .text_muted = "\x1b[90m", // Gray
        .success = "\x1b[32m", // Green
        .warning = "\x1b[33m", // Yellow
        .@"error" = "\x1b[31m", // Red
        .info = "\x1b[37m", // White
        .category_ai = "\x1b[37m", // White
        .category_data = "\x1b[37m", // White
        .category_system = "\x1b[37m", // White
        .category_tools = "\x1b[37m", // White
        .category_meta = "\x1b[90m", // Gray
    };
};

// ═══════════════════════════════════════════════════════════════════════════
// Theme Manager
// ═══════════════════════════════════════════════════════════════════════════

pub const ThemeManager = struct {
    current: *const Theme,
    available: []const *const Theme,

    const all_themes = [_]*const Theme{
        &themes.default,
        &themes.monokai,
        &themes.solarized,
        &themes.nord,
        &themes.gruvbox,
        &themes.high_contrast,
        &themes.minimal,
    };

    pub fn init() ThemeManager {
        return .{
            .current = &themes.default,
            .available = &all_themes,
        };
    }

    pub fn setTheme(self: *ThemeManager, name: []const u8) bool {
        for (self.available) |theme| {
            if (std.mem.eql(u8, theme.name, name)) {
                self.current = theme;
                return true;
            }
        }
        return false;
    }

    pub fn nextTheme(self: *ThemeManager) void {
        for (self.available, 0..) |theme, i| {
            if (theme == self.current) {
                const next_idx = (i + 1) % self.available.len;
                self.current = self.available[next_idx];
                return;
            }
        }
    }

    pub fn prevTheme(self: *ThemeManager) void {
        for (self.available, 0..) |theme, i| {
            if (theme == self.current) {
                const prev_idx = if (i == 0) self.available.len - 1 else i - 1;
                self.current = self.available[prev_idx];
                return;
            }
        }
    }

    pub fn getThemeNames(self: *const ThemeManager) [7][]const u8 {
        var names: [7][]const u8 = undefined;
        for (self.available, 0..) |theme, i| {
            names[i] = theme.name;
        }
        return names;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "theme manager initialization" {
    var manager = ThemeManager.init();
    try std.testing.expectEqualStrings("default", manager.current.name);
}

test "theme switching" {
    var manager = ThemeManager.init();

    const success = manager.setTheme("nord");
    try std.testing.expect(success);
    try std.testing.expectEqualStrings("nord", manager.current.name);

    const fail = manager.setTheme("nonexistent");
    try std.testing.expect(!fail);
    try std.testing.expectEqualStrings("nord", manager.current.name);
}

test "theme cycling" {
    var manager = ThemeManager.init();
    try std.testing.expectEqualStrings("default", manager.current.name);

    manager.nextTheme();
    try std.testing.expectEqualStrings("monokai", manager.current.name);

    manager.prevTheme();
    try std.testing.expectEqualStrings("default", manager.current.name);

    manager.prevTheme();
    try std.testing.expectEqualStrings("minimal", manager.current.name);
}
