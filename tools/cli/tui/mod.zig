pub const events = @import("events.zig");
pub const terminal = @import("terminal.zig");
pub const widgets = @import("widgets.zig");
pub const themes = @import("themes.zig");

pub const Key = events.Key;
pub const KeyCode = events.KeyCode;
pub const Modifiers = events.Modifiers;
pub const Event = events.Event;
pub const Mouse = events.Mouse;
pub const MouseButton = events.MouseButton;

pub const Terminal = terminal.Terminal;
pub const TerminalSize = terminal.TerminalSize;
pub const PlatformCapabilities = terminal.PlatformCapabilities;

// Widgets
pub const ProgressIndicator = widgets.ProgressIndicator;
pub const ProgressBar = widgets.ProgressBar;
pub const Dialog = widgets.Dialog;
pub const DialogResult = widgets.DialogResult;
pub const CommandPreview = widgets.CommandPreview;
pub const Toast = widgets.Toast;
pub const SpinnerStyle = widgets.SpinnerStyle;

// Themes
pub const Theme = themes.Theme;
pub const ThemeManager = themes.ThemeManager;
pub const builtinThemes = themes.themes;
