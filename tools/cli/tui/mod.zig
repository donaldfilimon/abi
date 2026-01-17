pub const events = @import("events.zig");
pub const terminal = @import("terminal.zig");

pub const Key = events.Key;
pub const KeyCode = events.KeyCode;
pub const Modifiers = events.Modifiers;
pub const Event = events.Event;
pub const Mouse = events.Mouse;
pub const MouseButton = events.MouseButton;

pub const Terminal = terminal.Terminal;
pub const TerminalSize = terminal.TerminalSize;
