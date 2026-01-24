pub const Modifiers = packed struct {
    ctrl: bool = false,
    alt: bool = false,
    shift: bool = false,
};

pub const KeyCode = enum {
    character,
    enter,
    escape,
    backspace,
    delete,
    tab,
    up,
    down,
    left,
    right,
    home,
    end,
    page_up,
    page_down,
    ctrl_c,
};

pub const Key = struct {
    code: KeyCode,
    char: ?u8 = null,
    mods: Modifiers = .{},
};

pub const MouseButton = enum {
    left,
    middle,
    right,
    wheel_up,
    wheel_down,
    none,
};

pub const Mouse = struct {
    row: u16,
    col: u16,
    button: MouseButton,
    pressed: bool,
};

pub const Event = union(enum) {
    key: Key,
    mouse: Mouse,
};

pub fn isChar(key: Key, ch: u8) bool {
    return key.code == .character and key.char != null and key.char.? == ch;
}
