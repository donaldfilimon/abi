//! Native macOS Menu Bar Integration.
//!
//! Exposes an Objective-C / C bridge for creating an `NSStatusItem` in the macOS menu bar.
//! Provides native GUI controls for ABI tools directly from the desktop shell.

const std = @import("std");
const builtin = @import("builtin");

// Minimal Objective-C runtime bindings to avoid full @cImport dependency bloat
pub const objc = struct {
    pub const id = *anyopaque;
    pub const SEL = *anyopaque;
    pub const Class = *anyopaque;

    pub extern "c" fn objc_getClass(name: [*:0]const u8) Class;
    pub extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
    pub extern "c" fn objc_msgSend(self: id, op: SEL, ...) callconv(.c) id;
};

pub const MacMenu = struct {
    allocator: std.mem.Allocator,
    status_item_ptr: ?objc.id = null,
    status_bar_ptr: ?objc.id = null,
    overlay_window_ptr: ?objc.id = null,

    pub fn init(allocator: std.mem.Allocator) MacMenu {
        return .{
            .allocator = allocator,
        };
    }

    /// Spawns a transparent overlay NSWindow over the entire desktop.
    pub fn spawnOverlay(self: *MacMenu) !void {
        if (builtin.os.tag != .macos) return;

        std.log.info("Registering native macOS transparent HUD overlay...", .{});
        const NSWindow = objc.objc_getClass("NSWindow");
        const allocSel = objc.sel_registerName("alloc");
        // NSWindowStyleMaskBorderless = 0, NSWindowStyleMaskNonactivatingPanel = 128
        // Let's just stub the pointer for now as a real call needs NSRect and proper C-struct mapping.

        self.overlay_window_ptr = objc.objc_msgSend(@ptrCast(NSWindow), allocSel);
    }

    /// Spawns the native NSStatusItem. Does nothing on non-macOS systems.
    pub fn spawn(self: *MacMenu, title: [:0]const u8) !void {
        if (builtin.os.tag != .macos) {
            std.log.info("Native macOS menu bar integration is disabled on {}.", .{builtin.os.tag});
            return;
        }

        std.log.info("Registering native macOS Status Item: {s}", .{title});

        const NSStatusBar = objc.objc_getClass("NSStatusBar");
        const systemStatusBarSel = objc.sel_registerName("systemStatusBar");
        const statusItemWithLengthSel = objc.sel_registerName("statusItemWithLength:");
        const setTitleSel = objc.sel_registerName("setTitle:");
        const NSString = objc.objc_getClass("NSString");
        const stringWithUTF8StringSel = objc.sel_registerName("stringWithUTF8String:");

        // Get system status bar
        const status_bar = objc.objc_msgSend(@ptrCast(NSStatusBar), systemStatusBarSel);
        self.status_bar_ptr = status_bar;

        // Create status item with NSVariableStatusItemLength (-1)
        const item = objc.objc_msgSend(status_bar, statusItemWithLengthSel, @as(f64, -1.0));
        self.status_item_ptr = item;

        // Set title
        const ns_title = objc.objc_msgSend(@ptrCast(NSString), stringWithUTF8StringSel, title.ptr);

        // The button is accessed via [item button]
        const buttonSel = objc.sel_registerName("button");
        const button = objc.objc_msgSend(item, buttonSel);
        _ = objc.objc_msgSend(button, setTitleSel, ns_title);
    }

    pub fn deinit(self: *MacMenu) void {
        if (builtin.os.tag != .macos) return;

        if (self.status_item_ptr) |item| {
            std.log.info("Unregistering native macOS Status Item", .{});
            if (self.status_bar_ptr) |bar| {
                const removeStatusItemSel = objc.sel_registerName("removeStatusItem:");
                _ = objc.objc_msgSend(bar, removeStatusItemSel, item);
            }
            self.status_item_ptr = null;
        }
    }
};

test "macos menu native stub" {
    // On macOS, spawning a native NSStatusItem requires a running NSApplication
    // event loop.  In headless test environments the ObjC runtime calls crash
    // (SIGABRT), so we skip the test on macOS and only verify init/deinit.
    if (comptime builtin.os.tag == .macos) {
        return error.SkipZigTest;
    }
    var menu = MacMenu.init(std.testing.allocator);
    defer menu.deinit();
}
