//! Integration Tests: Desktop Feature
//!
//! Verifies the desktop module exports, lifecycle queries, context
//! management, and macOS menu type availability through the public
//! `abi.desktop` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const desktop = abi.desktop;

// ---------------------------------------------------------------------------
// Feature gate
// ---------------------------------------------------------------------------

test "desktop: isEnabled reflects feature flag" {
    if (build_options.feat_desktop) {
        try std.testing.expect(desktop.isEnabled());
    } else {
        try std.testing.expect(!desktop.isEnabled());
    }
}

test "desktop: isInitialized reflects feature flag" {
    if (build_options.feat_desktop) {
        try std.testing.expect(desktop.isInitialized());
    } else {
        try std.testing.expect(!desktop.isInitialized());
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

test "desktop: DesktopError type is accessible" {
    const E = desktop.DesktopError;
    const err: E = error.PlatformUnsupported;
    try std.testing.expect(err == error.PlatformUnsupported);
}

test "desktop: Error alias matches DesktopError" {
    try std.testing.expect(desktop.Error == desktop.DesktopError);
}

// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------

test "desktop: Context init and deinit" {
    var ctx = desktop.Context.init(std.testing.allocator);
    defer ctx.deinit();

    if (build_options.feat_desktop) {
        try std.testing.expect(ctx.initialized);
    } else {
        try std.testing.expect(!ctx.initialized);
    }
}

test "desktop: Context stores allocator" {
    var ctx = desktop.Context.init(std.testing.allocator);
    defer ctx.deinit();
    _ = ctx.allocator;
}

// ---------------------------------------------------------------------------
// macOS menu types
// ---------------------------------------------------------------------------

test "desktop: macos_menu namespace exists" {
    const menu_ns = desktop.macos_menu;
    _ = menu_ns;
}

test "desktop: macos_menu MacMenu type available when enabled" {
    if (build_options.feat_desktop) {
        const MacMenu = desktop.macos_menu.MacMenu;
        var menu = MacMenu.init(std.testing.allocator);
        defer menu.deinit();
        try std.testing.expect(menu.status_item_ptr == null);
        try std.testing.expect(menu.status_bar_ptr == null);
        try std.testing.expect(menu.overlay_window_ptr == null);
    }
}

test "desktop: types namespace is accessible" {
    const T = desktop.types;
    const E = T.DesktopError;
    const err: E = error.IntegrationFailed;
    try std.testing.expect(err == error.IntegrationFailed);
}

test {
    std.testing.refAllDecls(@This());
}
