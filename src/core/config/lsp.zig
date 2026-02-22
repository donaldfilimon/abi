//! LSP configuration (ZLS client)

const std = @import("std");

/// Configuration for the built-in LSP client (ZLS).
pub const LspConfig = struct {
    /// Path to the ZLS binary (default: "zls" in PATH).
    zls_path: []const u8 = "zls",
    /// Optional Zig compiler path to pass to ZLS initialization options.
    zig_exe_path: ?[]const u8 = null,
    /// Optional workspace root path for LSP initialization.
    workspace_root: ?[]const u8 = null,
    /// Log level passed to ZLS (info, warn, error, debug).
    log_level: []const u8 = "info",
    /// Enable snippet support in completion results.
    enable_snippets: bool = true,

    pub fn defaults() LspConfig {
        return .{};
    }
};

test {
    std.testing.refAllDecls(@This());
}
