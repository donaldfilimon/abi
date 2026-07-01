//! MCP `plugin_list` / `plugin_run` support: load the bundled plugins through
//! the plugin manager and format their metadata. Extracted from handlers.zig —
//! each call owns a local PluginManager, so there is no shared state.

const std = @import("std");
const abi = @import("abi");

/// Load the bundled plugins into `pm`. Delegates to the shared
/// `abi.plugins.loadBundled` so the MCP surface stays symmetric with the CLI:
/// the same `bundled_plugin_paths` set, loaded with the same tolerant
/// (idempotent, skip-on-bad-manifest) behavior.
pub fn loadBundledPlugins(pm: *abi.plugins.PluginManager) void {
    abi.plugins.loadBundled(pm);
}

/// Return a one-line summary of the bundled plugins' metadata.
pub fn runPluginList(allocator: std.mem.Allocator) ![]u8 {
    var pm = abi.plugins.PluginManager.init(allocator);
    defer pm.deinit();
    loadBundledPlugins(&pm);

    const list = try pm.listPlugins();
    defer allocator.free(list);

    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "plugins count={d}", .{list.len});
    for (list) |plugin| {
        try out.print(
            allocator,
            " name={s} version={s} target={s} entry={s} description={s};",
            .{ plugin.name, plugin.version, plugin.target_feature, plugin.entry_point, plugin.description },
        );
    }
    return try out.toOwnedSlice(allocator);
}

test {
    std.testing.refAllDecls(@This());
}
