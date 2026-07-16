const std = @import("std");
const types = @import("types.zig");
const widgets = @import("dashboard_widgets.zig");

const appendRow = widgets.appendRow;
const appendMetricRow = widgets.appendMetricRow;

const MAX_PLUGIN_ROWS: usize = 6;

fn boolText(value: bool) []const u8 {
    return if (value) "yes" else "no";
}

pub fn paneColor(kind: types.PaneKind) []const u8 {
    return switch (kind) {
        .system => "\x1b[1;33m",
        .plugins => "\x1b[1;32m",
        .storage => "\x1b[1;35m",
        .scheduler => "\x1b[1;34m",
        .memory => "\x1b[1;31m",
        .agent_output => "\x1b[1;36m",
    };
}

fn appendPluginRows(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, plugin_names: []const []const u8) !void {
    const shown = @min(plugin_names.len, MAX_PLUGIN_ROWS);
    var i: usize = 0;
    while (i < shown) : (i += 1) {
        try appendRow(out, allocator, "plugin", plugin_names[i]);
    }
    if (plugin_names.len > shown) {
        var buf: [48]u8 = undefined;
        const more = try std.fmt.bufPrint(&buf, "+{d} more registered", .{plugin_names.len - shown});
        try appendRow(out, allocator, "plugin", more);
    }
}

pub fn appendPaneBody(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, ds: types.DashboardState, kind: types.PaneKind) !void {
    switch (kind) {
        .system => {
            try appendRow(out, allocator, "GPU backend", ds.gpu_backend);
            try appendRow(out, allocator, "accelerated", boolText(ds.gpu_accelerated));
            try appendRow(out, allocator, "native linked", boolText(ds.gpu_linked));
        },
        .plugins => {
            try appendMetricRow(out, allocator, "Registered", ds.plugin_count);
            try appendPluginRows(out, allocator, ds.plugin_names);
        },
        .storage => {
            try appendRow(out, allocator, "scope", "ephemeral CLI probe");
            try appendMetricRow(out, allocator, "Block chain", ds.wdbx_blocks);
            try appendMetricRow(out, allocator, "Vectors", ds.wdbx_vectors);
            try appendMetricRow(out, allocator, "KV Entries", ds.wdbx_entries);
            try appendMetricRow(out, allocator, "Spatial 3D", ds.wdbx_spatial_records);
        },
        .scheduler => {
            try appendRow(out, allocator, "source", ds.scheduler_source);
            try appendMetricRow(out, allocator, "Running", ds.scheduler_running);
            try appendMetricRow(out, allocator, "Pending", ds.scheduler_pending);
            try appendMetricRow(out, allocator, "Completed", ds.scheduler_completed);
            try appendMetricRow(out, allocator, "Failed", ds.scheduler_failed);
        },
        .memory => {
            try appendRow(out, allocator, "source", ds.memory_source);
            try appendMetricRow(out, allocator, "Peak bytes", ds.memory_peak);
            try appendMetricRow(out, allocator, "Current bytes", ds.memory_current);
            try appendMetricRow(out, allocator, "Leaked bytes", ds.memory_leaked);
        },
        .agent_output => {
            try appendRow(out, allocator, "Agent Output", "see right pane");
        },
    }
}

test {
    std.testing.refAllDecls(@This());
}
