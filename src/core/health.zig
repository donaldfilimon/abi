//! Framework-level health checking.
//!
//! Provides aggregate health status across all enabled features.
//! Used by the CLI `status` command and HTTP health endpoints.

const std = @import("std");

pub const Status = enum {
    healthy,
    degraded,
    unhealthy,
    unknown,

    pub fn httpCode(self: Status) u16 {
        return switch (self) {
            .healthy, .degraded => 200,
            .unhealthy, .unknown => 503,
        };
    }
};

pub const ComponentHealth = struct {
    name: []const u8,
    status: Status,
    message: ?[]const u8 = null,
};

pub const HealthReport = struct {
    status: Status,
    framework_state: []const u8,
    components: []const ComponentHealth,

    pub fn isReady(self: HealthReport) bool {
        return self.status == .healthy or self.status == .degraded;
    }
};

/// Check health of all framework components.
/// The `fw` parameter is `anytype` to avoid circular import with framework.zig.
pub fn check(fw: anytype, allocator: std.mem.Allocator) !HealthReport {
    var components = std.ArrayListUnmanaged(ComponentHealth).empty;
    errdefer components.deinit(allocator);

    const state_name = @tagName(fw.state);

    // Framework state check
    const fw_status: Status = switch (fw.state) {
        .running => .healthy,
        .initializing, .stopping => .degraded,
        .uninitialized, .stopped, .failed => .unhealthy,
    };
    try components.append(allocator, .{
        .name = "framework",
        .status = fw_status,
        .message = state_name,
    });

    // Feature checks — each is non-null if initialized successfully
    const feature_checks = .{
        .{ "gpu", fw.gpu },
        .{ "ai", fw.ai },
        .{ "database", fw.database },
        .{ "network", fw.network },
        .{ "web", fw.web },
        .{ "observability", fw.observability },
        .{ "cloud", fw.cloud },
        .{ "analytics", fw.analytics },
    };

    inline for (feature_checks) |entry| {
        const name = entry[0];
        const ctx = entry[1];
        if (ctx != null) {
            try components.append(allocator, .{ .name = name, .status = .healthy });
        } else {
            try components.append(allocator, .{
                .name = name,
                .status = .unknown,
                .message = "disabled or not initialized",
            });
        }
    }

    // Determine aggregate status
    var aggregate: Status = .healthy;
    for (components.items) |c| {
        switch (c.status) {
            .unhealthy => {
                aggregate = .unhealthy;
                break;
            },
            .degraded => aggregate = .degraded,
            .unknown, .healthy => {},
        }
    }

    return .{
        .status = aggregate,
        .framework_state = state_name,
        .components = try components.toOwnedSlice(allocator),
    };
}

/// Free a HealthReport's allocated memory.
pub fn freeReport(report: *const HealthReport, allocator: std.mem.Allocator) void {
    allocator.free(report.components);
}
