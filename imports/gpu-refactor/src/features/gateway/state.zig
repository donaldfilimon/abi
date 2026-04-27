const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
const time_mod = @import("../../foundation/mod.zig").time;
const gateway_types = @import("types.zig");
const routing = @import("routing.zig");
const rate_limit_mod = @import("rate_limit.zig");
const circuit_breaker_mod = @import("circuit_breaker.zig");
const gateway_stats = @import("stats.zig");

const RouteTree = routing.RouteTree;
const RadixNode = RouteTree.Node;
const RateLimiter = rate_limit_mod.RateLimiter;
const CircuitBreaker = circuit_breaker_mod.CircuitBreaker;
const LatencyHistogram = gateway_stats.LatencyHistogram;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: gateway_types.GatewayConfig,

    pub fn init(
        allocator: std.mem.Allocator,
        config: gateway_types.GatewayConfig,
    ) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

pub const RouteEntry = struct {
    route: gateway_types.Route,
    path_owned: []u8,
    upstream_owned: []u8,
};

pub const GatewayState = struct {
    allocator: std.mem.Allocator,
    config: gateway_types.GatewayConfig,
    routes: std.ArrayListUnmanaged(RouteEntry),
    radix_root: *RadixNode,
    route_limiters: std.StringHashMapUnmanaged(*RateLimiter),
    circuit_breakers: std.StringHashMapUnmanaged(*CircuitBreaker),
    latency: LatencyHistogram,
    rw_lock: sync.RwLock,

    stat_total_requests: std.atomic.Value(u64),
    stat_rate_limited: std.atomic.Value(u64),
    stat_cb_trips: std.atomic.Value(u64),
    stat_upstream_errors: std.atomic.Value(u64),

    fn create(
        allocator: std.mem.Allocator,
        config: gateway_types.GatewayConfig,
    ) !*GatewayState {
        const root = try allocator.create(RadixNode);
        root.* = .{};

        const s = try allocator.create(GatewayState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .routes = .empty,
            .radix_root = root,
            .route_limiters = .empty,
            .circuit_breakers = .empty,
            .latency = .{},
            .rw_lock = sync.RwLock.init(),
            .stat_total_requests = std.atomic.Value(u64).init(0),
            .stat_rate_limited = std.atomic.Value(u64).init(0),
            .stat_cb_trips = std.atomic.Value(u64).init(0),
            .stat_upstream_errors = std.atomic.Value(u64).init(0),
        };
        return s;
    }

    fn destroy(self: *GatewayState) void {
        const allocator = self.allocator;

        for (self.routes.items) |entry| {
            allocator.free(entry.path_owned);
            allocator.free(entry.upstream_owned);
        }
        self.routes.deinit(allocator);

        self.radix_root.deinitRecursive(allocator);
        allocator.destroy(self.radix_root);

        var rl_iter = self.route_limiters.iterator();
        while (rl_iter.next()) |entry| {
            allocator.destroy(entry.value_ptr.*);
        }
        self.route_limiters.deinit(allocator);

        var cb_iter = self.circuit_breakers.iterator();
        while (cb_iter.next()) |entry| {
            allocator.destroy(entry.value_ptr.*);
        }
        self.circuit_breakers.deinit(allocator);

        allocator.destroy(self);
    }

    pub fn insertRadixRoute(
        self: *GatewayState,
        path: []const u8,
        route_idx: u32,
    ) !void {
        try RouteTree.insert(self.radix_root, self.allocator, path, route_idx);
    }
};

pub var gw_state: ?*GatewayState = null;

pub fn nowNs() u128 {
    return (time_mod.Instant.now() catch return 0).nanos;
}

pub fn init(
    allocator: std.mem.Allocator,
    config: gateway_types.GatewayConfig,
) gateway_types.GatewayError!void {
    if (gw_state != null) return;
    gw_state = GatewayState.create(allocator, config) catch return error.OutOfMemory;
}

pub fn deinit() void {
    if (gw_state) |s| {
        s.destroy();
        gw_state = null;
    }
}

pub fn isInitialized() bool {
    return gw_state != null;
}

test {
    std.testing.refAllDecls(@This());
}
