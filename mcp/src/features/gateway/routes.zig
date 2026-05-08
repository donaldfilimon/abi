const std = @import("std");
const gateway_types = @import("types.zig");
const rate_limit_mod = @import("rate_limit.zig");
const routing = @import("routing.zig");
const gateway_state = @import("state.zig");

const RouteTree = routing.RouteTree;
const RateLimiter = rate_limit_mod.RateLimiter;

var route_view_buf: [256]gateway_types.Route = undefined;

pub fn addRoute(route: gateway_types.Route) gateway_types.GatewayError!void {
    const s = gateway_state.gw_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.routes.items.len >= s.config.max_routes) return error.TooManyRoutes;
    if (route.path.len == 0) return error.InvalidRoute;

    var limiter: ?*RateLimiter = null;
    if (route.rate_limit) |rl_config| {
        s.route_limiters.ensureUnusedCapacity(s.allocator, 1) catch
            return error.OutOfMemory;
        limiter = s.allocator.create(RateLimiter) catch return error.OutOfMemory;
        limiter.?.* = RateLimiter.init(rl_config, gateway_state.nowNs());
    }
    errdefer if (limiter) |l| s.allocator.destroy(l);

    const route_idx: u32 = @intCast(s.routes.items.len);

    const path_owned = s.allocator.dupe(u8, route.path) catch return error.OutOfMemory;
    const upstream_owned = s.allocator.dupe(u8, route.upstream) catch {
        s.allocator.free(path_owned);
        return error.OutOfMemory;
    };

    s.routes.append(s.allocator, .{
        .route = .{
            .path = path_owned,
            .method = route.method,
            .upstream = upstream_owned,
            .timeout_ms = route.timeout_ms,
            .rate_limit = route.rate_limit,
            .middlewares = route.middlewares,
        },
        .path_owned = path_owned,
        .upstream_owned = upstream_owned,
    }) catch {
        s.allocator.free(path_owned);
        s.allocator.free(upstream_owned);
        return error.OutOfMemory;
    };

    s.insertRadixRoute(path_owned, route_idx) catch {
        _ = s.routes.pop();
        s.allocator.free(path_owned);
        s.allocator.free(upstream_owned);
        return error.OutOfMemory;
    };

    if (limiter) |l| {
        s.route_limiters.putAssumeCapacity(path_owned, l);
    }
}

pub fn removeRoute(path: []const u8) gateway_types.GatewayError!bool {
    const s = gateway_state.gw_state orelse return error.FeatureDisabled;
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    for (s.routes.items, 0..) |entry, i| {
        if (std.mem.eql(u8, entry.path_owned, path)) {
            if (s.route_limiters.fetchRemove(entry.path_owned)) |kv| {
                s.allocator.destroy(kv.value);
            }

            s.allocator.free(entry.path_owned);
            s.allocator.free(entry.upstream_owned);
            _ = s.routes.orderedRemove(i);

            s.radix_root.deinitRecursive(s.allocator);
            s.radix_root.* = .{};
            for (s.routes.items, 0..) |remaining, new_idx| {
                s.insertRadixRoute(remaining.path_owned, @intCast(new_idx)) catch |err| {
                    std.log.err("gateway: radix rebuild failed after removeRoute: {t}", .{err});
                    return error.OutOfMemory;
                };
            }
            return true;
        }
    }
    return false;
}

pub fn getRoutes() []const gateway_types.Route {
    const s = gateway_state.gw_state orelse return &.{};
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const count = @min(s.routes.items.len, route_view_buf.len);
    for (s.routes.items[0..count], 0..) |entry, i| {
        route_view_buf[i] = entry.route;
    }
    return route_view_buf[0..count];
}

pub fn getRouteCount() usize {
    const s = gateway_state.gw_state orelse return 0;
    return s.routes.items.len;
}

pub fn matchRoute(
    path: []const u8,
    method: gateway_types.HttpMethod,
) gateway_types.GatewayError!?gateway_types.MatchResult {
    const s = gateway_state.gw_state orelse return error.FeatureDisabled;
    _ = s.stat_total_requests.fetchAdd(1, .monotonic);

    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    var tree_result = RouteTree.MatchResult{};
    if (RouteTree.match(s.radix_root, path, &tree_result)) {
        if (tree_result.terminal_idx) |idx| {
            if (idx < s.routes.items.len) {
                const entry = s.routes.items[idx];
                if (entry.route.method == method) {
                    var result = gateway_types.MatchResult{ .route = entry.route };
                    result.matched_route_idx = idx;
                    result.param_count = tree_result.param_count;
                    result.params = tree_result.params;
                    return result;
                }
            }
        }
    }

    return null;
}

test {
    std.testing.refAllDecls(@This());
}
