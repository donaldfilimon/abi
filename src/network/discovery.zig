//! Service Discovery module supporting Consul and etcd backends.
//!
//! Provides automatic service registration, health checking, and discovery
//! for distributed cluster coordination.
//!
//! This module is split for maintainability:
//! - discovery_types.zig: Type definitions, config, errors, and utilities

const std = @import("std");
const time = @import("../shared/utils_combined.zig");
const registry = @import("registry.zig");

// Import types from submodule
pub const discovery_types = @import("discovery_types.zig");

// Re-export types
pub const DiscoveryBackend = discovery_types.DiscoveryBackend;
pub const DiscoveryConfig = discovery_types.DiscoveryConfig;
pub const ServiceInstance = discovery_types.ServiceInstance;
pub const ServiceStatus = discovery_types.ServiceStatus;
pub const DiscoveryError = discovery_types.DiscoveryError;
pub const AddressPort = discovery_types.AddressPort;

// Re-export utility functions
pub const generateServiceId = discovery_types.generateServiceId;
pub const base64Encode = discovery_types.base64Encode;
pub const base64Decode = discovery_types.base64Decode;
pub const parseAddressPort = discovery_types.parseAddressPort;

pub const ServiceDiscovery = struct {
    allocator: std.mem.Allocator,
    config: DiscoveryConfig,
    registered: bool,
    last_heartbeat_ms: i64,
    cached_services: std.ArrayListUnmanaged(ServiceInstance),
    cache_valid_until_ms: i64,
    node_registry: ?*registry.NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) !ServiceDiscovery {
        var cfg = config;
        if (cfg.service_id.len == 0) {
            cfg.service_id = try generateServiceId(allocator, cfg.service_name);
        }

        return .{
            .allocator = allocator,
            .config = cfg,
            .registered = false,
            .last_heartbeat_ms = 0,
            .cached_services = std.ArrayListUnmanaged(ServiceInstance){},
            .cache_valid_until_ms = 0,
            .node_registry = null,
        };
    }

    pub fn deinit(self: *ServiceDiscovery) void {
        if (self.registered) {
            self.deregister() catch {};
        }
        self.clearCache();
        self.cached_services.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register this service instance with the discovery backend.
    pub fn register(self: *ServiceDiscovery) !void {
        if (self.registered) return;

        switch (self.config.backend) {
            .consul => try self.registerConsul(),
            .etcd => try self.registerEtcd(),
            .static => {},
            .dns => {},
        }

        self.registered = true;
        self.last_heartbeat_ms = time.nowMilliseconds();
    }

    /// Deregister this service instance from the discovery backend.
    pub fn deregister(self: *ServiceDiscovery) !void {
        if (!self.registered) return;

        switch (self.config.backend) {
            .consul => try self.deregisterConsul(),
            .etcd => try self.deregisterEtcd(),
            .static => {},
            .dns => {},
        }

        self.registered = false;
    }

    /// Send a heartbeat to maintain registration.
    pub fn heartbeat(self: *ServiceDiscovery) !void {
        if (!self.registered) return;

        const now = time.nowMilliseconds();
        const interval = @as(i64, @intCast(self.config.health_check_interval_ms));

        if (now - self.last_heartbeat_ms < interval) return;

        switch (self.config.backend) {
            .consul => try self.heartbeatConsul(),
            .etcd => try self.heartbeatEtcd(),
            .static => {},
            .dns => {},
        }

        self.last_heartbeat_ms = now;
    }

    /// Discover all instances of a service.
    pub fn discover(self: *ServiceDiscovery, service_name: []const u8) ![]const ServiceInstance {
        const now = time.nowMilliseconds();

        // Use cache if still valid
        if (now < self.cache_valid_until_ms) {
            return self.cached_services.items;
        }

        self.clearCache();

        switch (self.config.backend) {
            .consul => try self.discoverConsul(service_name),
            .etcd => try self.discoverEtcd(service_name),
            .static => try self.discoverStatic(),
            .dns => try self.discoverDns(service_name),
        }

        self.cache_valid_until_ms = now + @as(i64, @intCast(self.config.health_check_interval_ms));
        return self.cached_services.items;
    }

    /// Sync discovered services with a NodeRegistry.
    pub fn syncWithRegistry(self: *ServiceDiscovery, node_registry: *registry.NodeRegistry) !void {
        self.node_registry = node_registry;

        const services = try self.discover(self.config.service_name);

        for (services) |service| {
            const address = try std.fmt.allocPrint(
                self.allocator,
                "{s}:{d}",
                .{ service.address, service.port },
            );
            defer self.allocator.free(address);

            try node_registry.register(service.id, address);

            const status: registry.NodeStatus = switch (service.status) {
                .passing => .healthy,
                .warning => .degraded,
                .critical, .unknown => .offline,
            };
            _ = node_registry.setStatus(service.id, status);
        }
    }

    // Consul implementation

    fn registerConsul(self: *ServiceDiscovery) !void {
        const body = try self.buildConsulRegistration();
        defer self.allocator.free(body);

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v1/agent/service/register",
            .{self.config.endpoint},
        );
        defer self.allocator.free(url);

        try self.httpPut(url, body);
    }

    fn deregisterConsul(self: *ServiceDiscovery) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v1/agent/service/deregister/{s}",
            .{ self.config.endpoint, self.config.service_id },
        );
        defer self.allocator.free(url);

        try self.httpPut(url, "");
    }

    fn heartbeatConsul(self: *ServiceDiscovery) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v1/agent/check/pass/service:{s}",
            .{ self.config.endpoint, self.config.service_id },
        );
        defer self.allocator.free(url);

        try self.httpPut(url, "");
    }

    fn discoverConsul(self: *ServiceDiscovery, service_name: []const u8) !void {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v1/health/service/{s}?passing=true&dc={s}",
            .{ self.config.endpoint, service_name, self.config.datacenter },
        );
        defer self.allocator.free(url);

        const response = try self.httpGet(url);
        defer self.allocator.free(response);

        try self.parseConsulResponse(response);
    }

    pub fn buildConsulRegistration(self: *ServiceDiscovery) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8){};
        errdefer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{\"ID\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_id);
        try buffer.appendSlice(self.allocator, "\",\"Name\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_name);
        try buffer.appendSlice(self.allocator, "\",\"Address\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_address);
        try buffer.appendSlice(self.allocator, "\",\"Port\":");

        var port_buf: [8]u8 = undefined;
        const port_str = std.fmt.bufPrint(&port_buf, "{d}", .{self.config.service_port}) catch unreachable;
        try buffer.appendSlice(self.allocator, port_str);

        try buffer.appendSlice(self.allocator, ",\"Tags\":[");
        for (self.config.tags, 0..) |tag, i| {
            if (i > 0) try buffer.appendSlice(self.allocator, ",");
            try buffer.appendSlice(self.allocator, "\"");
            try buffer.appendSlice(self.allocator, tag);
            try buffer.appendSlice(self.allocator, "\"");
        }
        try buffer.appendSlice(self.allocator, "],\"Check\":{\"TTL\":\"");

        var ttl_buf: [16]u8 = undefined;
        const ttl_str = std.fmt.bufPrint(&ttl_buf, "{d}s", .{self.config.ttl_seconds}) catch unreachable;
        try buffer.appendSlice(self.allocator, ttl_str);
        try buffer.appendSlice(self.allocator, "\"}}");

        return buffer.toOwnedSlice(self.allocator);
    }

    fn parseConsulResponse(self: *ServiceDiscovery, response: []const u8) !void {
        // Parse JSON response from Consul health endpoint
        // Response format: [{"Node":{...},"Service":{...},"Checks":[...]}]
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response,
            .{},
        ) catch return DiscoveryError.InvalidResponse;
        defer parsed.deinit();

        if (parsed.value != .array) return DiscoveryError.InvalidResponse;

        for (parsed.value.array.items) |entry| {
            if (entry != .object) continue;

            const service_obj = entry.object.get("Service") orelse continue;
            if (service_obj != .object) continue;

            const service = service_obj.object;

            const id = blk: {
                const val = service.get("ID") orelse break :blk null;
                if (val != .string) break :blk null;
                break :blk val.string;
            };
            const name = blk: {
                const val = service.get("Service") orelse break :blk null;
                if (val != .string) break :blk null;
                break :blk val.string;
            };
            const address = blk: {
                const val = service.get("Address") orelse break :blk null;
                if (val != .string) break :blk null;
                break :blk val.string;
            };
            const port = blk: {
                const val = service.get("Port") orelse break :blk null;
                if (val != .integer) break :blk null;
                break :blk @as(u16, @intCast(val.integer));
            };

            if (id == null or name == null or address == null or port == null) continue;

            const status = self.parseConsulStatus(entry);

            const instance = ServiceInstance{
                .id = try self.allocator.dupe(u8, id.?),
                .name = try self.allocator.dupe(u8, name.?),
                .address = try self.allocator.dupe(u8, address.?),
                .port = port.?,
                .tags = &.{},
                .status = status,
                .metadata = std.StringArrayHashMapUnmanaged([]const u8){},
            };

            try self.cached_services.append(self.allocator, instance);
        }
    }

    fn parseConsulStatus(_: *ServiceDiscovery, entry: std.json.Value) ServiceStatus {
        const checks_val = entry.object.get("Checks") orelse return .unknown;
        if (checks_val != .array) return .unknown;

        var worst: ServiceStatus = .passing;
        for (checks_val.array.items) |check| {
            if (check != .object) continue;
            const status_val = check.object.get("Status") orelse continue;
            if (status_val != .string) continue;

            const status_str = status_val.string;
            if (std.mem.eql(u8, status_str, "critical")) return .critical;
            if (std.mem.eql(u8, status_str, "warning") and worst == .passing) {
                worst = .warning;
            }
        }

        return worst;
    }

    // etcd implementation

    fn registerEtcd(self: *ServiceDiscovery) !void {
        const key = try self.buildEtcdKey();
        defer self.allocator.free(key);

        const value = try self.buildEtcdValue();
        defer self.allocator.free(value);

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v3/kv/put",
            .{self.config.endpoint},
        );
        defer self.allocator.free(url);

        const body = try self.buildEtcdPutBody(key, value);
        defer self.allocator.free(body);

        try self.httpPost(url, body);
    }

    fn deregisterEtcd(self: *ServiceDiscovery) !void {
        const key = try self.buildEtcdKey();
        defer self.allocator.free(key);

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v3/kv/deleterange",
            .{self.config.endpoint},
        );
        defer self.allocator.free(url);

        const body = try self.buildEtcdDeleteBody(key);
        defer self.allocator.free(body);

        try self.httpPost(url, body);
    }

    fn heartbeatEtcd(self: *ServiceDiscovery) !void {
        // Re-register with updated timestamp
        try self.registerEtcd();
    }

    fn discoverEtcd(self: *ServiceDiscovery, service_name: []const u8) !void {
        const prefix = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}/",
            .{ self.config.namespace, service_name },
        );
        defer self.allocator.free(prefix);

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v3/kv/range",
            .{self.config.endpoint},
        );
        defer self.allocator.free(url);

        const body = try self.buildEtcdRangeBody(prefix);
        defer self.allocator.free(body);

        const response = try self.httpPost(url, body);
        defer self.allocator.free(response);

        try self.parseEtcdResponse(response);
    }

    pub fn buildEtcdKey(self: *ServiceDiscovery) ![]const u8 {
        return std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}/{s}",
            .{ self.config.namespace, self.config.service_name, self.config.service_id },
        );
    }

    fn buildEtcdValue(self: *ServiceDiscovery) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8){};
        errdefer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{\"id\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_id);
        try buffer.appendSlice(self.allocator, "\",\"address\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_address);
        try buffer.appendSlice(self.allocator, "\",\"port\":");

        var port_buf: [8]u8 = undefined;
        const port_str = std.fmt.bufPrint(&port_buf, "{d}", .{self.config.service_port}) catch unreachable;
        try buffer.appendSlice(self.allocator, port_str);
        try buffer.appendSlice(self.allocator, ",\"timestamp\":");

        var ts_buf: [24]u8 = undefined;
        const ts_str = std.fmt.bufPrint(&ts_buf, "{d}", .{time.nowMilliseconds()}) catch unreachable;
        try buffer.appendSlice(self.allocator, ts_str);
        try buffer.appendSlice(self.allocator, "}");

        return buffer.toOwnedSlice(self.allocator);
    }

    fn buildEtcdPutBody(self: *ServiceDiscovery, key: []const u8, value: []const u8) ![]const u8 {
        const key_b64 = try base64Encode(self.allocator, key);
        defer self.allocator.free(key_b64);

        const value_b64 = try base64Encode(self.allocator, value);
        defer self.allocator.free(value_b64);

        return std.fmt.allocPrint(
            self.allocator,
            "{{\"key\":\"{s}\",\"value\":\"{s}\"}}",
            .{ key_b64, value_b64 },
        );
    }

    fn buildEtcdDeleteBody(self: *ServiceDiscovery, key: []const u8) ![]const u8 {
        const key_b64 = try base64Encode(self.allocator, key);
        defer self.allocator.free(key_b64);

        return std.fmt.allocPrint(
            self.allocator,
            "{{\"key\":\"{s}\"}}",
            .{key_b64},
        );
    }

    fn buildEtcdRangeBody(self: *ServiceDiscovery, prefix: []const u8) ![]const u8 {
        const key_b64 = try base64Encode(self.allocator, prefix);
        defer self.allocator.free(key_b64);

        // Range end is prefix with last byte incremented
        var range_end = try self.allocator.dupe(u8, prefix);
        defer self.allocator.free(range_end);
        if (range_end.len > 0) {
            range_end[range_end.len - 1] += 1;
        }

        const end_b64 = try base64Encode(self.allocator, range_end);
        defer self.allocator.free(end_b64);

        return std.fmt.allocPrint(
            self.allocator,
            "{{\"key\":\"{s}\",\"range_end\":\"{s}\"}}",
            .{ key_b64, end_b64 },
        );
    }

    fn parseEtcdResponse(self: *ServiceDiscovery, response: []const u8) !void {
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response,
            .{},
        ) catch return DiscoveryError.InvalidResponse;
        defer parsed.deinit();

        if (parsed.value != .object) return DiscoveryError.InvalidResponse;

        const kvs = parsed.value.object.get("kvs") orelse return;
        if (kvs != .array) return;

        for (kvs.array.items) |kv| {
            if (kv != .object) continue;

            const value_b64 = blk: {
                const val = kv.object.get("value") orelse break :blk null;
                if (val != .string) break :blk null;
                break :blk val.string;
            };

            if (value_b64 == null) continue;

            const value = base64Decode(self.allocator, value_b64.?) catch continue;
            defer self.allocator.free(value);

            const instance = self.parseEtcdServiceValue(value) catch continue;
            try self.cached_services.append(self.allocator, instance);
        }
    }

    fn parseEtcdServiceValue(self: *ServiceDiscovery, value: []const u8) !ServiceInstance {
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            value,
            .{},
        ) catch return DiscoveryError.InvalidResponse;
        defer parsed.deinit();

        if (parsed.value != .object) return DiscoveryError.InvalidResponse;

        const obj = parsed.value.object;

        const id = blk: {
            const val = obj.get("id") orelse return DiscoveryError.InvalidResponse;
            if (val != .string) return DiscoveryError.InvalidResponse;
            break :blk val.string;
        };

        const address = blk: {
            const val = obj.get("address") orelse return DiscoveryError.InvalidResponse;
            if (val != .string) return DiscoveryError.InvalidResponse;
            break :blk val.string;
        };

        const port = blk: {
            const val = obj.get("port") orelse return DiscoveryError.InvalidResponse;
            if (val != .integer) return DiscoveryError.InvalidResponse;
            break :blk @as(u16, @intCast(val.integer));
        };

        return ServiceInstance{
            .id = try self.allocator.dupe(u8, id),
            .name = try self.allocator.dupe(u8, self.config.service_name),
            .address = try self.allocator.dupe(u8, address),
            .port = port,
            .tags = &.{},
            .status = .passing,
            .metadata = std.StringArrayHashMapUnmanaged([]const u8){},
        };
    }

    // Static discovery (uses pre-configured addresses)

    fn discoverStatic(self: *ServiceDiscovery) !void {
        if (self.node_registry) |reg| {
            for (reg.list()) |node| {
                const addr_port = parseAddressPort(node.address);
                const instance = ServiceInstance{
                    .id = try self.allocator.dupe(u8, node.id),
                    .name = try self.allocator.dupe(u8, self.config.service_name),
                    .address = try self.allocator.dupe(u8, addr_port.address),
                    .port = addr_port.port,
                    .tags = &.{},
                    .status = switch (node.status) {
                        .healthy => .passing,
                        .degraded => .warning,
                        .offline => .critical,
                    },
                    .metadata = std.StringArrayHashMapUnmanaged([]const u8){},
                };
                try self.cached_services.append(self.allocator, instance);
            }
        }
    }

    // DNS-based discovery

    fn discoverDns(self: *ServiceDiscovery, _: []const u8) !void {
        // DNS-based service discovery would query SRV records
        // For now, fall back to static discovery
        try self.discoverStatic();
    }

    // Helper functions

    fn clearCache(self: *ServiceDiscovery) void {
        for (self.cached_services.items) |*service| {
            service.deinit(self.allocator);
        }
        self.cached_services.clearRetainingCapacity();
    }

    fn httpGet(self: *ServiceDiscovery, url: []const u8) ![]const u8 {
        _ = url;
        // Simulated HTTP GET - in production, use actual HTTP client
        return try self.allocator.dupe(u8, "[]");
    }

    fn httpPut(self: *ServiceDiscovery, url: []const u8, body: []const u8) !void {
        _ = self;
        _ = url;
        _ = body;
        // Simulated HTTP PUT - in production, use actual HTTP client
    }

    fn httpPost(self: *ServiceDiscovery, url: []const u8, body: []const u8) ![]const u8 {
        _ = url;
        _ = body;
        // Simulated HTTP POST - in production, use actual HTTP client
        return try self.allocator.dupe(u8, "{}");
    }
};

test "service discovery initialization" {
    const allocator = std.testing.allocator;
    var discovery = try ServiceDiscovery.init(allocator, .{
        .service_name = "test-service",
        .service_address = "127.0.0.1",
        .service_port = 8080,
    });
    defer discovery.deinit();

    try std.testing.expect(!discovery.registered);
    try std.testing.expect(std.mem.startsWith(u8, discovery.config.service_id, "test-service-"));
}

test "consul registration body" {
    const allocator = std.testing.allocator;
    var discovery = try ServiceDiscovery.init(allocator, .{
        .backend = .consul,
        .service_name = "my-service",
        .service_id = "my-service-1",
        .service_address = "192.168.1.100",
        .service_port = 9000,
        .ttl_seconds = 30,
    });
    defer discovery.deinit();

    const body = try discovery.buildConsulRegistration();
    defer allocator.free(body);

    try std.testing.expect(std.mem.indexOf(u8, body, "\"ID\":\"my-service-1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"Name\":\"my-service\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"Port\":9000") != null);
}

test "etcd key building" {
    const allocator = std.testing.allocator;
    var discovery = try ServiceDiscovery.init(allocator, .{
        .backend = .etcd,
        .namespace = "/services",
        .service_name = "api",
        .service_id = "api-001",
    });
    defer discovery.deinit();

    const key = try discovery.buildEtcdKey();
    defer allocator.free(key);

    try std.testing.expectEqualStrings("/services/api/api-001", key);
}

test "address port parsing" {
    const result1 = parseAddressPort("192.168.1.1:8080");
    try std.testing.expectEqualStrings("192.168.1.1", result1.address);
    try std.testing.expectEqual(@as(u16, 8080), result1.port);

    const result2 = parseAddressPort("localhost");
    try std.testing.expectEqualStrings("localhost", result2.address);
    try std.testing.expectEqual(@as(u16, 9000), result2.port);
}

test "static discovery with registry" {
    const allocator = std.testing.allocator;

    var reg = registry.NodeRegistry.init(allocator);
    defer reg.deinit();

    try reg.register("node-1", "192.168.1.10:9000");
    try reg.register("node-2", "192.168.1.11:9001");

    var discovery = try ServiceDiscovery.init(allocator, .{
        .backend = .static,
        .service_name = "test-service",
    });
    defer discovery.deinit();

    discovery.node_registry = &reg;

    const services = try discovery.discover("test-service");
    try std.testing.expectEqual(@as(usize, 2), services.len);
}
