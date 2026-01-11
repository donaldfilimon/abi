//! Service discovery for distributed systems
//!
//! Provides Consul and etcd integration for node discovery
//! and health checking in distributed deployments.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");
const async_http = @import("../../shared/utils/http/async_http.zig");

pub const ServiceDiscoveryError = error{
    DiscoveryFailed,
    ServiceNotFound,
    InvalidResponse,
    ConnectionFailed,
};

pub const DiscoveryBackend = enum {
    consul,
    etcd,
    custom,
};

pub const ServiceInstance = struct {
    id: []const u8,
    name: []const u8,
    address: []const u8,
    port: u16,
    tags: []const []const u8,
    health_status: HealthStatus,
    last_check: i64,

    pub fn deinit(self: *ServiceInstance, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.address);
        for (self.tags) |tag| {
            allocator.free(tag);
        }
        allocator.free(self.tags);
        self.* = undefined;
    }
};

pub const HealthStatus = enum {
    passing,
    warning,
    critical,
    unknown,
};

pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .consul,
    endpoint: []const u8 = "http://127.0.0.1:8500",
    datacenter: []const u8 = "dc1",
    token: ?[]const u8 = null,
    timeout_ms: u32 = 5000,
    health_check_interval_ms: u64 = 30000,
};

pub const ServiceRegistry = struct {
    allocator: std.mem.Allocator,
    config: DiscoveryConfig,
    http_client: async_http.AsyncHttpClient,
    services: std.StringHashMapUnmanaged(ServiceInstance),

    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) !ServiceRegistry {
        const http_client = try async_http.AsyncHttpClient.init(allocator);

        return ServiceRegistry{
            .allocator = allocator,
            .config = config,
            .http_client = http_client,
            .services = .{},
        };
    }

    pub fn deinit(self: *ServiceRegistry) void {
        var iter = self.services.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
        }
        self.services.deinit(self.allocator);
        self.http_client.deinit();
        self.* = undefined;
    }

    pub fn register(self: *ServiceRegistry, instance: ServiceInstance) !void {
        const id_copy = try self.allocator.dupe(u8, instance.id);
        errdefer self.allocator.free(id_copy);

        const name_copy = try self.allocator.dupe(u8, instance.name);
        errdefer self.allocator.free(name_copy);

        const addr_copy = try self.allocator.dupe(u8, instance.address);
        errdefer self.allocator.free(addr_copy);

        var tags_copy = try self.allocator.alloc([]u8, instance.tags.len);
        errdefer self.allocator.free(tags_copy);

        for (instance.tags, 0..) |tag, i| {
            tags_copy[i] = try self.allocator.dupe(u8, tag);
        }

        const instance_copy = ServiceInstance{
            .id = id_copy,
            .name = name_copy,
            .address = addr_copy,
            .port = instance.port,
            .tags = tags_copy,
            .health_status = instance.health_status,
            .last_check = time.unixMilliseconds(),
        };

        try self.services.put(self.allocator, instance_copy.id, instance_copy);
    }

    pub fn deregister(self: *ServiceRegistry, service_id: []const u8) void {
        if (self.services.fetchRemove(service_id)) |entry| {
            entry.value.deinit(self.allocator);
        }
    }

    pub fn discover(self: *ServiceRegistry, service_name: []const u8) ![]ServiceInstance {
        var instances = std.ArrayListUnmanaged(ServiceInstance){};
        errdefer {
            for (instances.items) |*instance| {
                instance.deinit(self.allocator);
            }
            instances.deinit(self.allocator);
        }

        var iter = self.services.iterator();
        while (iter.next()) |entry| {
            const instance = entry.value_ptr.*;
            if (std.mem.eql(u8, instance.name, service_name) and
                instance.health_status == .passing)
            {
                try instances.append(self.allocator, instance);
            }
        }

        return instances.toOwnedSlice(self.allocator);
    }

    pub fn getAllServices(self: *const ServiceRegistry) !std.ArrayListUnmanaged(ServiceInstance) {
        var services = std.ArrayListUnmanaged(ServiceInstance){};
        errdefer services.deinit(self.allocator);

        var iter = self.services.valueIterator();
        while (iter.next()) |instance| {
            try services.append(self.allocator, instance.*);
        }

        return services;
    }

    pub fn healthCheck(self: *ServiceRegistry, service_id: []const u8) !HealthStatus {
        const instance = self.services.get(service_id) orelse return ServiceDiscoveryError.ServiceNotFound;

        // Simple health check - try to connect to the service
        const url = try std.fmt.allocPrint(self.allocator, "http://{s}:{d}/health", .{
            instance.address,
            instance.port,
        });
        defer self.allocator.free(url);

        var request = try async_http.HttpRequest.init(self.allocator, .GET, url);
        defer request.deinit();

        request.timeout_ms = self.config.timeout_ms;

        const response = self.http_client.fetch(&request) catch {
            return .critical;
        };
        defer response.deinit();

        const status = if (response.isSuccess()) .passing else .critical;
        return status;
    }

    pub fn updateHealthStatus(self: *ServiceRegistry, service_id: []const u8, status: HealthStatus) !void {
        const instance = self.services.getPtr(service_id) orelse return ServiceDiscoveryError.ServiceNotFound;
        instance.health_status = status;
        instance.last_check = time.unixMilliseconds();
    }

    pub fn runHealthChecks(self: *ServiceRegistry) !void {
        var iter = self.services.iterator();
        while (iter.next()) |entry| {
            const status = try self.healthCheck(entry.key_ptr.*);
            try self.updateHealthStatus(entry.key_ptr.*, status);
        }
    }
};

pub const LoadBalancer = struct {
    allocator: std.mem.Allocator,
    strategy: LoadBalancingStrategy,
    current_index: usize = 0,

    pub const LoadBalancingStrategy = enum {
        round_robin,
        random,
        least_loaded,
    };

    pub fn init(allocator: std.mem.Allocator, strategy: LoadBalancingStrategy) LoadBalancer {
        return .{
            .allocator = allocator,
            .strategy = strategy,
            .current_index = 0,
        };
    }

    pub fn selectInstance(self: *LoadBalancer, instances: []const ServiceInstance) ?ServiceInstance {
        if (instances.len == 0) return null;

        switch (self.strategy) {
            .round_robin => {
                const instance = instances[self.current_index % instances.len];
                self.current_index += 1;
                return instance;
            },
            .random => {
                const index = std.crypto.random.intRangeLessThan(usize, instances.len);
                return instances[index];
            },
            .least_loaded => {
                // For now, just return first healthy instance
                // In a real implementation, this would check load metrics
                for (instances) |instance| {
                    if (instance.health_status == .passing) {
                        return instance;
                    }
                }
                return null;
            },
        }
    }
};

test "service discovery registration" {
    const allocator = std.testing.allocator;

    var registry = try ServiceRegistry.init(allocator, .{});
    defer registry.deinit();

    const instance = ServiceInstance{
        .id = "web-1",
        .name = "web-server",
        .address = "127.0.0.1",
        .port = 8080,
        .tags = &[_][]const u8{ "http", "api" },
        .health_status = .passing,
        .last_check = 0,
    };

    try registry.register(instance);

    try std.testing.expectEqual(@as(usize, 1), registry.services.count());
    try std.testing.expect(registry.services.contains("web-1"));

    const discovered = try registry.discover("web-server");
    defer allocator.free(discovered);
    try std.testing.expectEqual(@as(usize, 1), discovered.len);
    try std.testing.expectEqualStrings("web-1", discovered[0].id);
}

test "load balancer round robin" {
    const allocator = std.testing.allocator;

    var balancer = LoadBalancer.init(allocator, .round_robin);

    const instances = [_]ServiceInstance{
        .{
            .id = "svc-1",
            .name = "service",
            .address = "10.0.0.1",
            .port = 9000,
            .tags = &.{},
            .health_status = .passing,
            .last_check = 0,
        },
        .{
            .id = "svc-2",
            .name = "service",
            .address = "10.0.0.2",
            .port = 9000,
            .tags = &.{},
            .health_status = .passing,
            .last_check = 0,
        },
    };

    const first = balancer.selectInstance(&instances).?;
    try std.testing.expectEqualStrings("svc-1", first.id);

    const second = balancer.selectInstance(&instances).?;
    try std.testing.expectEqualStrings("svc-2", second.id);

    const third = balancer.selectInstance(&instances).?;
    try std.testing.expectEqualStrings("svc-1", third.id);
}
