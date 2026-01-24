//! Type definitions and utilities for service discovery.
//!
//! Contains DiscoveryBackend, configuration, service instance types,
//! and helper functions used by the ServiceDiscovery implementation.

const std = @import("std");

pub const DiscoveryBackend = enum {
    consul,
    etcd,
    static,
    dns,
};

pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .static,
    /// Consul or etcd endpoint (e.g., "http://127.0.0.1:8500")
    endpoint: []const u8 = "http://127.0.0.1:8500",
    /// Service name for registration
    service_name: []const u8 = "abi-node",
    /// Service ID (unique per instance)
    service_id: []const u8 = "",
    /// Service address for registration
    service_address: []const u8 = "127.0.0.1",
    /// Service port
    service_port: u16 = 9000,
    /// Health check interval in milliseconds
    health_check_interval_ms: u64 = 10_000,
    /// TTL for service registration (seconds)
    ttl_seconds: u64 = 30,
    /// Tags for service registration
    tags: []const []const u8 = &.{},
    /// Enable TLS for backend connection
    enable_tls: bool = false,
    /// Datacenter for Consul
    datacenter: []const u8 = "dc1",
    /// Namespace for etcd keys
    namespace: []const u8 = "/abi/services",
    /// Token for authentication
    token: []const u8 = "",
};

pub const ServiceInstance = struct {
    id: []const u8,
    name: []const u8,
    address: []const u8,
    port: u16,
    tags: []const []const u8,
    status: ServiceStatus,
    metadata: std.StringArrayHashMapUnmanaged([]const u8),

    pub fn deinit(self: *ServiceInstance, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.address);
        for (self.tags) |tag| {
            allocator.free(tag);
        }
        allocator.free(self.tags);
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit(allocator);
        self.* = undefined;
    }
};

pub const ServiceStatus = enum {
    passing,
    warning,
    critical,
    unknown,
};

pub const DiscoveryError = error{
    BackendUnavailable,
    RegistrationFailed,
    DeregistrationFailed,
    DiscoveryFailed,
    InvalidResponse,
    AuthenticationFailed,
    ConnectionTimeout,
    InvalidConfiguration,
};

pub const AddressPort = struct {
    address: []const u8,
    port: u16,
};

/// Generate a unique service ID from service name and random bytes.
pub fn generateServiceId(allocator: std.mem.Allocator, service_name: []const u8) ![]const u8 {
    // Use timer-based seed with hash for uniqueness (Zig 0.16 compatible)
    var timer = std.time.Timer.start() catch return error.TimerUnsupported;
    const seed = timer.read();
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var id: [8]u8 = undefined;
    random.bytes(&id);

    var hex: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x:0>16}", .{std.mem.readInt(u64, &id, .big)}) catch unreachable;

    return std.fmt.allocPrint(allocator, "{s}-{s}", .{ service_name, hex[0..8] });
}

/// Base64 encode data using standard encoding.
pub fn base64Encode(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    const encoder = std.base64.standard.Encoder;
    const len = encoder.calcSize(data.len);
    const encoded = try allocator.alloc(u8, len);
    _ = encoder.encode(encoded, data);
    return encoded;
}

/// Base64 decode data using standard encoding.
pub fn base64Decode(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    const decoder = std.base64.standard.Decoder;
    const len = decoder.calcSizeForSlice(data) catch return error.InvalidBase64;
    const decoded = try allocator.alloc(u8, len);
    decoder.decode(decoded, data) catch {
        allocator.free(decoded);
        return error.InvalidBase64;
    };
    return decoded;
}

/// Parse an address:port string into separate components.
pub fn parseAddressPort(full_address: []const u8) AddressPort {
    if (std.mem.lastIndexOf(u8, full_address, ":")) |colon| {
        const port_str = full_address[colon + 1 ..];
        const port = std.fmt.parseInt(u16, port_str, 10) catch 9000;
        return .{
            .address = full_address[0..colon],
            .port = port,
        };
    }
    return .{
        .address = full_address,
        .port = 9000,
    };
}

test "address port parsing" {
    const result1 = parseAddressPort("192.168.1.1:8080");
    try std.testing.expectEqualStrings("192.168.1.1", result1.address);
    try std.testing.expectEqual(@as(u16, 8080), result1.port);

    const result2 = parseAddressPort("localhost");
    try std.testing.expectEqualStrings("localhost", result2.address);
    try std.testing.expectEqual(@as(u16, 9000), result2.port);
}
