//! Type definitions and utilities for service discovery.
//!
//! Contains DiscoveryBackend, configuration, service instance types,
//! and helper functions used by the ServiceDiscovery implementation.

const std = @import("std");
const time = @import("../../services/shared/time.zig");

var service_id_counter: std.atomic.Value(u64) = .{ .raw = 0 };

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
    // Build IDs from a monotonic process-local counter + random bytes.
    // Counter ensures uniqueness within a process; random suffix reduces cross-process collisions.
    const counter = service_id_counter.fetchAdd(1, .monotonic);
    const salt = @as(u64, @truncate(@intFromPtr(&service_id_counter)));
    var prng = std.Random.DefaultPrng.init(time.getSeed() ^ counter ^ salt);

    var id: [16]u8 = undefined;
    std.mem.writeInt(u64, id[0..8], counter, .little);
    prng.random().bytes(id[8..16]);

    const hex = std.fmt.bytesToHex(id, .lower);
    return std.fmt.allocPrint(allocator, "{s}-{s}", .{ service_name, hex });
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

test "service ID generation is unique and correctly formatted" {
    const allocator = std.testing.allocator;
    const prefix = "svc";
    const expected_len = prefix.len + 1 + 32; // "<prefix>-<16 random bytes as hex>"

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = seen.keyIterator();
        while (it.next()) |key| {
            allocator.free(key.*);
        }
        seen.deinit(allocator);
    }

    const rounds: usize = 256;
    for (0..rounds) |_| {
        const id = try generateServiceId(allocator, prefix);
        try std.testing.expect(std.mem.startsWith(u8, id, "svc-"));
        try std.testing.expectEqual(expected_len, id.len);

        if (seen.contains(id)) {
            allocator.free(id);
            return error.TestUnexpectedResult;
        }

        try seen.put(allocator, id, {});
    }

    try std.testing.expectEqual(rounds, seen.count());
}
