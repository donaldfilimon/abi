//! Service Discovery module supporting Consul and etcd backends.
//!
//! Provides automatic service registration, health checking, and discovery
//! for distributed cluster coordination.
//!
//! This module is split for maintainability:
//! - discovery_types.zig: Type definitions, config, errors, and utilities

const std = @import("std");
const time = @import("../../services/shared/utils.zig");
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
    owns_service_id: bool, // Track if we allocated the service_id

    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) !ServiceDiscovery {
        var cfg = config;
        var owns_id = false;
        if (cfg.service_id.len == 0) {
            cfg.service_id = try generateServiceId(allocator, cfg.service_name);
            owns_id = true;
        }

        return .{
            .allocator = allocator,
            .config = cfg,
            .registered = false,
            .last_heartbeat_ms = 0,
            .cached_services = std.ArrayListUnmanaged(ServiceInstance).empty,
            .cache_valid_until_ms = 0,
            .node_registry = null,
            .owns_service_id = owns_id,
        };
    }

    pub fn deinit(self: *ServiceDiscovery) void {
        if (self.registered) {
            self.deregister() catch |err| {
                std.log.debug("ServiceDiscovery.deregister failed during deinit: {t}", .{err});
            };
        }
        self.clearCache();
        self.cached_services.deinit(self.allocator);
        // Free generated service_id if we own it
        if (self.owns_service_id) {
            self.allocator.free(self.config.service_id);
        }
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
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{\"ID\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_id);
        try buffer.appendSlice(self.allocator, "\",\"Name\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_name);
        try buffer.appendSlice(self.allocator, "\",\"Address\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_address);
        try buffer.appendSlice(self.allocator, "\",\"Port\":");

        var port_buf: [8]u8 = undefined;
        // SAFETY: u16 max is 65535 (5 digits), buffer is 8 bytes - cannot overflow
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

        var ttl_buf: [24]u8 = undefined;
        // SAFETY: u64 max is 20 digits + 's' suffix = 21 chars, buffer is 24 bytes - cannot overflow
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

            const id_copy = try self.allocator.dupe(u8, id.?);
            errdefer self.allocator.free(id_copy);
            const name_copy = try self.allocator.dupe(u8, name.?);
            errdefer self.allocator.free(name_copy);
            const addr_copy = try self.allocator.dupe(u8, address.?);
            errdefer self.allocator.free(addr_copy);

            const instance = ServiceInstance{
                .id = id_copy,
                .name = name_copy,
                .address = addr_copy,
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

        _ = try self.httpPost(url, body);
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

        _ = try self.httpPost(url, body);
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
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        try buffer.appendSlice(self.allocator, "{\"id\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_id);
        try buffer.appendSlice(self.allocator, "\",\"address\":\"");
        try buffer.appendSlice(self.allocator, self.config.service_address);
        try buffer.appendSlice(self.allocator, "\",\"port\":");

        var port_buf: [8]u8 = undefined;
        // SAFETY: u16 max is 65535 (5 digits), buffer is 8 bytes - cannot overflow
        const port_str = std.fmt.bufPrint(&port_buf, "{d}", .{self.config.service_port}) catch unreachable;
        try buffer.appendSlice(self.allocator, port_str);
        try buffer.appendSlice(self.allocator, ",\"timestamp\":");

        var ts_buf: [24]u8 = undefined;
        // SAFETY: i64 timestamp max is 20 digits + sign = 21 chars, buffer is 24 bytes - cannot overflow
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

    // ── HTTP/1.1 client ──────────────────────────────────────────────

    pub const HttpError = error{
        ConnectionRefused,
        Timeout,
        InvalidResponse,
        TooLarge,
        InvalidUrl,
    };

    pub const ParsedUrl = struct {
        host: []const u8,
        port: u16,
        path: []const u8,
    };

    /// Parse an `http://host:port/path` URL into components.
    /// Only plain HTTP is supported (no HTTPS).
    pub fn parseUrl(url: []const u8) HttpError!ParsedUrl {
        const scheme = "http://";
        if (!std.mem.startsWith(u8, url, scheme))
            return HttpError.InvalidUrl;

        const after_scheme = url[scheme.len..];
        // Split host+port from path at the first '/'
        const slash_idx = std.mem.indexOfScalar(u8, after_scheme, '/') orelse after_scheme.len;
        const host_port = after_scheme[0..slash_idx];
        const path = if (slash_idx < after_scheme.len) after_scheme[slash_idx..] else "/";

        // Split host and port at last ':'
        if (std.mem.lastIndexOfScalar(u8, host_port, ':')) |colon| {
            const port = std.fmt.parseInt(u16, host_port[colon + 1 ..], 10) catch
                return HttpError.InvalidUrl;
            return .{ .host = host_port[0..colon], .port = port, .path = path };
        }
        // No explicit port – default to 80
        return .{ .host = host_port, .port = 80, .path = path };
    }

    const c = std.c;
    const max_response_bytes: usize = 4 * 1024 * 1024; // 4 MiB
    const http_timeout_secs: c_int = 5;

    /// Parse a dotted-quad IPv4 string into 4 bytes.
    fn parseIp4Bytes(text: []const u8) HttpError![4]u8 {
        var bytes: [4]u8 = .{ 0, 0, 0, 0 };
        var octet_idx: usize = 0;
        var start: usize = 0;
        for (text, 0..) |ch, i| {
            if (ch == '.') {
                if (octet_idx >= 3) return HttpError.ConnectionRefused;
                bytes[octet_idx] = std.fmt.parseInt(u8, text[start..i], 10) catch
                    return HttpError.ConnectionRefused;
                octet_idx += 1;
                start = i + 1;
            }
        }
        if (octet_idx != 3) return HttpError.ConnectionRefused;
        bytes[3] = std.fmt.parseInt(u8, text[start..], 10) catch
            return HttpError.ConnectionRefused;
        return bytes;
    }

    /// Open a TCP socket to `host:port`, set timeouts, and return the fd.
    fn httpConnect(host: []const u8, port: u16) HttpError!c.fd_t {
        const ip_bytes = parseIp4Bytes(host) catch return HttpError.ConnectionRefused;

        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return HttpError.ConnectionRefused;
        errdefer _ = c.close(fd);

        // Set recv/send timeouts
        const tv = c.timeval{ .sec = http_timeout_secs, .usec = 0 };
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(c.timeval));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, @ptrCast(&tv), @sizeOf(c.timeval));

        var sa: c.sockaddr.in = .{
            .family = c.AF.INET,
            .port = @byteSwap(port),
            .addr = @bitCast(ip_bytes),
        };
        if (c.connect(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) < 0)
            return HttpError.ConnectionRefused;

        return fd;
    }

    /// Send all bytes on `fd`, handling partial writes.
    fn httpSendAll(fd: c.fd_t, data: []const u8) HttpError!void {
        var sent: usize = 0;
        while (sent < data.len) {
            const n = c.send(fd, data[sent..].ptr, data.len - sent, 0);
            if (n <= 0) return HttpError.ConnectionRefused;
            sent += @intCast(n);
        }
    }

    /// Read the full response from `fd` into an allocated buffer.
    fn httpRecvAll(self: *ServiceDiscovery, fd: c.fd_t) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        errdefer buf.deinit(self.allocator);

        var tmp: [4096]u8 = undefined;
        while (true) {
            const n = c.recv(fd, &tmp, tmp.len, 0);
            if (n < 0) return HttpError.Timeout;
            if (n == 0) break;
            const count: usize = @intCast(n);
            if (buf.items.len + count > max_response_bytes) return HttpError.TooLarge;
            try buf.appendSlice(self.allocator, tmp[0..count]);
        }
        return buf.toOwnedSlice(self.allocator);
    }

    /// Extract the body from a raw HTTP response (everything after `\r\n\r\n`).
    pub fn httpExtractBody(self: *ServiceDiscovery, raw: []const u8) ![]const u8 {
        const delim = "\r\n\r\n";
        const idx = std.mem.indexOf(u8, raw, delim) orelse
            return HttpError.InvalidResponse;
        return try self.allocator.dupe(u8, raw[idx + delim.len ..]);
    }

    fn httpRequestImpl(self: *ServiceDiscovery, method: []const u8, url: []const u8, body: []const u8) ![]const u8 {
        const parsed = try parseUrl(url);

        const fd = try httpConnect(parsed.host, parsed.port);
        defer _ = c.close(fd);

        // Build request into a single buffer
        var req_buf: std.ArrayListUnmanaged(u8) = .empty;
        defer req_buf.deinit(self.allocator);

        // Request line
        try req_buf.appendSlice(self.allocator, method);
        try req_buf.appendSlice(self.allocator, " ");
        try req_buf.appendSlice(self.allocator, parsed.path);
        try req_buf.appendSlice(self.allocator, " HTTP/1.1\r\nHost: ");
        try req_buf.appendSlice(self.allocator, parsed.host);
        try req_buf.appendSlice(self.allocator, "\r\nConnection: close\r\n");

        if (body.len > 0) {
            try req_buf.appendSlice(self.allocator, "Content-Type: application/json\r\nContent-Length: ");
            var len_buf: [20]u8 = undefined;
            const len_str = std.fmt.bufPrint(&len_buf, "{d}", .{body.len}) catch unreachable;
            try req_buf.appendSlice(self.allocator, len_str);
            try req_buf.appendSlice(self.allocator, "\r\n");
        }

        try req_buf.appendSlice(self.allocator, "\r\n");
        if (body.len > 0) {
            try req_buf.appendSlice(self.allocator, body);
        }

        try httpSendAll(fd, req_buf.items);

        const raw = try self.httpRecvAll(fd);
        defer self.allocator.free(raw);

        return self.httpExtractBody(raw);
    }

    fn httpGet(self: *ServiceDiscovery, url: []const u8) ![]const u8 {
        return self.httpRequestImpl("GET", url, "");
    }

    fn httpPut(self: *ServiceDiscovery, url: []const u8, body: []const u8) !void {
        const response = try self.httpRequestImpl("PUT", url, body);
        self.allocator.free(response);
    }

    fn httpPost(self: *ServiceDiscovery, url: []const u8, body: []const u8) ![]const u8 {
        return self.httpRequestImpl("POST", url, body);
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

test "parseUrl extracts host, port, and path" {
    // Standard URL with host, port, and path
    const r1 = try ServiceDiscovery.parseUrl("http://127.0.0.1:8500/v1/health/service/web");
    try std.testing.expectEqualStrings("127.0.0.1", r1.host);
    try std.testing.expectEqual(@as(u16, 8500), r1.port);
    try std.testing.expectEqualStrings("/v1/health/service/web", r1.path);

    // URL with no explicit path
    const r2 = try ServiceDiscovery.parseUrl("http://10.0.0.1:2379");
    try std.testing.expectEqualStrings("10.0.0.1", r2.host);
    try std.testing.expectEqual(@as(u16, 2379), r2.port);
    try std.testing.expectEqualStrings("/", r2.path);

    // URL with default port (no port specified)
    const r3 = try ServiceDiscovery.parseUrl("http://consul.local/v1/agent");
    try std.testing.expectEqualStrings("consul.local", r3.host);
    try std.testing.expectEqual(@as(u16, 80), r3.port);
    try std.testing.expectEqualStrings("/v1/agent", r3.path);

    // Invalid scheme
    try std.testing.expectError(
        ServiceDiscovery.HttpError.InvalidUrl,
        ServiceDiscovery.parseUrl("https://secure.host/path"),
    );

    // Invalid port
    try std.testing.expectError(
        ServiceDiscovery.HttpError.InvalidUrl,
        ServiceDiscovery.parseUrl("http://host:notaport/path"),
    );
}

test "httpExtractBody parses HTTP response" {
    const allocator = std.testing.allocator;
    var discovery = try ServiceDiscovery.init(allocator, .{
        .service_name = "test-http",
    });
    defer discovery.deinit();

    const raw = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\n{\"status\":\"ok\"}";
    const body = try discovery.httpExtractBody(raw);
    defer allocator.free(body);

    try std.testing.expectEqualStrings("{\"status\":\"ok\"}", body);
}

test "httpGet against real server requires network (skip)" {
    // No server available in test environment
    return error.SkipZigTest;
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

test {
    std.testing.refAllDecls(@This());
}
