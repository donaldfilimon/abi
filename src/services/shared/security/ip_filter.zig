//! IP-based filtering and access control.
//!
//! This module provides:
//! - IP allowlist (whitelist)
//! - IP blocklist (blacklist)
//! - CIDR notation support
//! - Temporary bans with expiration
//! - Geographic filtering (placeholder)
//! - Rate-limit integration
//! - Automatic threat detection

const std = @import("std");
const time = @import("../time.zig");
const sync = @import("../sync.zig");

/// IP version
pub const IpVersion = enum {
    v4,
    v6,
};

/// IP address representation
pub const IpAddress = struct {
    bytes: [16]u8,
    version: IpVersion,

    /// Parse IP address from string
    pub fn parse(str: []const u8) ?IpAddress {
        // Try IPv4 first
        if (parseIpv4(str)) |ip| return ip;
        // Try IPv6
        if (parseIpv6(str)) |ip| return ip;
        return null;
    }

    /// Format IP address as string
    pub fn format(self: IpAddress, allocator: std.mem.Allocator) ![]const u8 {
        return switch (self.version) {
            .v4 => std.fmt.allocPrint(allocator, "{d}.{d}.{d}.{d}", .{
                self.bytes[0],
                self.bytes[1],
                self.bytes[2],
                self.bytes[3],
            }),
            .v6 => std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}:{x:0>2}{x:0>2}", .{
                self.bytes[0],  self.bytes[1],  self.bytes[2],  self.bytes[3],
                self.bytes[4],  self.bytes[5],  self.bytes[6],  self.bytes[7],
                self.bytes[8],  self.bytes[9],  self.bytes[10], self.bytes[11],
                self.bytes[12], self.bytes[13], self.bytes[14], self.bytes[15],
            }),
        };
    }

    /// Check equality
    pub fn eql(self: IpAddress, other: IpAddress) bool {
        if (self.version != other.version) return false;
        const len: usize = if (self.version == .v4) 4 else 16;
        return std.mem.eql(u8, self.bytes[0..len], other.bytes[0..len]);
    }

    /// Check if IP is in a CIDR range
    pub fn inCidr(self: IpAddress, cidr: CidrRange) bool {
        if (self.version != cidr.base.version) return false;

        const len: usize = if (self.version == .v4) 4 else 16;
        const full_bytes = cidr.prefix_len / 8;
        const remaining_bits = cidr.prefix_len % 8;

        // Check full bytes
        for (0..full_bytes) |i| {
            if (self.bytes[i] != cidr.base.bytes[i]) return false;
        }

        // Check remaining bits
        if (remaining_bits > 0 and full_bytes < len) {
            const mask: u8 = @as(u8, 0xFF) << @intCast(8 - remaining_bits);
            if ((self.bytes[full_bytes] & mask) != (cidr.base.bytes[full_bytes] & mask)) {
                return false;
            }
        }

        return true;
    }

    /// Check if this is a private IP address
    pub fn isPrivate(self: IpAddress) bool {
        if (self.version == .v4) {
            // 10.0.0.0/8
            if (self.bytes[0] == 10) return true;
            // 172.16.0.0/12
            if (self.bytes[0] == 172 and (self.bytes[1] & 0xF0) == 16) return true;
            // 192.168.0.0/16
            if (self.bytes[0] == 192 and self.bytes[1] == 168) return true;
            // 127.0.0.0/8 (loopback)
            if (self.bytes[0] == 127) return true;
        } else {
            // ::1 (loopback)
            var is_loopback = true;
            for (self.bytes[0..15]) |b| {
                if (b != 0) {
                    is_loopback = false;
                    break;
                }
            }
            if (is_loopback and self.bytes[15] == 1) return true;

            // fe80::/10 (link-local)
            if (self.bytes[0] == 0xFE and (self.bytes[1] & 0xC0) == 0x80) return true;

            // fc00::/7 (unique local)
            if ((self.bytes[0] & 0xFE) == 0xFC) return true;
        }
        return false;
    }
};

fn parseIpv4(str: []const u8) ?IpAddress {
    var result = IpAddress{
        .bytes = [_]u8{0} ** 16,
        .version = .v4,
    };

    var parts: u8 = 0;
    var current_value: u32 = 0;
    var has_digit = false;

    for (str) |c| {
        if (c == '.') {
            if (!has_digit or parts >= 3 or current_value > 255) return null;
            result.bytes[parts] = @intCast(current_value);
            parts += 1;
            current_value = 0;
            has_digit = false;
        } else if (c >= '0' and c <= '9') {
            current_value = current_value * 10 + (c - '0');
            has_digit = true;
        } else {
            return null;
        }
    }

    if (!has_digit or parts != 3 or current_value > 255) return null;
    result.bytes[3] = @intCast(current_value);

    return result;
}

fn parseIpv6(str: []const u8) ?IpAddress {
    // Simplified IPv6 parsing
    var result = IpAddress{
        .bytes = [_]u8{0} ** 16,
        .version = .v6,
    };

    // Handle :: expansion
    const double_colon = std.mem.indexOf(u8, str, "::");
    _ = double_colon;

    // For now, just parse hex groups
    var byte_idx: usize = 0;
    var current_value: u16 = 0;
    var chars_in_group: u8 = 0;

    for (str) |c| {
        if (c == ':') {
            if (chars_in_group > 0) {
                if (byte_idx >= 14) return null;
                result.bytes[byte_idx] = @intCast(current_value >> 8);
                result.bytes[byte_idx + 1] = @intCast(current_value & 0xFF);
                byte_idx += 2;
                current_value = 0;
                chars_in_group = 0;
            }
        } else if (std.ascii.isHex(c)) {
            if (chars_in_group >= 4) return null;
            const val = std.fmt.charToDigit(c, 16) catch return null;
            current_value = current_value * 16 + val;
            chars_in_group += 1;
        } else {
            return null;
        }
    }

    // Last group
    if (chars_in_group > 0 and byte_idx < 16) {
        result.bytes[byte_idx] = @intCast(current_value >> 8);
        result.bytes[byte_idx + 1] = @intCast(current_value & 0xFF);
    }

    return result;
}

/// CIDR range
pub const CidrRange = struct {
    base: IpAddress,
    prefix_len: u8,

    /// Parse CIDR from string (e.g., "192.168.1.0/24")
    pub fn parse(str: []const u8) ?CidrRange {
        const slash_idx = std.mem.indexOf(u8, str, "/") orelse return null;

        const ip_str = str[0..slash_idx];
        const prefix_str = str[slash_idx + 1 ..];

        const ip = IpAddress.parse(ip_str) orelse return null;
        const prefix = std.fmt.parseInt(u8, prefix_str, 10) catch return null;

        const max_prefix: u8 = if (ip.version == .v4) 32 else 128;
        if (prefix > max_prefix) return null;

        return .{
            .base = ip,
            .prefix_len = prefix,
        };
    }

    /// Check if an IP is in this range
    pub fn contains(self: CidrRange, ip: IpAddress) bool {
        return ip.inCidr(self);
    }
};

/// Block reason
pub const BlockReason = enum {
    manual,
    rate_limit,
    brute_force,
    malicious_request,
    geographic,
    reputation,
    temporary,

    pub fn toString(self: BlockReason) []const u8 {
        return switch (self) {
            .manual => "Manually blocked",
            .rate_limit => "Rate limit exceeded",
            .brute_force => "Brute force detected",
            .malicious_request => "Malicious request",
            .geographic => "Geographic restriction",
            .reputation => "Bad reputation",
            .temporary => "Temporary block",
        };
    }
};

/// Block entry
pub const BlockEntry = struct {
    ip: IpAddress,
    reason: BlockReason,
    blocked_at: i64,
    expires_at: ?i64,
    description: ?[]const u8,
    violations: u32,
};

/// IP filter configuration
pub const IpFilterConfig = struct {
    /// Enable filtering
    enabled: bool = true,
    /// Default action for unknown IPs
    default_action: Action = .allow,
    /// Block duration for automatic bans (seconds)
    auto_ban_duration: i64 = 3600,
    /// Violations before auto-ban
    violations_threshold: u32 = 10,
    /// Allow private IPs regardless of rules
    allow_private: bool = true,
    /// Enable logging
    enable_logging: bool = true,
    /// Maximum entries in blocklist
    max_blocklist_size: usize = 10000,
    /// Maximum entries in allowlist
    max_allowlist_size: usize = 1000,

    pub const Action = enum {
        allow,
        deny,
    };
};

/// IP filter
pub const IpFilter = struct {
    allocator: std.mem.Allocator,
    config: IpFilterConfig,
    /// Blocked IPs
    blocklist: std.ArrayListUnmanaged(BlockEntry),
    /// Blocked CIDR ranges
    blocked_ranges: std.ArrayListUnmanaged(BlockedRange),
    /// Allowed IPs
    allowlist: std.ArrayListUnmanaged(IpAddress),
    /// Allowed CIDR ranges
    allowed_ranges: std.ArrayListUnmanaged(CidrRange),
    /// Violation counts
    violations: std.AutoHashMapUnmanaged(u128, ViolationInfo),
    /// Statistics
    stats: IpFilterStats,
    mutex: sync.Mutex,

    const BlockedRange = struct {
        range: CidrRange,
        reason: BlockReason,
        expires_at: ?i64,
    };

    const ViolationInfo = struct {
        count: u32,
        first_seen: i64,
        last_seen: i64,
    };

    pub const IpFilterStats = struct {
        total_checks: u64 = 0,
        allowed: u64 = 0,
        blocked: u64 = 0,
        auto_bans: u64 = 0,
        active_blocks: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: IpFilterConfig) IpFilter {
        return .{
            .allocator = allocator,
            .config = config,
            .blocklist = std.ArrayListUnmanaged(BlockEntry).empty,
            .blocked_ranges = std.ArrayListUnmanaged(BlockedRange).empty,
            .allowlist = std.ArrayListUnmanaged(IpAddress).empty,
            .allowed_ranges = std.ArrayListUnmanaged(CidrRange).empty,
            .violations = std.AutoHashMapUnmanaged(u128, ViolationInfo){},
            .stats = .{},
            .mutex = .{},
        };
    }

    pub fn deinit(self: *IpFilter) void {
        for (self.blocklist.items) |entry| {
            if (entry.description) |desc| {
                self.allocator.free(desc);
            }
        }
        self.blocklist.deinit(self.allocator);
        self.blocked_ranges.deinit(self.allocator);
        self.allowlist.deinit(self.allocator);
        self.allowed_ranges.deinit(self.allocator);
        self.violations.deinit(self.allocator);
    }

    /// Check if an IP is allowed
    pub fn check(self: *IpFilter, ip_str: []const u8) FilterResult {
        if (!self.config.enabled) {
            return .{ .allowed = true, .reason = null };
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        self.stats.total_checks += 1;

        const ip = IpAddress.parse(ip_str) orelse {
            // Invalid IP - block
            self.stats.blocked += 1;
            return .{ .allowed = false, .reason = .manual };
        };

        // Allow private IPs if configured
        if (self.config.allow_private and ip.isPrivate()) {
            self.stats.allowed += 1;
            return .{ .allowed = true, .reason = null };
        }

        // Check explicit allowlist first
        for (self.allowlist.items) |allowed| {
            if (ip.eql(allowed)) {
                self.stats.allowed += 1;
                return .{ .allowed = true, .reason = null };
            }
        }

        // Check allowed ranges
        for (self.allowed_ranges.items) |range| {
            if (range.contains(ip)) {
                self.stats.allowed += 1;
                return .{ .allowed = true, .reason = null };
            }
        }

        const now = time.unixSeconds();

        // Check blocklist
        for (self.blocklist.items) |entry| {
            if (ip.eql(entry.ip)) {
                // Check if block has expired
                if (entry.expires_at) |exp| {
                    if (now > exp) continue;
                }
                self.stats.blocked += 1;
                return .{ .allowed = false, .reason = entry.reason };
            }
        }

        // Check blocked ranges
        for (self.blocked_ranges.items) |entry| {
            if (entry.expires_at) |exp| {
                if (now > exp) continue;
            }
            if (entry.range.contains(ip)) {
                self.stats.blocked += 1;
                return .{ .allowed = false, .reason = entry.reason };
            }
        }

        // Default action
        if (self.config.default_action == .deny) {
            self.stats.blocked += 1;
            return .{ .allowed = false, .reason = .manual };
        }

        self.stats.allowed += 1;
        return .{ .allowed = true, .reason = null };
    }

    pub const FilterResult = struct {
        allowed: bool,
        reason: ?BlockReason,
    };

    /// Block an IP
    pub fn block(self: *IpFilter, ip_str: []const u8, reason: BlockReason, duration: ?i64, description: ?[]const u8) !void {
        const ip = IpAddress.parse(ip_str) orelse return error.InvalidIpAddress;

        self.mutex.lock();
        defer self.mutex.unlock();

        // Check size limit
        if (self.blocklist.items.len >= self.config.max_blocklist_size) {
            // Remove oldest entry
            if (self.blocklist.items.len > 0) {
                const removed = self.blocklist.orderedRemove(0);
                if (removed.description) |desc| {
                    self.allocator.free(desc);
                }
            }
        }

        const now = time.unixSeconds();

        try self.blocklist.append(self.allocator, .{
            .ip = ip,
            .reason = reason,
            .blocked_at = now,
            .expires_at = if (duration) |d| now + d else null,
            .description = if (description) |d| try self.allocator.dupe(u8, d) else null,
            .violations = 0,
        });

        self.stats.active_blocks += 1;
    }

    /// Block a CIDR range
    pub fn blockRange(self: *IpFilter, cidr_str: []const u8, reason: BlockReason, duration: ?i64) !void {
        const range = CidrRange.parse(cidr_str) orelse return error.InvalidCidrRange;

        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();

        try self.blocked_ranges.append(self.allocator, .{
            .range = range,
            .reason = reason,
            .expires_at = if (duration) |d| now + d else null,
        });
    }

    /// Unblock an IP
    pub fn unblock(self: *IpFilter, ip_str: []const u8) bool {
        const ip = IpAddress.parse(ip_str) orelse return false;

        self.mutex.lock();
        defer self.mutex.unlock();

        var i: usize = 0;
        while (i < self.blocklist.items.len) {
            if (ip.eql(self.blocklist.items[i].ip)) {
                const removed = self.blocklist.orderedRemove(i);
                if (removed.description) |desc| {
                    self.allocator.free(desc);
                }
                self.stats.active_blocks -|= 1;
                return true;
            } else {
                i += 1;
            }
        }

        return false;
    }

    /// Allow an IP (add to allowlist)
    pub fn allow(self: *IpFilter, ip_str: []const u8) !void {
        const ip = IpAddress.parse(ip_str) orelse return error.InvalidIpAddress;

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.allowlist.items.len >= self.config.max_allowlist_size) {
            return error.AllowlistFull;
        }

        try self.allowlist.append(self.allocator, ip);
    }

    /// Allow a CIDR range
    pub fn allowRange(self: *IpFilter, cidr_str: []const u8) !void {
        const range = CidrRange.parse(cidr_str) orelse return error.InvalidCidrRange;

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.allowed_ranges.append(self.allocator, range);
    }

    /// Record a violation for an IP
    pub fn recordViolation(self: *IpFilter, ip_str: []const u8) !void {
        const ip = IpAddress.parse(ip_str) orelse return error.InvalidIpAddress;

        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();
        const key = ipToKey(ip);

        const result = try self.violations.getOrPut(self.allocator, key);
        if (result.found_existing) {
            result.value_ptr.count += 1;
            result.value_ptr.last_seen = now;
        } else {
            result.value_ptr.* = .{
                .count = 1,
                .first_seen = now,
                .last_seen = now,
            };
        }

        // Check if should auto-ban
        if (result.value_ptr.count >= self.config.violations_threshold) {
            // Add to blocklist
            try self.blocklist.append(self.allocator, .{
                .ip = ip,
                .reason = .brute_force,
                .blocked_at = now,
                .expires_at = now + self.config.auto_ban_duration,
                .description = null,
                .violations = result.value_ptr.count,
            });

            self.stats.auto_bans += 1;
            self.stats.active_blocks += 1;

            // Reset violation count
            result.value_ptr.count = 0;
        }
    }

    /// Clean up expired entries
    pub fn cleanup(self: *IpFilter) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixSeconds();
        var removed: u32 = 0;

        // Clean blocklist
        var i: usize = 0;
        while (i < self.blocklist.items.len) {
            if (self.blocklist.items[i].expires_at) |exp| {
                if (now > exp) {
                    const entry = self.blocklist.orderedRemove(i);
                    if (entry.description) |desc| {
                        self.allocator.free(desc);
                    }
                    removed += 1;
                    self.stats.active_blocks -|= 1;
                    continue;
                }
            }
            i += 1;
        }

        // Clean blocked ranges
        i = 0;
        while (i < self.blocked_ranges.items.len) {
            if (self.blocked_ranges.items[i].expires_at) |exp| {
                if (now > exp) {
                    _ = self.blocked_ranges.orderedRemove(i);
                    removed += 1;
                    continue;
                }
            }
            i += 1;
        }

        return removed;
    }

    /// Get statistics
    pub fn getStats(self: *IpFilter) IpFilterStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    fn ipToKey(ip: IpAddress) u128 {
        var key: u128 = 0;
        const len: usize = if (ip.version == .v4) 4 else 16;
        for (ip.bytes[0..len]) |b| {
            key = (key << 8) | b;
        }
        return key;
    }
};

/// IP filter errors
pub const IpFilterError = error{
    InvalidIpAddress,
    InvalidCidrRange,
    BlocklistFull,
    AllowlistFull,
    OutOfMemory,
};

// Tests

test "ipv4 parsing" {
    const ip = IpAddress.parse("192.168.1.1").?;
    try std.testing.expectEqual(IpVersion.v4, ip.version);
    try std.testing.expectEqual(@as(u8, 192), ip.bytes[0]);
    try std.testing.expectEqual(@as(u8, 168), ip.bytes[1]);
    try std.testing.expectEqual(@as(u8, 1), ip.bytes[2]);
    try std.testing.expectEqual(@as(u8, 1), ip.bytes[3]);

    try std.testing.expect(IpAddress.parse("256.1.1.1") == null);
    try std.testing.expect(IpAddress.parse("1.2.3") == null);
}

test "cidr matching" {
    const range = CidrRange.parse("192.168.0.0/16").?;

    const ip1 = IpAddress.parse("192.168.1.1").?;
    try std.testing.expect(range.contains(ip1));

    const ip2 = IpAddress.parse("192.168.255.255").?;
    try std.testing.expect(range.contains(ip2));

    const ip3 = IpAddress.parse("192.169.0.1").?;
    try std.testing.expect(!range.contains(ip3));
}

test "private ip detection" {
    const private1 = IpAddress.parse("10.0.0.1").?;
    try std.testing.expect(private1.isPrivate());

    const private2 = IpAddress.parse("192.168.1.1").?;
    try std.testing.expect(private2.isPrivate());

    const private3 = IpAddress.parse("172.16.0.1").?;
    try std.testing.expect(private3.isPrivate());

    const public = IpAddress.parse("8.8.8.8").?;
    try std.testing.expect(!public.isPrivate());
}

test "ip filter basic operations" {
    const allocator = std.testing.allocator;
    var filter = IpFilter.init(allocator, .{});
    defer filter.deinit();

    // Block an IP
    try filter.block("1.2.3.4", .manual, null, "Test block");

    // Check blocked
    const result1 = filter.check("1.2.3.4");
    try std.testing.expect(!result1.allowed);
    try std.testing.expectEqual(BlockReason.manual, result1.reason.?);

    // Check allowed
    const result2 = filter.check("8.8.8.8");
    try std.testing.expect(result2.allowed);

    // Unblock
    try std.testing.expect(filter.unblock("1.2.3.4"));

    // Should be allowed now
    const result3 = filter.check("1.2.3.4");
    try std.testing.expect(result3.allowed);
}

test "allowlist takes precedence" {
    const allocator = std.testing.allocator;
    var filter = IpFilter.init(allocator, .{ .default_action = .deny });
    defer filter.deinit();

    // Add to allowlist
    try filter.allow("1.2.3.4");

    // Should be allowed even with deny default
    const result = filter.check("1.2.3.4");
    try std.testing.expect(result.allowed);

    // Other IPs should be blocked
    const result2 = filter.check("5.6.7.8");
    try std.testing.expect(!result2.allowed);
}

test {
    std.testing.refAllDecls(@This());
}
