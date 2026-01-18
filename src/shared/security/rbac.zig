//! Role-based access control (RBAC) implementation.
const std = @import("std");
const time = @import("../utils.zig");

pub const Permission = enum {
    read,
    write,
    delete,
    admin,
    execute,
    manage_users,
    manage_roles,
    view_metrics,
    view_logs,
    configure,
};

pub const Role = struct {
    name: []const u8,
    permissions: []const Permission,
    is_system: bool,
    description: []const u8,
};

pub const RoleAssignment = struct {
    user_id: []const u8,
    role_name: []const u8,
    granted_at: i64,
    granted_by: []const u8,
    expires_at: ?i64,
};

pub const RbacConfig = struct {
    default_roles: bool = true,
    allow_custom_roles: bool = true,
    max_roles_per_user: usize = 10,
    permission_cache_size: usize = 1000,
};

pub const RbacManager = struct {
    allocator: std.mem.Allocator,
    config: RbacConfig,
    roles: std.StringArrayHashMapUnmanaged(*Role),
    role_assignments: std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(RoleAssignment)),
    user_permissions: std.AutoHashMapUnmanaged(u64, []const Permission),
    permission_cache: std.AutoHashMapUnmanaged(u64, bool),

    pub fn init(allocator: std.mem.Allocator, config: RbacConfig) !RbacManager {
        var manager = RbacManager{
            .allocator = allocator,
            .config = config,
            .roles = std.StringArrayHashMapUnmanaged(*Role).empty,
            .role_assignments = std.StringArrayHashMapUnmanaged(std.ArrayListUnmanaged(RoleAssignment)).empty,
            .user_permissions = .{},
            .permission_cache = .{},
        };

        if (config.default_roles) {
            try manager.createDefaultRoles();
        }

        return manager;
    }

    pub fn deinit(self: *RbacManager) void {
        var role_it = self.roles.iterator();
        while (role_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr);
        }
        self.roles.deinit(self.allocator);

        var assign_it = self.role_assignments.iterator();
        while (assign_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |assignment| {
                self.allocator.free(assignment.user_id);
                self.allocator.free(assignment.role_name);
                self.allocator.free(assignment.granted_by);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.role_assignments.deinit(self.allocator);

        var perm_it = self.user_permissions.valueIterator();
        while (perm_it.next()) |perms| {
            self.allocator.free(perms.*);
        }
        self.user_permissions.deinit(self.allocator);
        self.permission_cache.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn createRole(
        self: *RbacManager,
        name: []const u8,
        permissions: []const Permission,
        description: []const u8,
    ) !void {
        if (!self.config.allow_custom_roles and !isSystemRoleName(name)) {
            return error.CustomRolesNotAllowed;
        }

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        const description_copy = try self.allocator.dupe(u8, description);
        errdefer self.allocator.free(description_copy);

        const perms_copy = try self.allocator.alloc(Permission, permissions.len);
        @memcpy(perms_copy, permissions);

        const role = try self.allocator.create(Role);
        errdefer self.allocator.destroy(role);

        role.* = .{
            .name = name_copy,
            .permissions = perms_copy,
            .is_system = isSystemRoleName(name),
            .description = description_copy,
        };

        try self.roles.put(self.allocator, name_copy, role);
    }

    pub fn assignRole(
        self: *RbacManager,
        user_id: []const u8,
        role_name: []const u8,
        granted_by: []const u8,
    ) !void {
        const role = self.roles.get(role_name) orelse return error.RoleNotFound;
        if (role == null) return error.RoleNotFound;

        const granted_by_copy = try self.allocator.dupe(u8, granted_by);
        errdefer self.allocator.free(granted_by_copy);

        const assignment = RoleAssignment{
            .user_id = try self.allocator.dupe(u8, user_id),
            .role_name = try self.allocator.dupe(u8, role_name),
            .granted_at = time.unixSeconds(),
            .granted_by = granted_by_copy,
            .expires_at = null,
        };

        var assignments = self.role_assignments.get(user_id) orelse blk: {
            const list = std.ArrayListUnmanaged(RoleAssignment).empty;
            try self.role_assignments.put(self.allocator, try self.allocator.dupe(u8, user_id), list);
            break :blk self.role_assignments.get(user_id).?;
        };
        try assignments.append(self.allocator, assignment);

        self.invalidateUserCache(user_id);
    }

    pub fn revokeRole(self: *RbacManager, user_id: []const u8, role_name: []const u8) bool {
        const assignments = self.role_assignments.get(user_id) orelse return false;
        var found = false;
        var i: usize = 0;
        while (i < assignments.items.len) {
            if (std.mem.eql(u8, assignments.items[i].role_name, role_name)) {
                const removed = assignments.orderedRemove(i);
                self.allocator.free(removed.user_id);
                self.allocator.free(removed.role_name);
                self.allocator.free(removed.granted_by);
                found = true;
            } else {
                i += 1;
            }
        }
        if (found) {
            self.invalidateUserCache(user_id);
        }
        return found;
    }

    pub fn hasPermission(
        self: *RbacManager,
        user_id: []const u8,
        permission: Permission,
    ) !bool {
        const cache_key = self.getCacheKey(user_id, permission);
        if (self.permission_cache.get(cache_key)) |cached| {
            return cached;
        }

        const has_perm = self.checkPermissionDirect(user_id, permission);
        try self.permission_cache.put(self.allocator, cache_key, has_perm);
        return has_perm;
    }

    pub fn hasAnyPermission(
        self: *RbacManager,
        user_id: []const u8,
        permissions: []const Permission,
    ) !bool {
        for (permissions) |perm| {
            if (try self.hasPermission(user_id, perm)) {
                return true;
            }
        }
        return false;
    }

    pub fn hasAllPermissions(
        self: *RbacManager,
        user_id: []const u8,
        permissions: []const Permission,
    ) !bool {
        for (permissions) |perm| {
            if (!try self.hasPermission(user_id, perm)) {
                return false;
            }
        }
        return true;
    }

    pub fn getUserRoles(self: *RbacManager, user_id: []const u8) []*const Role {
        const assignments = self.role_assignments.get(user_id) orelse return &.{};
        var result = std.ArrayListUnmanaged(*const Role).empty;
        for (assignments.items) |assignment| {
            if (self.roles.get(assignment.role_name)) |role| {
                result.appendAssumeCapacity(role);
            }
        }
        return result.items;
    }

    pub fn getRolePermissions(self: *RbacManager, role_name: []const u8) ?[]const Permission {
        const role = self.roles.get(role_name) orelse return null;
        return role.?.permissions;
    }

    fn checkPermissionDirect(self: *RbacManager, user_id: []const u8, permission: Permission) bool {
        const roles = self.getUserRoles(user_id);
        for (roles) |role| {
            for (role.permissions) |role_perm| {
                if (role_perm == permission) return true;
            }
        }
        return false;
    }

    /// Invalidate cached permissions for a specific user.
    /// Called when role assignments change.
    fn invalidateUserCache(self: *RbacManager, user_id: []const u8) void {
        // Remove all cached permission entries for this user
        // by iterating through the cache and removing matching keys
        var keys_to_remove = std.ArrayListUnmanaged(u64).empty;
        defer keys_to_remove.deinit(self.allocator);

        // Find all cache keys for this user
        var it = self.permission_cache.iterator();
        while (it.next()) |entry| {
            // Check if this cache key belongs to this user by comparing hash prefix
            const user_hash = self.computeUserHash(user_id);
            // Keys are composed of user hash + permission enum, so we check the prefix
            const key = entry.key_ptr.*;
            if ((key / 17) == user_hash) {
                keys_to_remove.append(self.allocator, key) catch continue;
            }
        }

        // Remove identified cache entries
        for (keys_to_remove.items) |key| {
            _ = self.permission_cache.remove(key);
        }

        // Also remove from user_permissions cache
        const user_hash = self.computeUserHash(user_id);
        if (self.user_permissions.fetchRemove(user_hash)) |kv| {
            self.allocator.free(kv.value);
        }
    }

    /// Compute a hash for the user ID to use as cache key prefix
    fn computeUserHash(_: *RbacManager, user_id: []const u8) u64 {
        var hash: u64 = 0;
        for (user_id) |byte| {
            hash = hash *% 31 +% byte;
        }
        return hash;
    }

    /// Clear all cached permissions (useful for bulk updates)
    pub fn clearPermissionCache(self: *RbacManager) void {
        // Clear user permissions cache
        var perm_it = self.user_permissions.valueIterator();
        while (perm_it.next()) |perms| {
            self.allocator.free(perms.*);
        }
        self.user_permissions.clearRetainingCapacity();

        // Clear permission check cache
        self.permission_cache.clearRetainingCapacity();
    }

    fn getCacheKey(_: *RbacManager, user_id: []const u8, permission: Permission) u64 {
        var hash: u64 = 0;
        for (user_id) |byte| {
            hash = hash * 31 + byte;
        }
        hash = hash * 17 + @intFromEnum(permission);
        return hash;
    }

    fn createDefaultRoles(self: *RbacManager) !void {
        try self.createRole("admin", &.{
            .read,
            .write,
            .delete,
            .admin,
            .execute,
            .manage_users,
            .manage_roles,
            .view_metrics,
            .view_logs,
            .configure,
        }, "Full administrative access");

        try self.createRole("user", &.{ .read, .write, .execute }, "Regular user access");

        try self.createRole("readonly", &.{.read}, "Read-only access");

        try self.createRole("metrics", &.{ .view_metrics, .view_logs }, "Access to metrics and logs");

        try self.createRole("manager", &.{
            .read,
            .write,
            .execute,
            .manage_users,
            .view_metrics,
            .view_logs,
        }, "Management access");
    }
};

fn isSystemRoleName(name: []const u8) bool {
    const system_roles = &.{ "admin", "user", "readonly", "metrics", "manager" };
    for (system_roles) |role| {
        if (std.mem.eql(u8, name, role)) return true;
    }
    return false;
}

pub const RbacError = error{
    RoleNotFound,
    CustomRolesNotAllowed,
    MaxRolesExceeded,
    PermissionDenied,
};

test "rbac role creation" {
    const allocator = std.testing.allocator;
    var rbac = try RbacManager.init(allocator, .{});
    defer rbac.deinit();

    try rbac.createRole("test_role", &.{ .read, .write }, "Test role");

    const role = rbac.roles.get("test_role");
    try std.testing.expect(role != null);
    try std.testing.expectEqualStrings("test_role", role.?.name);
}

test "rbac role assignment" {
    const allocator = std.testing.allocator;
    var rbac = try RbacManager.init(allocator, .{});
    defer rbac.deinit();

    try rbac.assignRole("user1", "admin", "system");

    const has_admin = try rbac.hasPermission("user1", .admin);
    try std.testing.expect(has_admin);

    const has_delete = try rbac.hasPermission("user1", .delete);
    try std.testing.expect(has_delete);
}

test "rbac permission check" {
    const allocator = std.testing.allocator;
    var rbac = try RbacManager.init(allocator, .{});
    defer rbac.deinit();

    try rbac.assignRole("user1", "readonly", "system");

    const has_read = try rbac.hasPermission("user1", .read);
    try std.testing.expect(has_read);

    const has_write = try rbac.hasPermission("user1", .write);
    try std.testing.expect(!has_write);
}

test "rbac role revocation" {
    const allocator = std.testing.allocator;
    var rbac = try RbacManager.init(allocator, .{});
    defer rbac.deinit();

    try rbac.assignRole("user1", "admin", "system");
    const revoked = rbac.revokeRole("user1", "admin");
    try std.testing.expect(revoked);

    const has_admin = try rbac.hasPermission("user1", .admin);
    try std.testing.expect(!has_admin);
}
