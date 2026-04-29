//! Permission management for the mobile feature.
//!
//! Simulated permission system that tracks grant/deny state per permission type.

const types = @import("types.zig");

const Permission = types.Permission;
const PermissionStatus = types.PermissionStatus;
const permission_count = types.permission_count;

/// Check the current status of a permission.
pub fn checkPermission(permissions: *const [permission_count]PermissionStatus, perm: Permission) PermissionStatus {
    return permissions[@intFromEnum(perm)];
}

/// Request a permission (simulated: always grants).
pub fn requestPermission(permissions: *[permission_count]PermissionStatus, perm: Permission) PermissionStatus {
    permissions[@intFromEnum(perm)] = .granted;
    return .granted;
}

/// Revoke a previously granted permission.
pub fn revokePermission(permissions: *[permission_count]PermissionStatus, perm: Permission) void {
    permissions[@intFromEnum(perm)] = .denied;
}

const std = @import("std");

test "permission lifecycle" {
    var perms: [permission_count]PermissionStatus = @splat(.not_requested);

    try std.testing.expectEqual(PermissionStatus.not_requested, checkPermission(&perms, .camera));

    _ = requestPermission(&perms, .camera);
    try std.testing.expectEqual(PermissionStatus.granted, checkPermission(&perms, .camera));

    revokePermission(&perms, .camera);
    try std.testing.expectEqual(PermissionStatus.denied, checkPermission(&perms, .camera));
}
