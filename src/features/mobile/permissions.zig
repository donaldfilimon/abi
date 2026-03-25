//! Permission management for the mobile feature.
//!
//! Provides check, request, and revoke operations for mobile permissions
//! (camera, microphone, location, etc.) using a simulated permission store.

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
