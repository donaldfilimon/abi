//! macOS OS-keychain credential storage backend (opt-in, additive to the
//! file-based store in `credentials.zig`).
//!
//! On macOS this talks directly to Security.framework's SecItem C API via
//! FFI (`SecItemAdd` / `SecItemCopyMatching` / `SecItemUpdate` /
//! `SecItemDelete`) — it never shells out to `/usr/bin/security`, since a
//! secret passed on argv would be visible to other users via `ps` on a
//! multi-user box.
//!
//! On every other target, all three entry points below return
//! `error.KeychainUnsupported`: an explicit disclosed gap, never a silent
//! no-op success. Windows Credential Manager and Linux Secret Service remain
//! Proposed — not implemented in this module.
//!
//! Scope note: this stores secrets in the macOS **login keychain** with the
//! OS's default at-rest protection. It is NOT hardware-backed (no Secure
//! Enclave / biometric gating) and has NOT been independently audited.

const std = @import("std");
const builtin = @import("builtin");

/// Store `secret` under (`service`, `account`) in the OS keychain. Overwrites
/// any existing item for the same (service, account) pair. Returns
/// `error.KeychainUnsupported` on non-macOS targets.
pub fn keychainStore(service: []const u8, account: []const u8, secret: []const u8) !void {
    return macos_keychain.store(service, account, secret);
}

/// Load the secret stored under (`service`, `account`). Returns `null` when
/// no such item exists (this is not an error). Returns
/// `error.KeychainUnsupported` on non-macOS targets. Caller owns the
/// returned slice and must free it with `allocator`.
pub fn keychainLoad(allocator: std.mem.Allocator, service: []const u8, account: []const u8) !?[]u8 {
    return macos_keychain.load(allocator, service, account);
}

/// Delete the item stored under (`service`, `account`). A missing item is
/// treated as success (idempotent). Returns `error.KeychainUnsupported` on
/// non-macOS targets.
pub fn keychainDelete(service: []const u8, account: []const u8) !void {
    return macos_keychain.delete(service, account);
}

const macos_keychain = if (builtin.target.os.tag == .macos) struct {
    // -- CoreFoundation / Security C ABI --
    //
    // Every CF/Sec handle is toll-free-bridged to a plain opaque pointer, so a
    // single alias covers CFStringRef/CFDataRef/CFDictionaryRef/CFTypeRef.
    // Framework linking (`Security` + `CoreFoundation`) happens in build.zig;
    // symbols below resolve against those without needing a library tag on
    // an extern declaration (matches the existing `objc_msgSend` convention
    // in src/features/gpu/metal_shared.zig).
    const CFRef = ?*const anyopaque;
    const CFIndex = isize;
    const OSStatus = i32;

    const errSecSuccess: OSStatus = 0;
    const errSecItemNotFound: OSStatus = -25300;
    const errSecDuplicateItem: OSStatus = -25299;

    const kCFStringEncodingUTF8: u32 = 0x08000100;

    extern fn CFDictionaryCreate(
        allocator: CFRef,
        keys: [*]const CFRef,
        values: [*]const CFRef,
        numValues: CFIndex,
        keyCallBacks: CFRef,
        valueCallBacks: CFRef,
    ) callconv(.c) CFRef;

    extern fn CFDataCreate(allocator: CFRef, bytes: [*]const u8, length: CFIndex) callconv(.c) CFRef;
    extern fn CFDataGetBytePtr(data: CFRef) callconv(.c) [*]const u8;
    extern fn CFDataGetLength(data: CFRef) callconv(.c) CFIndex;
    extern fn CFRelease(cf: CFRef) callconv(.c) void;
    extern fn CFStringCreateWithBytes(
        allocator: CFRef,
        bytes: [*]const u8,
        numBytes: CFIndex,
        encoding: u32,
        isExternalRepresentation: c_int,
    ) callconv(.c) CFRef;

    extern const kCFBooleanTrue: CFRef;

    extern const kSecClass: CFRef;
    extern const kSecClassGenericPassword: CFRef;
    extern const kSecAttrService: CFRef;
    extern const kSecAttrAccount: CFRef;
    extern const kSecValueData: CFRef;
    extern const kSecReturnData: CFRef;
    extern const kSecMatchLimit: CFRef;
    extern const kSecMatchLimitOne: CFRef;

    extern fn SecItemAdd(query: CFRef, result: ?*CFRef) callconv(.c) OSStatus;
    extern fn SecItemCopyMatching(query: CFRef, result: ?*CFRef) callconv(.c) OSStatus;
    extern fn SecItemUpdate(query: CFRef, attributesToUpdate: CFRef) callconv(.c) OSStatus;
    extern fn SecItemDelete(query: CFRef) callconv(.c) OSStatus;

    fn makeCFString(bytes: []const u8) !*const anyopaque {
        const s = CFStringCreateWithBytes(null, bytes.ptr, @intCast(bytes.len), kCFStringEncodingUTF8, 0);
        return s orelse error.KeychainStoreFailed;
    }

    fn makeQuery(keys: []const CFRef, values: []const CFRef) !*const anyopaque {
        std.debug.assert(keys.len == values.len);
        // NULL key/value callbacks: our keys/values are either static Security
        // constants (stable process-lifetime addresses) or CFStrings/CFData we
        // create and keep alive for the call's duration ourselves, so we do
        // not need CFDictionaryCreate to retain them or hash beyond pointer
        // identity.
        const dict = CFDictionaryCreate(null, keys.ptr, values.ptr, @intCast(keys.len), null, null);
        return dict orelse error.KeychainStoreFailed;
    }

    fn store(service: []const u8, account: []const u8, secret: []const u8) !void {
        const service_cf = try makeCFString(service);
        defer CFRelease(service_cf);
        const account_cf = try makeCFString(account);
        defer CFRelease(account_cf);
        const data_cf = CFDataCreate(null, secret.ptr, @intCast(secret.len)) orelse return error.KeychainStoreFailed;
        defer CFRelease(data_cf);

        // Try add first; a pre-existing item falls through to an update below,
        // so callers never have to branch on "first write vs. rotate".
        {
            const keys = [_]CFRef{ kSecClass, kSecAttrService, kSecAttrAccount, kSecValueData };
            const values = [_]CFRef{ kSecClassGenericPassword, service_cf, account_cf, data_cf };
            const query = try makeQuery(&keys, &values);
            defer CFRelease(query);
            const status = SecItemAdd(query, null);
            if (status == errSecSuccess) return;
            if (status != errSecDuplicateItem) return error.KeychainStoreFailed;
        }

        // Duplicate: update the existing item's secret data in place.
        {
            const match_keys = [_]CFRef{ kSecClass, kSecAttrService, kSecAttrAccount };
            const match_values = [_]CFRef{ kSecClassGenericPassword, service_cf, account_cf };
            const match_query = try makeQuery(&match_keys, &match_values);
            defer CFRelease(match_query);

            const update_keys = [_]CFRef{kSecValueData};
            const update_values = [_]CFRef{data_cf};
            const attrs = try makeQuery(&update_keys, &update_values);
            defer CFRelease(attrs);

            const status = SecItemUpdate(match_query, attrs);
            if (status != errSecSuccess) return error.KeychainStoreFailed;
        }
    }

    fn load(allocator: std.mem.Allocator, service: []const u8, account: []const u8) !?[]u8 {
        const service_cf = try makeCFString(service);
        defer CFRelease(service_cf);
        const account_cf = try makeCFString(account);
        defer CFRelease(account_cf);

        const keys = [_]CFRef{ kSecClass, kSecAttrService, kSecAttrAccount, kSecReturnData, kSecMatchLimit };
        const values = [_]CFRef{ kSecClassGenericPassword, service_cf, account_cf, kCFBooleanTrue, kSecMatchLimitOne };
        const query = try makeQuery(&keys, &values);
        defer CFRelease(query);

        var result: CFRef = null;
        const status = SecItemCopyMatching(query, &result);
        if (status == errSecItemNotFound) return null;
        if (status != errSecSuccess) return error.KeychainLoadFailed;
        const data_cf = result orelse return null;
        defer CFRelease(data_cf);

        const len: usize = @intCast(CFDataGetLength(data_cf));
        const ptr = CFDataGetBytePtr(data_cf);
        return try allocator.dupe(u8, ptr[0..len]);
    }

    fn delete(service: []const u8, account: []const u8) !void {
        const service_cf = try makeCFString(service);
        defer CFRelease(service_cf);
        const account_cf = try makeCFString(account);
        defer CFRelease(account_cf);

        const keys = [_]CFRef{ kSecClass, kSecAttrService, kSecAttrAccount };
        const values = [_]CFRef{ kSecClassGenericPassword, service_cf, account_cf };
        const query = try makeQuery(&keys, &values);
        defer CFRelease(query);

        const status = SecItemDelete(query);
        if (status != errSecSuccess and status != errSecItemNotFound) return error.KeychainDeleteFailed;
    }
} else struct {
    fn store(service: []const u8, account: []const u8, secret: []const u8) !void {
        _ = service;
        _ = account;
        _ = secret;
        return error.KeychainUnsupported;
    }

    fn load(allocator: std.mem.Allocator, service: []const u8, account: []const u8) !?[]u8 {
        _ = allocator;
        _ = service;
        _ = account;
        return error.KeychainUnsupported;
    }

    fn delete(service: []const u8, account: []const u8) !void {
        _ = service;
        _ = account;
        return error.KeychainUnsupported;
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "macOS keychain store/load/update/delete round trip (opt-in, hits real keychain)" {
    // This test performs real SecItemAdd/SecItemCopyMatching/SecItemUpdate/
    // SecItemDelete calls against the login keychain. Unsigned dev binaries
    // get a new code identity on every rebuild, which can trigger a keychain
    // trust-store prompt on an interactive session — not safe to run
    // unconditionally in `./build.sh check`. Opt in explicitly.
    //
    // `zig build test` never calls `foundation/env.zig`'s `install()` (only
    // `main`/`mcp/main.zig` do), so `env.get` always reports "unset" inside
    // any test binary regardless of the real process environment. This gate
    // needs the *actual* invoking shell's env var, so it reads libc `getenv`
    // directly — a narrow, test-only, macOS-only exception to the
    // portable-env-access convention (libc is already linked on this
    // target).
    if (comptime builtin.target.os.tag != .macos) return error.SkipZigTest;
    if (std.c.getenv("ABI_KEYCHAIN_TEST") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const service = "abi-credentials-test";
    var account_buf: [64]u8 = undefined;
    const account = try std.fmt.bufPrint(&account_buf, "abi-test-account-{d}", .{std.c.getpid()});

    defer keychainDelete(service, account) catch |err| {
        std.log.warn("keychain test cleanup failed: {s}", .{@errorName(err)});
    };

    try keychainStore(service, account, "test-secret-value");

    const loaded = try keychainLoad(allocator, service, account);
    defer if (loaded) |v| allocator.free(v);
    try std.testing.expectEqualStrings("test-secret-value", loaded orelse return error.MissingSecret);

    // SecItemAdd hit errSecDuplicateItem here and fell through to
    // SecItemUpdate; confirm the update actually took.
    try keychainStore(service, account, "test-secret-value-2");
    const loaded2 = try keychainLoad(allocator, service, account);
    defer if (loaded2) |v| allocator.free(v);
    try std.testing.expectEqualStrings("test-secret-value-2", loaded2 orelse return error.MissingSecretAfterUpdate);

    try keychainDelete(service, account);
    const loaded3 = try keychainLoad(allocator, service, account);
    try std.testing.expect(loaded3 == null);

    // Deleting an already-absent item is idempotent, not an error.
    try keychainDelete(service, account);
}
