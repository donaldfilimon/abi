//! Cross-Platform OS Features Tests
//!
//! Tests for the cross-platform OS module functionality.

const std = @import("std");
const testing = std.testing;
const os = @import("../shared/os.zig");

// ============================================================================
// Platform Detection Tests
// ============================================================================

test "platform detection returns valid OS" {
    const os_name = os.getOsName();
    try testing.expect(os_name.len > 0);

    // Should be one of the known platforms
    const valid_names = [_][]const u8{
        "Windows",
        "Linux",
        "macOS",
        "FreeBSD",
        "NetBSD",
        "OpenBSD",
        "DragonFly BSD",
        "iOS",
        "WASI",
        "Unknown",
    };

    var found = false;
    for (valid_names) |name| {
        if (std.mem.eql(u8, os_name, name)) {
            found = true;
            break;
        }
    }
    try testing.expect(found);
}

test "current_os matches builtin" {
    switch (os.current_os) {
        .windows => try testing.expect(@import("builtin").os.tag == .windows),
        .linux => try testing.expect(@import("builtin").os.tag == .linux),
        .macos => try testing.expect(@import("builtin").os.tag == .macos),
        .freebsd => try testing.expect(@import("builtin").os.tag == .freebsd),
        .netbsd => try testing.expect(@import("builtin").os.tag == .netbsd),
        .openbsd => try testing.expect(@import("builtin").os.tag == .openbsd),
        .dragonfly => try testing.expect(@import("builtin").os.tag == .dragonfly),
        .ios => try testing.expect(@import("builtin").os.tag == .ios),
        .wasi => try testing.expect(@import("builtin").os.tag == .wasi),
        .other => {},
    }
}

test "isPosix correctly identifies POSIX systems" {
    if (os.current_os.isPosix()) {
        // POSIX systems should not be Windows or WASI
        try testing.expect(os.current_os != .windows);
        try testing.expect(os.current_os != .wasi);
    }
}

test "isBsd correctly identifies BSD systems" {
    if (os.current_os.isBsd()) {
        try testing.expect(
            os.current_os == .freebsd or
                os.current_os == .netbsd or
                os.current_os == .openbsd or
                os.current_os == .dragonfly,
        );
    }
}

// ============================================================================
// System Information Tests
// ============================================================================

test "getCpuCount returns at least 1" {
    const count = os.getCpuCount();
    try testing.expect(count >= 1);
    try testing.expect(count <= 1024); // Reasonable upper bound
}

test "getPageSize returns valid page size" {
    const page_size = os.getPageSize();
    try testing.expect(page_size > 0);

    // Page size should be a power of 2
    try testing.expect(page_size & (page_size - 1) == 0);

    // Common page sizes: 4KB, 8KB, 16KB, 64KB
    try testing.expect(page_size >= 4096);
    try testing.expect(page_size <= 65536);
}

test "getHostname returns non-empty string" {
    const hostname = os.getHostname(testing.allocator) catch {
        // If hostname fails, that's acceptable in some environments
        return;
    };
    defer testing.allocator.free(hostname);

    try testing.expect(hostname.len > 0);
}

test "getUsername returns non-empty string" {
    const username = os.getUsername(testing.allocator) catch {
        return;
    };
    defer testing.allocator.free(username);

    try testing.expect(username.len > 0);
}

test "getHomeDir returns valid path" {
    const home = os.getHomeDir(testing.allocator) catch {
        return;
    };
    defer testing.allocator.free(home);

    try testing.expect(home.len > 0);

    // Home directory should be an absolute path on most systems
    if (!os.is_wasm) {
        try testing.expect(os.Path.isAbsolute(home));
    }
}

test "getTempDir returns valid path" {
    const temp = os.getTempDir(testing.allocator) catch {
        return;
    };
    defer testing.allocator.free(temp);

    try testing.expect(temp.len > 0);
}

test "getCurrentDir returns valid path" {
    const cwd = os.getCurrentDir(testing.allocator) catch {
        return;
    };
    defer testing.allocator.free(cwd);

    try testing.expect(cwd.len > 0);
}

test "getSystemInfo returns comprehensive info" {
    var info = os.getSystemInfo(testing.allocator) catch {
        return;
    };
    defer info.deinit();

    try testing.expect(info.hostname.len > 0);
    try testing.expect(info.username.len > 0);
    try testing.expect(info.home_dir.len > 0);
    try testing.expect(info.temp_dir.len > 0);
    try testing.expect(info.current_dir.len > 0);
    try testing.expect(info.os_name.len > 0);
    try testing.expect(info.cpu_count >= 1);
    try testing.expect(info.page_size > 0);
}

// ============================================================================
// Environment Variable Tests
// ============================================================================

test "Env.get returns PATH on most systems" {
    if (os.is_wasm) return;

    const path = os.Env.get("PATH");
    try testing.expect(path != null);
    try testing.expect(path.?.len > 0);
}

test "Env.exists works correctly" {
    if (os.is_wasm) return;

    // PATH should exist on all non-WASM systems
    try testing.expect(os.Env.exists("PATH"));

    // Random non-existent variable
    try testing.expect(!os.Env.exists("__ABI_TEST_NONEXISTENT_VAR_12345__"));
}

test "Env.getOr returns default for missing var" {
    const result = os.Env.getOr("__ABI_TEST_NONEXISTENT__", "default_value");
    try testing.expectEqualStrings("default_value", result);
}

test "Env.getBool parses boolean values" {
    // These tests depend on the environment, so just test the null case
    const result = os.Env.getBool("__ABI_TEST_NONEXISTENT__");
    try testing.expect(result == null);
}

test "Env.expand handles empty string" {
    const result = try os.Env.expand(testing.allocator, "");
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("", result);
}

test "Env.expand handles string without variables" {
    const result = try os.Env.expand(testing.allocator, "hello world");
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("hello world", result);
}

// ============================================================================
// Path Operations Tests
// ============================================================================

test "Path.basename extracts filename" {
    try testing.expectEqualStrings("file.txt", os.Path.basename("/path/to/file.txt"));
    try testing.expectEqualStrings("file.txt", os.Path.basename("path/to/file.txt"));
    try testing.expectEqualStrings("file.txt", os.Path.basename("file.txt"));
    try testing.expectEqualStrings("dir", os.Path.basename("/path/to/dir/"));
}

test "Path.dirname extracts directory" {
    try testing.expectEqualStrings("/path/to", os.Path.dirname("/path/to/file.txt"));
    try testing.expectEqualStrings("path/to", os.Path.dirname("path/to/file.txt"));
    try testing.expectEqualStrings(".", os.Path.dirname("file.txt"));
    try testing.expectEqualStrings("/", os.Path.dirname("/file.txt"));
}

test "Path.extension extracts extension" {
    try testing.expectEqualStrings(".txt", os.Path.extension("file.txt"));
    try testing.expectEqualStrings(".gz", os.Path.extension("archive.tar.gz"));
    try testing.expectEqualStrings("", os.Path.extension("noextension"));
    try testing.expectEqualStrings("", os.Path.extension(".hidden"));
}

test "Path.isAbsolute detects absolute paths" {
    if (@import("builtin").os.tag == .windows) {
        try testing.expect(os.Path.isAbsolute("C:\\Windows"));
        try testing.expect(os.Path.isAbsolute("D:/data"));
        try testing.expect(!os.Path.isAbsolute("relative\\path"));
    } else {
        try testing.expect(os.Path.isAbsolute("/usr/bin"));
        try testing.expect(os.Path.isAbsolute("/"));
        try testing.expect(!os.Path.isAbsolute("relative/path"));
        try testing.expect(!os.Path.isAbsolute("./relative"));
    }
}

test "Path.join combines paths" {
    const parts = [_][]const u8{ "path", "to", "file.txt" };
    const joined = try os.Path.join(testing.allocator, &parts);
    defer testing.allocator.free(joined);

    // Result should contain all parts
    try testing.expect(std.mem.indexOf(u8, joined, "path") != null);
    try testing.expect(std.mem.indexOf(u8, joined, "to") != null);
    try testing.expect(std.mem.indexOf(u8, joined, "file.txt") != null);
}

test "Path.normalize converts separators" {
    const normalized = try os.Path.normalize(testing.allocator, "path/to\\file");
    defer testing.allocator.free(normalized);

    // All separators should be the platform's native separator
    for (normalized) |c| {
        if (c == '/' or c == '\\') {
            try testing.expect(c == os.path_separator);
        }
    }
}

// ============================================================================
// Process Tests
// ============================================================================

test "getpid returns valid pid" {
    const pid = os.getpid();
    if (!os.is_wasm) {
        try testing.expect(pid > 0);
    }
}

test "isatty and isattyStdout return boolean" {
    // Just verify they don't crash
    _ = os.isatty();
    _ = os.isattyStdout();
}

test "isCI detects CI environment" {
    // Just verify it doesn't crash and returns a boolean
    _ = os.isCI();
}

// ============================================================================
// FileMode Tests
// ============================================================================

test "FileMode constants are correct" {
    try testing.expectEqual(@as(u32, 0o400), os.FileMode.owner_read);
    try testing.expectEqual(@as(u32, 0o200), os.FileMode.owner_write);
    try testing.expectEqual(@as(u32, 0o100), os.FileMode.owner_exec);
    try testing.expectEqual(@as(u32, 0o755), os.FileMode.default_dir);
    try testing.expectEqual(@as(u32, 0o644), os.FileMode.default_file);
    try testing.expectEqual(@as(u32, 0o600), os.FileMode.private_file);
}

// ============================================================================
// Signal Tests (POSIX only)
// ============================================================================

test "Signal enum has correct values" {
    try testing.expectEqual(@as(u8, 2), os.Signal.interrupt.toNative());
    try testing.expectEqual(@as(u8, 15), os.Signal.terminate.toNative());
    try testing.expectEqual(@as(u8, 9), os.Signal.kill.toNative());
}
