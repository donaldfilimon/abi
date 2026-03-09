const std = @import("std");
const builtin = @import("builtin");
const util = @import("util.zig");
const baseline = @import("baseline.zig");

/// cel_doctor.zig — Comprehensive .cel toolchain diagnostics.
///
/// Checks:
///   1. Platform detection (macOS version, whether CEL is needed)
///   2. .cel directory structure (config.sh, build.sh, patches/)
///   3. .cel/bin/zig and .cel/bin/zls binary presence and version
///   4. Patch inventory and validation
///   5. Version consistency (.zigversion, .cel/config.sh, baseline)
///   6. Stock zig status and PATH precedence
///   7. LLVM/cmake prerequisites for building
///   8. Actionable remediation steps
pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var issues: usize = 0;
    var warnings: usize = 0;

    std.debug.print("\n", .{});
    std.debug.print("=== ABI .cel Toolchain Doctor ===\n\n", .{});

    // ── 1. Platform ─────────────────────────────────────────────────────
    std.debug.print("Platform:\n", .{});
    if (builtin.os.tag == .macos) {
        const ver_res = util.captureCommand(allocator, io, "sw_vers -productVersion") catch null;
        if (ver_res) |res| {
            defer allocator.free(res.output);
            const ver = util.trimSpace(res.output);
            std.debug.print("  macOS version: {s}\n", .{ver});

            // Check if blocked
            if (builtin.os.version_range.semver.min.major >= 26) {
                std.debug.print("  Status: BLOCKED — macOS 26+ linker incompatibility\n", .{});
                std.debug.print("  CEL toolchain: REQUIRED for binary-emitting builds\n", .{});
            } else {
                std.debug.print("  Status: OK — stock Zig should work\n", .{});
                std.debug.print("  CEL toolchain: Optional enhancement\n", .{});
            }
        } else {
            std.debug.print("  macOS version: unknown\n", .{});
        }
    } else {
        std.debug.print("  OS: {s}\n", .{@tagName(builtin.os.tag)});
        std.debug.print("  CEL: Not needed (Darwin-specific toolchain fix)\n", .{});
    }
    std.debug.print("\n", .{});

    // ── 2. Directory structure ──────────────────────────────────────────
    std.debug.print(".cel directory structure:\n", .{});

    const cel_files = [_]struct { path: []const u8, label: []const u8, required: bool }{
        .{ .path = ".cel/config.sh", .label = "config.sh", .required = true },
        .{ .path = ".cel/build.sh", .label = "build.sh", .required = true },
        .{ .path = ".cel/README.md", .label = "README.md", .required = false },
        .{ .path = ".cel/patches", .label = "patches/", .required = true },
    };

    for (cel_files) |f| {
        if (f.required and f.path[f.path.len - 1] != '/') {
            if (util.fileExists(io, f.path)) {
                std.debug.print("  {s}: FOUND\n", .{f.label});
            } else {
                std.debug.print("  {s}: MISSING\n", .{f.label});
                issues += 1;
            }
        } else if (f.path[f.path.len - 1] == '/' or std.mem.endsWith(u8, f.path, "patches")) {
            if (util.dirExists(io, f.path)) {
                std.debug.print("  {s}: FOUND\n", .{f.label});
            } else {
                std.debug.print("  {s}: MISSING\n", .{f.label});
                if (f.required) issues += 1;
            }
        } else {
            if (util.fileExists(io, f.path)) {
                std.debug.print("  {s}: FOUND\n", .{f.label});
            } else {
                std.debug.print("  {s}: missing (optional)\n", .{f.label});
            }
        }
    }
    std.debug.print("\n", .{});

    // ── 3. CEL binary ──────────────────────────────────────────────────
    std.debug.print(".cel toolchain binaries:\n", .{});
    const cel_zig_exists = util.fileExists(io, ".cel/bin/zig");
    if (cel_zig_exists) {
        std.debug.print("  .cel/bin/zig: FOUND\n", .{});

        const ver_res = util.captureCommand(allocator, io, ".cel/bin/zig version") catch null;
        if (ver_res) |res| {
            defer allocator.free(res.output);
            const ver = util.trimSpace(res.output);
            std.debug.print("  Version: {s}\n", .{ver});

            // Check version consistency
            if (!std.mem.eql(u8, ver, baseline.zig_version)) {
                std.debug.print("  WARNING: CEL version ({s}) != baseline ({s})\n", .{ ver, baseline.zig_version });
                warnings += 1;
            } else {
                std.debug.print("  Version match: YES (matches baseline)\n", .{});
            }
        }
    } else {
        std.debug.print("  .cel/bin/zig: NOT BUILT\n", .{});
        if (builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26) {
            std.debug.print("  Action: Run .cel/build.sh to compile the patched toolchain\n", .{});
            issues += 1;
        }
    }
    const cel_zls_exists = util.fileExists(io, ".cel/bin/zls");
    if (cel_zls_exists) {
        std.debug.print("  .cel/bin/zls: FOUND\n", .{});
        const ver_res = util.captureCommand(allocator, io, ".cel/bin/zls --version") catch null;
        if (ver_res) |res| {
            defer allocator.free(res.output);
            const ver = util.trimSpace(res.output);
            if (ver.len > 0) std.debug.print("  ZLS version: {s}\n", .{ver});
        }
    } else {
        std.debug.print("  .cel/bin/zls: NOT BUILT\n", .{});
        if (cel_zig_exists) {
            std.debug.print("  Action: Run .cel/build.sh --zls-only to build ZLS with CEL Zig\n", .{});
            warnings += 1;
        }
    }
    std.debug.print("\n", .{});

    // ── 4. Patches ─────────────────────────────────────────────────────
    std.debug.print("Patches:\n", .{});
    if (util.dirExists(io, ".cel/patches")) {
        const ls_res = util.captureCommand(allocator, io, "ls -1 .cel/patches/*.patch 2>/dev/null || echo '(none)'") catch null;
        if (ls_res) |res| {
            defer allocator.free(res.output);
            var lines = std.mem.splitScalar(u8, res.output, '\n');
            var count: usize = 0;
            while (lines.next()) |line| {
                const trimmed = util.trimSpace(line);
                if (trimmed.len == 0) continue;
                if (std.mem.eql(u8, trimmed, "(none)")) continue;
                // Extract filename
                if (std.mem.lastIndexOfScalar(u8, trimmed, '/')) |idx| {
                    std.debug.print("  - {s}\n", .{trimmed[idx + 1 ..]});
                } else {
                    std.debug.print("  - {s}\n", .{trimmed});
                }
                count += 1;
            }
            std.debug.print("  Total: {d} patch(es)\n", .{count});
        }
    } else {
        std.debug.print("  No patches directory\n", .{});
    }
    std.debug.print("\n", .{});

    // ── 5. Version consistency ──────────────────────────────────────────
    std.debug.print("Version consistency:\n", .{});
    std.debug.print("  baseline.zig:  {s}\n", .{baseline.zig_version});

    const zigversion_raw = util.readFileAlloc(allocator, io, ".zigversion", 1024) catch null;
    if (zigversion_raw) |raw| {
        defer allocator.free(raw);
        const ver = util.trimSpace(raw);
        std.debug.print("  .zigversion:   {s}\n", .{ver});
        if (!std.mem.eql(u8, ver, baseline.zig_version)) {
            std.debug.print("  WARNING: .zigversion and baseline.zig disagree\n", .{});
            warnings += 1;
        }
    } else {
        std.debug.print("  .zigversion:   MISSING\n", .{});
        issues += 1;
    }

    // Check .cel/config.sh version
    const config_res = util.captureCommand(allocator, io,
        \\sh -c '. .cel/config.sh 2>/dev/null && printf "%s" "$ZIG_VERSION"'
    ) catch null;
    if (config_res) |res| {
        defer allocator.free(res.output);
        const cel_ver = util.trimSpace(res.output);
        if (cel_ver.len > 0) {
            std.debug.print("  .cel/config:   {s}\n", .{cel_ver});
            if (!std.mem.eql(u8, cel_ver, baseline.zig_version)) {
                std.debug.print("  WARNING: .cel/config.sh version != baseline\n", .{});
                warnings += 1;
            }
        }
    }
    std.debug.print("\n", .{});

    // ── 6. Stock zig and PATH ──────────────────────────────────────────
    std.debug.print("Stock Zig:\n", .{});
    if (try util.commandExists(allocator, io, "zig")) {
        const path_res = try util.captureCommand(allocator, io, "command -v zig");
        defer allocator.free(path_res.output);
        const zig_path = util.trimSpace(path_res.output);

        const ver_res = try util.captureCommand(allocator, io, "zig version");
        defer allocator.free(ver_res.output);
        const zig_ver = util.trimSpace(ver_res.output);

        std.debug.print("  Path:    {s}\n", .{zig_path});
        std.debug.print("  Version: {s}\n", .{zig_ver});

        // Check if CEL is on PATH
        if (std.mem.indexOf(u8, zig_path, ".cel/bin") != null) {
            std.debug.print("  Source:  .cel patched toolchain (good!)\n", .{});
        } else if (std.mem.indexOf(u8, zig_path, ".zvm") != null) {
            std.debug.print("  Source:  ZVM managed\n", .{});
        } else {
            std.debug.print("  Source:  System or other\n", .{});
        }
    } else {
        std.debug.print("  Not found on PATH\n", .{});
        issues += 1;
    }
    std.debug.print("\n", .{});

    // ── 7. Build prerequisites ─────────────────────────────────────────
    std.debug.print("Build prerequisites (for .cel/build.sh):\n", .{});
    const prereqs = [_][]const u8{ "git", "cmake", "cc", "c++" };
    for (prereqs) |cmd| {
        if (try util.commandExists(allocator, io, cmd)) {
            std.debug.print("  {s}: FOUND\n", .{cmd});
        } else {
            std.debug.print("  {s}: MISSING\n", .{cmd});
            warnings += 1;
        }
    }

    // Check for LLVM (needed for Zig compilation)
    if (util.dirExists(io, "zig-bootstrap-emergency/out/build-llvm-host")) {
        std.debug.print("  llvm: bootstrap LLVM artifacts\n", .{});
    } else if (util.dirExists(io, "/opt/homebrew/opt/llvm@21")) {
        std.debug.print("  llvm: Homebrew llvm@21 (/opt/homebrew/opt/llvm@21)\n", .{});
    } else if (util.dirExists(io, "/usr/local/opt/llvm@21")) {
        std.debug.print("  llvm: Homebrew llvm@21 (/usr/local/opt/llvm@21)\n", .{});
    } else if (try util.commandExists(allocator, io, "llvm-config")) {
        const llvm_res = util.captureCommand(allocator, io, "llvm-config --version") catch null;
        if (llvm_res) |res| {
            defer allocator.free(res.output);
            const ver = util.trimSpace(res.output);
            std.debug.print("  llvm: {s}\n", .{ver});
            if (!std.mem.startsWith(u8, ver, "21.")) {
                std.debug.print("  WARNING: current CEL pin expects LLVM 21.x\n", .{});
                warnings += 1;
            }
        }
    } else if (try util.commandExists(allocator, io, "brew")) {
        const brew_res = util.captureCommand(allocator, io, "brew --prefix llvm@21 2>/dev/null") catch null;
        if (brew_res) |res| {
            defer allocator.free(res.output);
            const prefix = util.trimSpace(res.output);
            if (prefix.len > 0 and util.dirExists(io, prefix)) {
                std.debug.print("  llvm: Homebrew llvm@21 ({s})\n", .{prefix});
            } else {
                std.debug.print("  llvm: Not installed (run: brew install llvm@21)\n", .{});
                warnings += 1;
            }
        }
    }
    std.debug.print("\n", .{});

    // ── Summary ────────────────────────────────────────────────────────
    std.debug.print("=== Summary ===\n", .{});
    if (issues == 0 and warnings == 0) {
        std.debug.print("OK: .cel toolchain doctor found no issues.\n", .{});
    } else {
        if (issues > 0) {
            std.debug.print("ISSUES: {d} issue(s) found\n", .{issues});
        }
        if (warnings > 0) {
            std.debug.print("WARNINGS: {d} warning(s) found\n", .{warnings});
        }
    }

    // ── Remediation ────────────────────────────────────────────────────
    if (issues > 0 and builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26) {
        std.debug.print("\nRemediation steps:\n", .{});
        if (!cel_zig_exists) {
            std.debug.print("  1. Build the CEL toolchain:\n", .{});
            std.debug.print("     ./.cel/build.sh\n\n", .{});
            std.debug.print("  2. Activate it:\n", .{});
            std.debug.print("     eval \"$(./tools/scripts/use_cel.sh)\"\n\n", .{});
            std.debug.print("  3. Verify:\n", .{});
            std.debug.print("     zig version\n", .{});
            std.debug.print("     zls --version\n", .{});
            std.debug.print("     zig build full-check\n\n", .{});
        }
    }

    std.debug.print("\n", .{});

    if (issues > 0) {
        std.process.exit(1);
    }
}
