const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const download = @import("download.zig");
const archive = @import("archive.zig");

const ToolchainSource = enum {
    zvm,
    zigly,
};

const ToolchainSelection = enum {
    zvm_active,
    zigly_cache,
    install_via_zvm,
    install_via_zigly,
};

const ResolvedToolchain = struct {
    source: ToolchainSource,
    zig_path: []u8,
    zls_path: ?[]u8,

    pub fn deinit(self: ResolvedToolchain, allocator: std.mem.Allocator) void {
        allocator.free(self.zig_path);
        if (self.zls_path) |zls_path| allocator.free(zls_path);
    }
};

pub fn resolveVersion(config: *core.Config, raw_version: []const u8) ![]const u8 {
    if (raw_version.len > 0) {
        return raw_version;
    }

    if (config.project_version) |pv| {
        return pv;
    }

    std.debug.print("ERROR: No version provided and no .zigversion file found in current directory.\n", .{});
    return error.NoVersion;
}

pub fn getOsArch() struct { os: []const u8, arch: []const u8 } {
    const os_tag = switch (builtin.os.tag) {
        .macos => "macos",
        .linux => "linux",
        else => "unknown",
    };

    const arch_tag = switch (builtin.cpu.arch) {
        .aarch64 => "aarch64",
        .x86_64 => "x86_64",
        .x86 => "x86",
        else => "unknown",
    };

    return .{ .os = os_tag, .arch = arch_tag };
}

fn versionMatches(requested: []const u8, actual: ?[]const u8) bool {
    const actual_version = actual orelse return false;
    return std.mem.eql(u8, requested, actual_version);
}

fn selectToolchain(
    requested: []const u8,
    has_zvm: bool,
    active_zvm_version: ?[]const u8,
    has_zigly_cache: bool,
) ToolchainSelection {
    if (has_zvm) {
        if (versionMatches(requested, active_zvm_version)) return .zvm_active;
        return .install_via_zvm;
    }
    if (has_zigly_cache) return .zigly_cache;
    return .install_via_zigly;
}

fn fileExistsAbsolute(io: std.Io, path: []const u8) bool {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn allocZiglyCacheDir(config: *core.Config, version: []const u8) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions", version });
}

fn allocZiglyZigPath(config: *core.Config, version: []const u8) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions", version, "bin", "zig" });
}

fn allocZiglyZlsPath(config: *core.Config, version: []const u8) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions", version, "bin", "zls" });
}

fn allocZvmSelfPath(config: *core.Config) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".zvm", "self", "zvm" });
}

fn allocZvmBinZigPath(config: *core.Config) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".zvm", "bin", "zig" });
}

fn allocZvmBinZlsPath(config: *core.Config) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".zvm", "bin", "zls" });
}

fn allocZvmMasterZigPath(config: *core.Config) ![]u8 {
    return try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".zvm", "master", "zig" });
}

fn hasZvm(config: *core.Config) !bool {
    const zvm_path = try allocZvmSelfPath(config);
    defer config.allocator.free(zvm_path);
    return fileExistsAbsolute(config.io, zvm_path);
}

fn probeBinaryVersion(config: *core.Config, binary_path: []const u8) !?[]u8 {
    if (!fileExistsAbsolute(config.io, binary_path)) return null;

    const output_path = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "version.txt" });
    defer config.allocator.free(output_path);

    var child_args = std.ArrayListUnmanaged([]const u8).empty;
    defer child_args.deinit(config.allocator);
    try child_args.append(config.allocator, "sh");
    try child_args.append(config.allocator, "-c");
    try child_args.append(config.allocator, "\"$1\" version > \"$2\"");
    try child_args.append(config.allocator, "sh");
    try child_args.append(config.allocator, binary_path);
    try child_args.append(config.allocator, output_path);

    var child = try std.process.spawn(config.io, .{ .argv = child_args.items });
    const term = try child.wait(config.io);
    switch (term) {
        .exited => |code| {
            if (code != 0) return null;
        },
        else => return null,
    }

    const file = std.Io.Dir.openFileAbsolute(config.io, output_path, .{}) catch return null;
    defer file.close(config.io);
    const stat = file.stat(config.io) catch return null;
    if (stat.size == 0) return null;

    const content = try config.allocator.alloc(u8, @min(stat.size, 128));
    const bytes_read = try file.readPositionalAll(config.io, content, 0);
    const trimmed = std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
    if (trimmed.len == 0) {
        config.allocator.free(content);
        return null;
    }
    const version = try config.allocator.dupe(u8, trimmed);
    config.allocator.free(content);
    return version;
}

fn tryRunCommand(config: *core.Config, argv: []const []const u8) !bool {
    var child = try std.process.spawn(config.io, .{ .argv = argv });
    const term = try child.wait(config.io);
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn resolveActiveZvmToolchain(config: *core.Config, version: []const u8) !?ResolvedToolchain {
    const zvm_zig = try allocZvmBinZigPath(config);
    const active_version = try probeBinaryVersion(config, zvm_zig);
    defer if (active_version) |v| config.allocator.free(v);

    if (!versionMatches(version, active_version)) {
        config.allocator.free(zvm_zig);
        return null;
    }

    const zvm_zls = try allocZvmBinZlsPath(config);
    if (!fileExistsAbsolute(config.io, zvm_zls)) {
        config.allocator.free(zvm_zls);
        return ResolvedToolchain{
            .source = .zvm,
            .zig_path = zvm_zig,
            .zls_path = null,
        };
    }

    return ResolvedToolchain{
        .source = .zvm,
        .zig_path = zvm_zig,
        .zls_path = zvm_zls,
    };
}

fn resolveZiglyCacheToolchain(config: *core.Config, version: []const u8) !?ResolvedToolchain {
    const zig_path = try allocZiglyZigPath(config, version);
    if (!fileExistsAbsolute(config.io, zig_path)) {
        config.allocator.free(zig_path);
        return null;
    }

    const zls_path = try allocZiglyZlsPath(config, version);
    if (!fileExistsAbsolute(config.io, zls_path)) {
        config.allocator.free(zls_path);
        return ResolvedToolchain{
            .source = .zigly,
            .zig_path = zig_path,
            .zls_path = null,
        };
    }

    return ResolvedToolchain{
        .source = .zigly,
        .zig_path = zig_path,
        .zls_path = zls_path,
    };
}

fn syncViaZvm(config: *core.Config, version: []const u8) !ResolvedToolchain {
    const zvm_path = try allocZvmSelfPath(config);
    defer config.allocator.free(zvm_path);

    std.debug.print("==> ZVM detected. Activating Zig {s} through ZVM...\n", .{version});

    // Check ZVM master first — this is the primary source for dev builds
    const master_zig = try allocZvmMasterZigPath(config);
    defer config.allocator.free(master_zig);
    const master_version = try probeBinaryVersion(config, master_zig);
    defer if (master_version) |v| config.allocator.free(v);

    if (versionMatches(version, master_version)) {
        const use_master = [_][]const u8{ zvm_path, "use", "master" };
        if (try tryRunCommand(config, &use_master)) {
            if (try resolveActiveZvmToolchain(config, version)) |toolchain| return toolchain;
            return error.ZvmResolutionFailed;
        }
        std.debug.print("ERROR: ZVM has master matching version {s}, but 'zvm use master' failed.\n", .{version});
        return error.ZvmUseFailed;
    }

    // Fall back to snapshot install if master doesn't match
    const install_exact = [_][]const u8{ zvm_path, "install", "--zls", "--nomirror", version };
    const installed_exact = try tryRunCommand(config, &install_exact);

    if (installed_exact) {
        const sync = [_][]const u8{ zvm_path, "use", "--sync" };
        _ = try tryRunCommand(config, &sync);
        if (try resolveActiveZvmToolchain(config, version)) |toolchain| return toolchain;
    }

    // Try installing master as a last resort
    const install_master = [_][]const u8{ zvm_path, "install", "--zls", "--nomirror", "master" };
    _ = try tryRunCommand(config, &install_master);

    const master_version_after = try probeBinaryVersion(config, master_zig);
    defer if (master_version_after) |v| config.allocator.free(v);

    if (versionMatches(version, master_version_after)) {
        const use_master = [_][]const u8{ zvm_path, "use", "master" };
        if (try tryRunCommand(config, &use_master)) {
            if (try resolveActiveZvmToolchain(config, version)) |toolchain| return toolchain;
            return error.ZvmResolutionFailed;
        }
    }

    std.debug.print("ERROR: ZVM is installed but could not activate Zig {s}.\n", .{version});
    std.debug.print("       Tried master, snapshot install, and master fallback.\n", .{});
    return error.UnsupportedZigVersion;
}

fn ensureToolchain(config: *core.Config, version: []const u8) !ResolvedToolchain {
    const zvm_installed = try hasZvm(config);
    const active_zvm_path = if (zvm_installed) try allocZvmBinZigPath(config) else null;
    defer if (active_zvm_path) |path| config.allocator.free(path);
    const active_zvm_version = if (active_zvm_path) |path| try probeBinaryVersion(config, path) else null;
    defer if (active_zvm_version) |v| config.allocator.free(v);

    const zigly_cache_path = try allocZiglyZigPath(config, version);
    const has_zigly_cache = fileExistsAbsolute(config.io, zigly_cache_path);
    config.allocator.free(zigly_cache_path);

    switch (selectToolchain(version, zvm_installed, active_zvm_version, has_zigly_cache)) {
        .zvm_active => {
            if (try resolveActiveZvmToolchain(config, version)) |toolchain| return toolchain;
            return error.ZvmResolutionFailed;
        },
        .zigly_cache => {
            if (try resolveZiglyCacheToolchain(config, version)) |toolchain| return toolchain;
            return error.ZiglyResolutionFailed;
        },
        .install_via_zvm => return syncViaZvm(config, version),
        .install_via_zigly => {
            try installViaZiglyCache(config, version);
            if (try resolveZiglyCacheToolchain(config, version)) |toolchain| return toolchain;
            return error.ZiglyResolutionFailed;
        },
    }
}

pub fn doStatus(config: *core.Config, raw_version: []const u8) !void {
    const version = try resolveVersion(config, raw_version);
    const toolchain = try ensureToolchain(config, version);
    defer toolchain.deinit(config.allocator);

    const out_str = std.fmt.allocPrint(config.allocator, "{s}\n", .{toolchain.zig_path}) catch return;
    defer config.allocator.free(out_str);
    _ = std.Io.File.stdout().writeStreamingAll(config.io, out_str) catch {};
}

fn installViaZiglyCache(config: *core.Config, version: []const u8) !void {
    const cache_dir = try allocZiglyCacheDir(config, version);
    defer config.allocator.free(cache_dir);

    const zig_bin = try allocZiglyZigPath(config, version);
    defer config.allocator.free(zig_bin);

    const zig_bin_exists = fileExistsAbsolute(config.io, zig_bin);
    if (zig_bin_exists) {
        std.debug.print("==> Zig {s} is already installed.\n", .{version});
        return;
    }

    std.debug.print("==> Installing Zig {s}...\n", .{version});

    const os_arch = getOsArch();
    if (std.mem.eql(u8, os_arch.os, "unknown") or std.mem.eql(u8, os_arch.arch, "unknown")) {
        std.debug.print("ERROR: Unsupported OS or Architecture.\n", .{});
        return error.UnsupportedPlatform;
    }

    const url = try std.fmt.allocPrint(config.allocator, "https://ziglang.org/builds/zig-{s}-{s}-{s}.tar.xz", .{ os_arch.arch, os_arch.os, version });
    defer config.allocator.free(url);

    const tmp_tarball = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "zig.tar.xz" });
    defer config.allocator.free(tmp_tarball);

    const extract_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "extract" });
    defer config.allocator.free(extract_dir);

    _ = std.Io.Dir.cwd().deleteTree(config.io, extract_dir) catch {};
    try std.Io.Dir.cwd().createDirPath(config.io, extract_dir);

    try download.downloadFile(config.allocator, config.io, url, tmp_tarball);

    // Verify SHA256 checksum against the download index (best-effort)
    if (fetchExpectedShaSum(config, version, os_arch.arch, os_arch.os)) |expected_hash| {
        std.debug.print("==> Verifying checksum...\n", .{});
        if (computeFileSha256(config, tmp_tarball)) |actual_hash| {
            if (!std.mem.eql(u8, expected_hash, actual_hash)) {
                std.debug.print("ERROR: Checksum mismatch!\n  expected: {s}\n  got:      {s}\n", .{ expected_hash, actual_hash });
                return error.ChecksumMismatch;
            }
            std.debug.print("==> Checksum OK.\n", .{});
        } else |_| {
            std.debug.print("==> WARNING: Could not compute checksum (shasum not available?). Skipping.\n", .{});
        }
    } else |_| {
        std.debug.print("==> WARNING: Could not fetch expected checksum. Skipping verification.\n", .{});
    }

    std.debug.print("==> Extracting...\n", .{});
    try archive.extractTarball(config.allocator, config.io, tmp_tarball, extract_dir);

    // Find the inner directory
    var dir = try std.Io.Dir.openDirAbsolute(config.io, extract_dir, .{ .iterate = true });
    defer dir.close(config.io);

    var it = dir.iterate();
    var inner_dirname: ?[]const u8 = null;
    while (try it.next(config.io)) |entry| {
        if (entry.kind == .directory and std.mem.startsWith(u8, entry.name, "zig-")) {
            inner_dirname = try config.allocator.dupe(u8, entry.name);
            break;
        }
    }

    if (inner_dirname) |name| {
        defer config.allocator.free(name);

        try std.Io.Dir.cwd().createDirPath(config.io, cache_dir);

        // Move files from inner_dirname to cache_dir
        const inner_path = try std.fs.path.join(config.allocator, &[_][]const u8{ extract_dir, name });
        defer config.allocator.free(inner_path);

        // We'll rename the whole directory to cache_dir if cache_dir is empty,
        // or just move its contents. Since we just created cache_dir, it's safer to delete it and rename.
        try std.Io.Dir.cwd().deleteTree(config.io, cache_dir);
        try std.Io.Dir.renameAbsolute(inner_path, cache_dir, config.io);

        std.debug.print("==> Zig {s} installed to {s}\n", .{ version, cache_dir });
    } else {
        std.debug.print("ERROR: Unexpected archive structure.\n", .{});
        return error.UnexpectedArchive;
    }

    // Attempt ZLS download (best effort, don't fail installation if it fails)
    try downloadZls(config, version);
}

pub fn doInstall(config: *core.Config, raw_version: []const u8) !void {
    const version = try resolveVersion(config, raw_version);
    const toolchain = try ensureToolchain(config, version);
    defer toolchain.deinit(config.allocator);

    switch (toolchain.source) {
        .zvm => std.debug.print("==> Zig {s} is active via ZVM.\n", .{version}),
        .zigly => std.debug.print("==> Zig {s} is ready in the zigly cache.\n", .{version}),
    }
}

pub fn downloadZls(config: *core.Config, version: []const u8) !void {
    const cache_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions", version });
    defer config.allocator.free(cache_dir);

    const zls_bin = try std.fs.path.join(config.allocator, &[_][]const u8{ cache_dir, "bin", "zls" });
    defer config.allocator.free(zls_bin);

    const zls_bin_exists = if (std.Io.Dir.accessAbsolute(config.io, zls_bin, .{})) true else |_| false;
    if (zls_bin_exists) {
        std.debug.print("==> ZLS is already installed for this version.\n", .{});
        return;
    }

    std.debug.print("==> Installing ZLS for Zig {s}...\n", .{version});

    const os_arch = getOsArch();
    if (std.mem.eql(u8, os_arch.os, "unknown") or std.mem.eql(u8, os_arch.arch, "unknown")) {
        std.debug.print("==> WARNING: Unsupported OS or architecture for ZLS download; continuing without ZLS.\n", .{});
        return;
    }

    const tmp_tarball = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "zls.tar.xz" });
    defer config.allocator.free(tmp_tarball);
    const urls = [_][]const u8{
        try std.fmt.allocPrint(config.allocator, "https://builds.zigtools.org/zls-{s}-{s}-{s}.tar.xz", .{ os_arch.arch, os_arch.os, version }),
        try std.fmt.allocPrint(config.allocator, "https://github.com/zigtools/zls/releases/download/{s}/zls-{s}-{s}.tar.xz", .{ version, os_arch.os, os_arch.arch }),
    };
    defer config.allocator.free(urls[0]);
    defer config.allocator.free(urls[1]);

    var downloaded_exact = false;
    for (urls) |url| {
        if (download.downloadFile(config.allocator, config.io, url, tmp_tarball)) |_| {
            downloaded_exact = true;
            break;
        } else |_| {}
    }

    if (!downloaded_exact) {
        const zvm_zls = try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".zvm", "bin", "zls" });
        defer config.allocator.free(zvm_zls);

        const zvm_zls_exists = if (std.Io.Dir.accessAbsolute(config.io, zvm_zls, .{})) true else |_| false;
        if (zvm_zls_exists) {
            std.debug.print("==> No exact pre-built ZLS found for Zig {s}; using the existing ZVM ZLS.\n", .{version});
            const cache_bin_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ cache_dir, "bin" });
            defer config.allocator.free(cache_bin_dir);
            try std.Io.Dir.cwd().createDirPath(config.io, cache_bin_dir);
            try std.Io.Dir.copyFileAbsolute(zvm_zls, zls_bin, config.io, .{});

            var child_args = std.ArrayListUnmanaged([]const u8).empty;
            defer child_args.deinit(config.allocator);
            try child_args.append(config.allocator, "chmod");
            try child_args.append(config.allocator, "+x");
            try child_args.append(config.allocator, zls_bin);
            var child = try std.process.spawn(config.io, .{ .argv = child_args.items });
            _ = try child.wait(config.io);
            return;
        }

        std.debug.print("==> WARNING: No exact pre-built ZLS found for Zig {s}; continuing without ZLS.\n", .{version});
        return;
    }

    const extract_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "zls_extract" });
    defer config.allocator.free(extract_dir);

    _ = std.Io.Dir.cwd().deleteTree(config.io, extract_dir) catch {};
    try std.Io.Dir.cwd().createDirPath(config.io, extract_dir);

    std.debug.print("==> Extracting ZLS {s}...\n", .{version});
    try archive.extractTarball(config.allocator, config.io, tmp_tarball, extract_dir);

    // Find the zls binary
    const extracted_zls = try std.fs.path.join(config.allocator, &[_][]const u8{ extract_dir, "zls" });
    defer config.allocator.free(extracted_zls);

    const extracted_zls_exists = if (std.Io.Dir.accessAbsolute(config.io, extracted_zls, .{})) true else |_| false;
    if (extracted_zls_exists) {
        const cache_bin_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ cache_dir, "bin" });
        defer config.allocator.free(cache_bin_dir);
        try std.Io.Dir.cwd().createDirPath(config.io, cache_bin_dir);

        try std.Io.Dir.copyFileAbsolute(extracted_zls, zls_bin, config.io, .{});

        var child_args = std.ArrayListUnmanaged([]const u8).empty;
        try child_args.append(config.allocator, "chmod");
        try child_args.append(config.allocator, "+x");
        try child_args.append(config.allocator, zls_bin);
        var child = try std.process.spawn(config.io, .{ .argv = child_args.items });
        _ = try child.wait(config.io);

        std.debug.print("==> ZLS installed.\n", .{});
    } else {
        std.debug.print("==> WARNING: ZLS binary not found in the downloaded archive.\n", .{});
    }
}

pub fn doUse(config: *core.Config, raw_version: []const u8) !void {
    const version = try resolveVersion(config, raw_version);
    const toolchain = try ensureToolchain(config, version);
    defer toolchain.deinit(config.allocator);

    const local_bin = try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".local", "bin" });
    defer config.allocator.free(local_bin);

    try std.Io.Dir.cwd().createDirPath(config.io, local_bin);

    std.debug.print("==> Setting Zig {s} as default in {s}...\n", .{ version, local_bin });

    const local_zig = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zig" });
    defer config.allocator.free(local_zig);

    _ = std.Io.Dir.deleteFileAbsolute(config.io, local_zig) catch {};
    try std.Io.Dir.symLinkAbsolute(config.io, toolchain.zig_path, local_zig, .{});
    std.debug.print("==> Symlinked zig -> {s}\n", .{local_zig});

    if (toolchain.zls_path) |zls_path| {
        const local_zls = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zls" });
        defer config.allocator.free(local_zls);
        _ = std.Io.Dir.deleteFileAbsolute(config.io, local_zls) catch {};
        try std.Io.Dir.symLinkAbsolute(config.io, zls_path, local_zls, .{});
        std.debug.print("==> Symlinked zls -> {s}\n", .{local_zls});
    }

    const default_file = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "default" });
    defer config.allocator.free(default_file);
    const df = try std.Io.Dir.createFileAbsolute(config.io, default_file, .{});
    defer df.close(config.io);
    _ = try df.writePositionalAll(config.io, version, 0);
}

pub fn doList(config: *core.Config) !void {
    std.debug.print("==> Installed versions:\n", .{});

    const default_file = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "default" });
    defer config.allocator.free(default_file);

    var active_version: ?[]const u8 = null;
    if (std.Io.Dir.openFileAbsolute(config.io, default_file, .{})) |df| {
        defer df.close(config.io);
        if (df.stat(config.io)) |stat| {
            if (config.allocator.alloc(u8, stat.size)) |content| {
                if (df.readPositionalAll(config.io, content, 0)) |bytes_read| {
                    active_version = std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
                } else |_| {}
            } else |_| {}
        } else |_| {}
    } else |_| {}

    const versions_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions" });
    defer config.allocator.free(versions_dir);

    var dir = std.Io.Dir.openDirAbsolute(config.io, versions_dir, .{ .iterate = true }) catch return;
    defer dir.close(config.io);

    var it = dir.iterate();
    var count: usize = 0;
    while (try it.next(config.io)) |entry| {
        if (entry.kind == .directory) {
            count += 1;
            const is_active = if (active_version) |av| std.mem.eql(u8, entry.name, av) else false;
            if (is_active) {
                std.debug.print("  * {s} (active)\n", .{entry.name});
            } else {
                std.debug.print("    {s}\n", .{entry.name});
            }
        }
    }

    if (count == 0) {
        std.debug.print("  (none)\n", .{});
    }
}

pub fn doCurrent(config: *core.Config) !void {
    const default_file = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "default" });
    defer config.allocator.free(default_file);

    var active_version: []const u8 = "none";
    if (std.Io.Dir.openFileAbsolute(config.io, default_file, .{})) |df| {
        defer df.close(config.io);
        if (df.stat(config.io)) |stat| {
            if (config.allocator.alloc(u8, stat.size)) |content| {
                if (df.readPositionalAll(config.io, content, 0)) |bytes_read| {
                    active_version = std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
                } else |_| {}
            } else |_| {}
        } else |_| {}
    } else |_| {}

    std.debug.print("Global active version: {s}\n", .{active_version});

    const zvm_version = blk: {
        const zvm_zig = try allocZvmBinZigPath(config);
        defer config.allocator.free(zvm_zig);
        break :blk try probeBinaryVersion(config, zvm_zig);
    };
    defer if (zvm_version) |v| config.allocator.free(v);

    if (zvm_version) |version| {
        std.debug.print("ZVM active version:  {s}\n", .{version});
    }

    if (config.project_version) |pv| {
        std.debug.print("Project .zigversion:   {s}\n", .{pv});
        const zvm_matches = versionMatches(pv, zvm_version);
        if (!zvm_matches and !std.mem.eql(u8, active_version, pv)) {
            std.debug.print("==> WARNING: Project version does not match the active Zig toolchain.\n", .{});
            std.debug.print("==> Run 'zigly use' to activate the project version.\n", .{});
        }
    }
}

pub fn doClean(config: *core.Config) !void {
    std.debug.print("==> Cleaning up {s}...\n", .{config.zigly_dir});

    const tmp_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp" });
    defer config.allocator.free(tmp_dir);
    _ = std.Io.Dir.cwd().deleteTree(config.io, tmp_dir) catch {};

    const versions_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "versions" });
    defer config.allocator.free(versions_dir);
    _ = std.Io.Dir.cwd().deleteTree(config.io, versions_dir) catch {};

    const default_file = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "default" });
    defer config.allocator.free(default_file);
    _ = std.Io.Dir.deleteFileAbsolute(config.io, default_file) catch {};

    std.debug.print("==> All cached downloads and versions removed.\n", .{});
}

pub fn doBootstrap(config: *core.Config) !void {
    std.debug.print("==> Bootstrapping project environment...\n", .{});

    if (config.project_version) |pv| {
        try doUse(config, pv);
        std.debug.print("==> Bootstrap complete! You can now run your project builds.\n", .{});
    } else {
        std.debug.print("ERROR: No .zigversion found. Cannot bootstrap.\n", .{});
        return error.NoProjectVersion;
    }
}

pub fn doDoctor(config: *core.Config) !void {
    std.debug.print("==> Toolchain Health Doctor\n", .{});

    const os_arch = getOsArch();
    std.debug.print("OS:         {s} ({s})\n", .{ os_arch.os, os_arch.arch });

    if (config.project_version) |pv| {
        std.debug.print("Project:    {s}\n", .{pv});
    } else {
        std.debug.print("Project:    No .zigversion found\n", .{});
    }

    const zvm_zig = try allocZvmBinZigPath(config);
    defer config.allocator.free(zvm_zig);
    const zvm_version = try probeBinaryVersion(config, zvm_zig);
    defer if (zvm_version) |v| config.allocator.free(v);
    if (zvm_version) |version| {
        std.debug.print("ZVM zig:    {s} ({s})\n", .{ zvm_zig, version });
    } else {
        std.debug.print("ZVM zig:    unavailable or version probe failed\n", .{});
    }

    const local_bin = try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".local", "bin" });
    defer config.allocator.free(local_bin);

    const local_zig = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zig" });
    defer config.allocator.free(local_zig);

    const local_zig_exists = if (std.Io.Dir.accessAbsolute(config.io, local_zig, .{})) true else |_| false;
    if (local_zig_exists) {
        std.debug.print("Zig path:   {s}\n", .{local_zig});
    } else {
        std.debug.print("Zig path:   MISSING\n", .{});
    }

    const local_zls = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zls" });
    defer config.allocator.free(local_zls);

    const local_zls_exists = if (std.Io.Dir.accessAbsolute(config.io, local_zls, .{})) true else |_| false;
    if (local_zls_exists) {
        std.debug.print("ZLS path:   {s}\n", .{local_zls});
    } else {
        std.debug.print("ZLS path:   MISSING\n", .{});
    }

    const path_env = config.environ_map.get("PATH") orelse "";
    if (std.mem.indexOf(u8, path_env, local_bin) != null) {
        std.debug.print("PATH:       {s} is in PATH\n", .{local_bin});
    } else {
        std.debug.print("PATH:       {s} NOT in PATH\n", .{local_bin});
    }
}

pub fn doUnlink(config: *core.Config) !void {
    const local_bin = try std.fs.path.join(config.allocator, &[_][]const u8{ config.home_dir, ".local", "bin" });
    defer config.allocator.free(local_bin);

    const local_zig = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zig" });
    defer config.allocator.free(local_zig);

    const local_zls = try std.fs.path.join(config.allocator, &[_][]const u8{ local_bin, "zls" });
    defer config.allocator.free(local_zls);

    var removed: u8 = 0;

    if (std.Io.Dir.deleteFileAbsolute(config.io, local_zig)) |_| {
        std.debug.print("==> Removed {s}\n", .{local_zig});
        removed += 1;
    } else |_| {}

    if (std.Io.Dir.deleteFileAbsolute(config.io, local_zls)) |_| {
        std.debug.print("==> Removed {s}\n", .{local_zls});
        removed += 1;
    } else |_| {}

    if (removed == 0) {
        std.debug.print("==> No symlinks found in {s}\n", .{local_bin});
    } else {
        std.debug.print("==> Unlinked {d} symlink(s).\n", .{removed});
    }
}

pub fn doCheck(config: *core.Config) !void {
    const version = if (config.project_version) |pv| pv else {
        std.debug.print("ERROR: No .zigversion found. Cannot check for updates.\n", .{});
        return error.NoProjectVersion;
    };

    std.debug.print("Current project version: {s}\n", .{version});
    std.debug.print("==> Fetching latest version from ziglang.org...\n", .{});

    // Fetch the download index and extract the latest master version
    const tmp_index = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "index.json" });
    defer config.allocator.free(tmp_index);

    try std.Io.Dir.cwd().createDirPath(config.io, try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp" }));

    if (download.downloadFile(config.allocator, config.io, "https://ziglang.org/download/index.json", tmp_index)) |_| {
        // Read the file and find the master version string
        if (std.Io.Dir.openFileAbsolute(config.io, tmp_index, .{})) |f| {
            defer f.close(config.io);
            if (f.stat(config.io)) |stat| {
                const max_size = @min(stat.size, 65536);
                if (config.allocator.alloc(u8, max_size)) |content| {
                    if (f.readPositionalAll(config.io, content, 0)) |bytes_read| {
                        const data = content[0..bytes_read];
                        // Look for "master" section and extract version
                        if (std.mem.indexOf(u8, data, "\"master\"")) |_| {
                            // Find the version field after "master"
                            if (extractVersionFromIndex(data)) |latest| {
                                std.debug.print("Latest master version:  {s}\n", .{latest});
                                if (std.mem.eql(u8, version, latest)) {
                                    std.debug.print("==> You are up to date.\n", .{});
                                } else {
                                    std.debug.print("==> Update available! Run 'zigly install' to sync the project toolchain.\n", .{});
                                }
                                return;
                            }
                        }
                    } else |_| {}
                } else |_| {}
            } else |_| {}
        } else |_| {}
        std.debug.print("==> Could not parse version from download index.\n", .{});
    } else |_| {
        std.debug.print("==> Could not fetch download index. Check your internet connection.\n", .{});
    }
}

fn extractVersionFromIndex(data: []const u8) ?[]const u8 {
    // Find "master" key, then look for "version" field within the next ~500 bytes
    const master_pos = std.mem.indexOf(u8, data, "\"master\"") orelse return null;
    const search_end = @min(master_pos + 500, data.len);
    const section = data[master_pos..search_end];

    // Look for the latest development version string in the download index.
    const needle = "\"version\":";
    const ver_pos = std.mem.indexOf(u8, section, needle) orelse
        std.mem.indexOf(u8, section, "\"version\" :") orelse return null;

    // Find the opening quote of the value
    const after_key = section[ver_pos + needle.len ..];
    const quote_start = std.mem.indexOf(u8, after_key, "\"") orelse return null;
    const value_start = after_key[quote_start + 1 ..];
    const quote_end = std.mem.indexOf(u8, value_start, "\"") orelse return null;
    return value_start[0..quote_end];
}

pub fn doListRemote(config: *core.Config) !void {
    _ = config;
    std.debug.print("==> list-remote is currently delegated to curl/jq or requires a browser.\n", .{});
    std.debug.print("Visit https://ziglang.org/download to see available versions.\n", .{});
}

/// Fetch the expected SHA256 hash for a platform tarball from ziglang.org/download/index.json.
/// Looks for the "master" section → "{arch}-{os}" → "shasum" field.
fn fetchExpectedShaSum(config: *core.Config, version: []const u8, arch: []const u8, os: []const u8) ![]const u8 {
    _ = version;
    const tmp_dir = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp" });
    defer config.allocator.free(tmp_dir);
    try std.Io.Dir.cwd().createDirPath(config.io, tmp_dir);

    const index_path = try std.fs.path.join(config.allocator, &[_][]const u8{ tmp_dir, "index.json" });
    defer config.allocator.free(index_path);

    try download.downloadFile(config.allocator, config.io, "https://ziglang.org/download/index.json", index_path);

    const f = try std.Io.Dir.openFileAbsolute(config.io, index_path, .{});
    defer f.close(config.io);
    const stat = try f.stat(config.io);
    const max_size = @min(stat.size, 256 * 1024);
    const content = try config.allocator.alloc(u8, max_size);
    const bytes_read = try f.readPositionalAll(config.io, content, 0);
    const data = content[0..bytes_read];

    // Find "master" section, then "{arch}-{os}" platform key, then "shasum"
    const platform_key = try std.fmt.allocPrint(config.allocator, "\"{s}-{s}\"", .{ arch, os });
    defer config.allocator.free(platform_key);

    const master_pos = std.mem.indexOf(u8, data, "\"master\"") orelse return error.ParseError;
    const section = data[master_pos..];

    const platform_pos = std.mem.indexOf(u8, section, platform_key) orelse return error.ParseError;
    const platform_section = section[platform_pos..@min(platform_pos + 500, section.len)];

    return extractJsonStringField(platform_section, "shasum") orelse error.ParseError;
}

/// Compute SHA256 of a file using the system `shasum` command.
fn computeFileSha256(config: *core.Config, file_path: []const u8) ![]const u8 {
    const hash_file = try std.fs.path.join(config.allocator, &[_][]const u8{ config.zigly_dir, "tmp", "sha256.txt" });
    defer config.allocator.free(hash_file);

    const cmd = try std.fmt.allocPrint(config.allocator, "shasum -a 256 '{s}' | cut -d' ' -f1 > '{s}'", .{ file_path, hash_file });
    defer config.allocator.free(cmd);

    var child_args = std.ArrayListUnmanaged([]const u8).empty;
    try child_args.append(config.allocator, "sh");
    try child_args.append(config.allocator, "-c");
    try child_args.append(config.allocator, cmd);
    var child = try std.process.spawn(config.io, .{ .argv = child_args.items });
    const term = try child.wait(config.io);
    switch (term) {
        .exited => |code| {
            if (code != 0) return error.HashComputeFailed;
        },
        else => return error.HashComputeFailed,
    }

    const f = try std.Io.Dir.openFileAbsolute(config.io, hash_file, .{});
    defer f.close(config.io);
    const stat = try f.stat(config.io);
    const content = try config.allocator.alloc(u8, @min(stat.size, 128));
    const bytes_read = try f.readPositionalAll(config.io, content, 0);
    return std.mem.trim(u8, content[0..bytes_read], " \n\r\t");
}

/// Extract a JSON string value for a given key from a section of text.
/// Simple parser — finds "key": "value" and returns value.
fn extractJsonStringField(section: []const u8, key: []const u8) ?[]const u8 {
    // Build needle: "key"
    // We search for "key" followed by optional whitespace, colon, optional whitespace, quote
    const key_with_quotes_start = std.mem.indexOf(u8, section, "\"") orelse return null;
    _ = key_with_quotes_start;

    // Search for the key name
    var pos: usize = 0;
    while (pos < section.len) {
        const key_pos = std.mem.indexOfPos(u8, section, pos, key) orelse return null;
        // Verify it's a proper JSON key (preceded by quote)
        if (key_pos > 0 and section[key_pos - 1] == '"') {
            const after_key = key_pos + key.len;
            if (after_key < section.len and section[after_key] == '"') {
                // Found "key" — now find : "value"
                var i = after_key + 1;
                // Skip whitespace and colon
                while (i < section.len and (section[i] == ' ' or section[i] == ':' or section[i] == '\n' or section[i] == '\t')) : (i += 1) {}
                if (i < section.len and section[i] == '"') {
                    const val_start = i + 1;
                    const val_end = std.mem.indexOf(u8, section[val_start..], "\"") orelse return null;
                    return section[val_start .. val_start + val_end];
                }
            }
        }
        pos = key_pos + 1;
    }
    return null;
}

pub fn printUsage() void {
    const usage =
        \\Zigly - Zig Version Manager
        \\
        \\Usage: zigly <command> [options]
        \\
        \\Commands:
        \\  install [version]    Install a specific version of Zig + ZLS (prefers ZVM when available)
        \\  use [version]        Activate the requested version and link ~/.local/bin
        \\  unlink               Remove zig/zls symlinks from ~/.local/bin
        \\  check                Check if a newer Zig version is available
        \\  list, ls             List installed versions
        \\  list-remote, lsr     List available versions from ziglang.org
        \\  current              Show the currently active version and project status
        \\  clean                Remove all cached versions and downloads
        \\  bootstrap            One-command project setup (install from .zigversion and link)
        \\  doctor               Report toolchain health diagnostics
        \\  status               Print path to the active zig binary (ZVM-first when versions match)
        \\
        \\Examples:
        \\  zigly install master
        \\  zigly use 0.13.0
        \\  zigly bootstrap
        \\
    ;
    std.debug.print("{s}", .{usage});
}

test "versionMatches requires an exact version string match" {
    try std.testing.expect(versionMatches("0.17.0-dev.135+9df02121d", "0.17.0-dev.135+9df02121d"));
    try std.testing.expect(!versionMatches("0.17.0-dev.135+9df02121d", "0.17.0-dev.27+0dd99c37c"));
    try std.testing.expect(!versionMatches("0.17.0-dev.135+9df02121d", null));
}

test "selectToolchain prefers an active matching zvm binary" {
    try std.testing.expectEqual(
        ToolchainSelection.zvm_active,
        selectToolchain("0.17.0-dev.135+9df02121d", true, "0.17.0-dev.135+9df02121d", true),
    );
}

test "selectToolchain installs via zvm when zvm is present but mismatched" {
    try std.testing.expectEqual(
        ToolchainSelection.install_via_zvm,
        selectToolchain("0.17.0-dev.135+9df02121d", true, "0.17.0-dev.27+0dd99c37c", true),
    );
}

test "selectToolchain falls back to zigly cache only when zvm is absent" {
    try std.testing.expectEqual(
        ToolchainSelection.zigly_cache,
        selectToolchain("0.17.0-dev.135+9df02121d", false, null, true),
    );
    try std.testing.expectEqual(
        ToolchainSelection.install_via_zigly,
        selectToolchain("0.17.0-dev.135+9df02121d", false, null, false),
    );
}

test {
    std.testing.refAllDecls(@This());
}
