const std = @import("std");
const builtin = @import("builtin");
const util = @import("util");

// ── Status types ────────────────────────────────────────────────────────

pub const CheckStatus = enum {
    ok,
    missing,
    degraded,

    pub fn label(self: CheckStatus) []const u8 {
        return switch (self) {
            .ok => "OK",
            .missing => "MISSING",
            .degraded => "DEGRADED",
        };
    }
};

pub const CheckResult = struct {
    name: []const u8,
    status: CheckStatus,
    detail: []const u8,
};

pub const PreflightReport = struct {
    zig_binary: CheckResult,
    zig_version_match: CheckResult,
    platform: CheckResult,
    linker: CheckResult,
    gpu_backend: CheckResult,
    git: CheckResult,
    curl: CheckResult,
    env_openai: CheckResult,
    env_anthropic: CheckResult,
    env_ollama_host: CheckResult,
    env_ollama_model: CheckResult,
    env_hf_token: CheckResult,
    env_discord: CheckResult,
    network: CheckResult,

    const field_count = @typeInfo(PreflightReport).@"struct".fields.len;

    pub fn overallStatus(self: PreflightReport) CheckStatus {
        var worst: CheckStatus = .ok;
        inline for (@typeInfo(PreflightReport).@"struct".fields) |f| {
            const result: CheckResult = @field(self, f.name);
            switch (result.status) {
                .missing => return .missing,
                .degraded => worst = .degraded,
                .ok => {},
            }
        }
        return worst;
    }

    pub fn countByStatus(self: PreflightReport, status: CheckStatus) usize {
        var n: usize = 0;
        inline for (@typeInfo(PreflightReport).@"struct".fields) |f| {
            const result: CheckResult = @field(self, f.name);
            if (result.status == status) n += 1;
        }
        return n;
    }
};

// ── Check helpers ───────────────────────────────────────────────────────

const is_blocked_darwin = builtin.os.tag == .macos and
    builtin.os.version_range.semver.min.major >= 26;

fn checkZigBinary(allocator: std.mem.Allocator, io: std.Io) CheckResult {
    const exists = util.commandExists(allocator, io, "zig") catch false;
    if (!exists) {
        return .{ .name = "zig binary", .status = .missing, .detail = "no 'zig' found on PATH" };
    }
    const res = util.captureCommand(allocator, io, "command -v zig") catch {
        return .{ .name = "zig binary", .status = .ok, .detail = "(path unknown)" };
    };
    defer allocator.free(res.output);
    const path = util.trimSpace(res.output);
    if (path.len == 0) {
        return .{ .name = "zig binary", .status = .ok, .detail = "(path empty)" };
    }
    // Copy path into stable memory — the deferred free above will release res.output
    const stable = allocator.dupe(u8, path) catch "(alloc failed)";
    return .{ .name = "zig binary", .status = .ok, .detail = stable };
}

fn checkZigVersion(allocator: std.mem.Allocator, io: std.Io) CheckResult {
    const expected_raw = util.readFileAlloc(allocator, io, ".zigversion", 1024) catch {
        return .{ .name = "zig version", .status = .degraded, .detail = ".zigversion file not found" };
    };
    defer allocator.free(expected_raw);
    const expected = util.trimSpace(expected_raw);

    const ver_res = util.captureCommand(allocator, io, "zig version") catch {
        return .{ .name = "zig version", .status = .missing, .detail = "could not run 'zig version'" };
    };
    defer allocator.free(ver_res.output);
    const actual = util.trimSpace(ver_res.output);

    if (std.mem.eql(u8, actual, expected)) {
        return .{ .name = "zig version", .status = .ok, .detail = "matches .zigversion" };
    }

    const msg = std.fmt.allocPrint(allocator, "want {s}, have {s}", .{ expected, actual }) catch "mismatch";
    return .{ .name = "zig version", .status = .degraded, .detail = msg };
}

fn checkPlatform() CheckResult {
    const os_name = @tagName(builtin.os.tag);
    if (builtin.os.tag == .macos) {
        const major = builtin.os.version_range.semver.min.major;
        if (major >= 26) {
            return .{ .name = "platform", .status = .degraded, .detail = "macOS 26+ (darwin " ++ std.fmt.comptimePrint("{d}", .{major}) ++ ") — limited linker support" };
        }
        return .{ .name = "platform", .status = .ok, .detail = "macOS (darwin " ++ std.fmt.comptimePrint("{d}", .{major}) ++ ")" };
    }
    return .{ .name = "platform", .status = .ok, .detail = os_name };
}

fn checkLinker() CheckResult {
    if (is_blocked_darwin) {
        return .{ .name = "linker", .status = .degraded, .detail = "blocked darwin — requires host-built Zig matching .zigversion" };
    }
    if (builtin.os.tag == .macos) {
        return .{ .name = "linker", .status = .ok, .detail = "macOS system linker" };
    }
    return .{ .name = "linker", .status = .ok, .detail = "default linker" };
}

fn checkGpuBackend(allocator: std.mem.Allocator, io: std.Io) CheckResult {
    if (builtin.os.tag == .macos) {
        if (util.fileExists(io, "/System/Library/Frameworks/Metal.framework/Metal")) {
            return .{ .name = "gpu/metal", .status = .ok, .detail = "Metal.framework present" };
        }
        // Try via test -d as fallback
        const res = util.captureCommand(allocator, io, "test -d /System/Library/Frameworks/Metal.framework && echo yes || echo no") catch {
            return .{ .name = "gpu/metal", .status = .degraded, .detail = "could not probe Metal.framework" };
        };
        defer allocator.free(res.output);
        const out = util.trimSpace(res.output);
        if (std.mem.eql(u8, out, "yes")) {
            return .{ .name = "gpu/metal", .status = .ok, .detail = "Metal.framework directory present" };
        }
        return .{ .name = "gpu/metal", .status = .missing, .detail = "Metal.framework not found" };
    }
    if (builtin.os.tag == .linux) {
        return .{ .name = "gpu/vulkan", .status = .degraded, .detail = "vulkan availability not probed (run vulkaninfo manually)" };
    }
    return .{ .name = "gpu", .status = .degraded, .detail = "no GPU backend detection for this platform" };
}

fn checkTool(allocator: std.mem.Allocator, io: std.Io, name: []const u8) CheckResult {
    const exists = util.commandExists(allocator, io, name) catch false;
    if (exists) {
        return .{ .name = name, .status = .ok, .detail = "found on PATH" };
    }
    return .{ .name = name, .status = .missing, .detail = "not found on PATH" };
}

fn checkEnvVar(allocator: std.mem.Allocator, io: std.Io, name: []const u8) CheckResult {
    const cmd = std.fmt.allocPrint(allocator, "printf '%s' \"${s}\"", .{name}) catch {
        return .{ .name = name, .status = .degraded, .detail = "alloc failed" };
    };
    defer allocator.free(cmd);

    const res = util.captureCommand(allocator, io, cmd) catch {
        return .{ .name = name, .status = .degraded, .detail = "could not read env" };
    };
    defer allocator.free(res.output);

    const val = util.trimSpace(res.output);
    if (val.len > 0) {
        return .{ .name = name, .status = .ok, .detail = "SET" };
    }
    return .{ .name = name, .status = .missing, .detail = "UNSET" };
}

fn checkHostZigCache(allocator: std.mem.Allocator, io: std.Io) CheckResult {
    // Check whether the canonical host-built Zig cache contains a matching binary.
    const expected_raw = util.readFileAlloc(allocator, io, ".zigversion", 1024) catch {
        return .{ .name = "host-zig-cache", .status = .degraded, .detail = ".zigversion not readable" };
    };
    defer allocator.free(expected_raw);
    const expected = util.trimSpace(expected_raw);

    const cache_check = std.fmt.allocPrint(allocator, "test -x \"$HOME/.cache/abi-host-zig/{s}/bin/zig\" && echo yes || echo no", .{expected}) catch {
        return .{ .name = "host-zig-cache", .status = .degraded, .detail = "alloc failed" };
    };
    defer allocator.free(cache_check);

    const res = util.captureCommand(allocator, io, cache_check) catch {
        return .{ .name = "host-zig-cache", .status = .degraded, .detail = "could not probe cache" };
    };
    defer allocator.free(res.output);

    const out = util.trimSpace(res.output);
    if (std.mem.eql(u8, out, "yes")) {
        return .{ .name = "host-zig-cache", .status = .ok, .detail = "host-built Zig found in canonical cache" };
    }
    if (is_blocked_darwin) {
        return .{ .name = "host-zig-cache", .status = .missing, .detail = "NOT FOUND — host-built Zig required on blocked Darwin" };
    }
    return .{ .name = "host-zig-cache", .status = .degraded, .detail = "not found (optional on this platform)" };
}

fn checkNetwork(allocator: std.mem.Allocator, io: std.Io) CheckResult {
    // Probe whether curl is available as a proxy for network tooling
    const has_curl = util.commandExists(allocator, io, "curl") catch false;
    if (!has_curl) {
        return .{ .name = "network", .status = .degraded, .detail = "curl not found — server test probes unavailable" };
    }
    return .{ .name = "network", .status = .ok, .detail = "curl available for server test probes" };
}

// ── Report builder ──────────────────────────────────────────────────────

fn buildReport(allocator: std.mem.Allocator, io: std.Io) PreflightReport {
    return .{
        .zig_binary = checkZigBinary(allocator, io),
        .zig_version_match = checkZigVersion(allocator, io),
        .platform = checkPlatform(),
        .linker = checkLinker(),
        .gpu_backend = checkGpuBackend(allocator, io),
        .git = checkTool(allocator, io, "git"),
        .curl = checkTool(allocator, io, "curl"),
        .env_openai = checkEnvVar(allocator, io, "ABI_OPENAI_API_KEY"),
        .env_anthropic = checkEnvVar(allocator, io, "ABI_ANTHROPIC_API_KEY"),
        .env_ollama_host = checkEnvVar(allocator, io, "ABI_OLLAMA_HOST"),
        .env_ollama_model = checkEnvVar(allocator, io, "ABI_OLLAMA_MODEL"),
        .env_hf_token = checkEnvVar(allocator, io, "ABI_HF_API_TOKEN"),
        .env_discord = checkEnvVar(allocator, io, "DISCORD_BOT_TOKEN"),
        .network = checkNetwork(allocator, io),
    };
}

// ── Printer ─────────────────────────────────────────────────────────────

fn printResult(result: CheckResult) void {
    const pad_width = 28;
    const name_len = result.name.len;
    const pad = if (name_len < pad_width) pad_width - name_len else 1;
    std.debug.print("  {s}:", .{result.name});
    var i: usize = 0;
    while (i < pad) : (i += 1) std.debug.print(" ", .{});
    std.debug.print("{s}", .{result.status.label()});
    if (result.detail.len > 0) {
        std.debug.print(" ({s})", .{result.detail});
    }
    std.debug.print("\n", .{});
}

fn printSection(comptime title: []const u8, results: []const CheckResult) void {
    std.debug.print("\n[{s}]\n", .{title});
    for (results) |r| printResult(r);
}

fn printReport(report: PreflightReport) void {
    std.debug.print("=== ABI Preflight Diagnostics ===\n", .{});

    printSection("Zig Toolchain", &.{
        report.zig_binary,
        report.zig_version_match,
    });

    printSection("Platform", &.{
        report.platform,
        report.linker,
        report.gpu_backend,
    });

    printSection("Tools", &.{
        report.git,
        report.curl,
    });

    printSection("Environment Variables", &.{
        report.env_openai,
        report.env_anthropic,
        report.env_ollama_host,
        report.env_ollama_model,
        report.env_hf_token,
        report.env_discord,
    });

    printSection("Network", &.{
        report.network,
    });

    const ok_count = report.countByStatus(.ok);
    const missing_count = report.countByStatus(.missing);
    const degraded_count = report.countByStatus(.degraded);
    const overall = report.overallStatus();

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("  available:  {d}\n", .{ok_count});
    std.debug.print("  missing:    {d}\n", .{missing_count});
    std.debug.print("  degraded:   {d}\n", .{degraded_count});

    switch (overall) {
        .ok => std.debug.print("  RESULT: OK (all integration prerequisites satisfied)\n", .{}),
        .degraded => std.debug.print("  RESULT: DEGRADED (integration tests can run with reduced coverage)\n", .{}),
        .missing => std.debug.print("  RESULT: BLOCKED (critical prerequisites missing — see MISSING items above)\n", .{}),
    }
}

// ── JSON Output ─────────────────────────────────────────────────────────

fn printJsonReport(report: PreflightReport) void {
    std.debug.print("{{", .{});
    std.debug.print("\"overall\":\"{s}\",", .{report.overallStatus().label()});
    std.debug.print("\"ok\":{d},\"missing\":{d},\"degraded\":{d},", .{
        report.countByStatus(.ok),
        report.countByStatus(.missing),
        report.countByStatus(.degraded),
    });
    std.debug.print("\"checks\":[", .{});
    var first = true;
    inline for (@typeInfo(PreflightReport).@"struct".fields) |f| {
        const result: CheckResult = @field(report, f.name);
        if (!first) std.debug.print(",", .{});
        first = false;
        std.debug.print("{{\"name\":\"{s}\",\"status\":\"{s}\",\"detail\":\"{s}\"}}", .{
            result.name,
            result.status.label(),
            result.detail,
        });
    }
    std.debug.print("]}}\n", .{});
}

// ── Entrypoint ──────────────────────────────────────────────────────────

/// Exit codes:
///   0 = all checks OK
///   1 = blocked (critical tool missing — e.g. zig binary)
///   2 = degraded (some features unavailable but tests can run)
pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    // Parse arguments for --json flag.
    var json_mode = false;
    var args_iter = try std.process.Args.Iterator.initAllocator(init.args, allocator);
    defer args_iter.deinit();
    _ = args_iter.next(); // skip binary name
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--json")) {
            json_mode = true;
        }
    }

    const report = buildReport(allocator, io);

    if (json_mode) {
        printJsonReport(report);
    } else {
        printReport(report);
    }

    // Distinct exit codes based on overall status:
    //   0 = OK, 1 = blocked (missing critical), 2 = degraded
    const overall = report.overallStatus();
    switch (overall) {
        .missing => std.process.exit(1),
        .degraded => std.process.exit(2),
        .ok => {},
    }
}
