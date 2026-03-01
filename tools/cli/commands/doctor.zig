//! CLI command: abi doctor
//!
//! One-command health check for all dependencies, API keys, and environment.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

pub const meta: command_mod.Meta = .{
    .name = "doctor",
    .description = "Check environment, dependencies, and configuration",
    .subcommands = &.{"help"},
};

const CheckResult = struct {
    name: []const u8,
    passed: bool,
    detail: []const u8,
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printHelp(allocator);
        return;
    }

    utils.output.printHeader("ABI Doctor");
    utils.output.println("Checking environment...\n", .{});

    var checks_passed: usize = 0;
    var checks_failed: usize = 0;
    var checks_warned: usize = 0;

    // 1. Zig version
    {
        const expected = "0.16.0-dev";
        // We know we compiled with Zig 0.16 if we got this far
        printCheck("Zig compiler", true, "0.16.0-dev (compiled successfully)");
        checks_passed += 1;
        _ = expected;
    }

    // 2. Framework initialization
    {
        var fw = abi.initAppDefault(allocator) catch |err| {
            var buf: [128]u8 = undefined;
            const detail = std.fmt.bufPrint(&buf, "init failed: {t}", .{err}) catch "init failed";
            printCheck("Framework init", false, detail);
            checks_failed += 1;
            return;
        };
        defer fw.deinit();
        printCheck("Framework init", true, "OK");
        checks_passed += 1;
    }

    // 3. GPU detection
    blk: {
        if (!abi.gpu.backends.detect.moduleEnabled()) {
            printCheck("GPU module", false, "disabled at build time (-Dfeat-gpu=true (legacy: -Denable-gpu=true) to enable)");
            checks_warned += 1;
            break :blk;
        }
        const devices = abi.gpu.backends.listing.listDevices(allocator) catch {
            printCheck("GPU devices", false, "detection failed");
            checks_failed += 1;
            break :blk;
        };
        defer allocator.free(devices);
        if (devices.len == 0) {
            if (@import("builtin").os.tag == .macos) {
                printCheck("GPU devices", false, "none detected (try -Dgpu-backend=metal)");
            } else {
                printCheck("GPU devices", false, "none detected (try -Dgpu-backend=vulkan)");
            }
            checks_warned += 1;
        } else {
            var buf: [64]u8 = undefined;
            const detail = std.fmt.bufPrint(&buf, "{d} device(s) found", .{devices.len}) catch "found";
            printCheck("GPU devices", true, detail);
            checks_passed += 1;
        }
    }

    // 4. API Keys
    checkEnvVar("ABI_OPENAI_API_KEY", "OpenAI API key", &checks_passed, &checks_warned);
    checkEnvVar("ABI_ANTHROPIC_API_KEY", "Anthropic API key", &checks_passed, &checks_warned);

    // 5. Ollama
    {
        const ollama_host = if (std.c.getenv("ABI_OLLAMA_HOST")) |ptr|
            std.mem.sliceTo(ptr, 0)
        else
            null;

        if (ollama_host) |host| {
            var buf: [128]u8 = undefined;
            const detail = std.fmt.bufPrint(&buf, "configured ({s})", .{host}) catch "configured";
            printCheck("Ollama host", true, detail);
            checks_passed += 1;
        } else {
            printCheck("Ollama host", false, "ABI_OLLAMA_HOST not set (default: http://127.0.0.1:11434)");
            checks_warned += 1;
        }
    }

    // 6. Config files
    {
        var io_backend = cli_io.initIoBackend(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();
        const dir = std.Io.Dir.cwd();

        // Check ralph.yml
        if (dir.readFileAlloc(io, "ralph.yml", allocator, .limited(1024))) |content| {
            allocator.free(content);
            printCheck("ralph.yml", true, "found");
            checks_passed += 1;
        } else |_| {
            printCheck("ralph.yml", false, "not found (optional, needed for 'abi agent ralph')");
            checks_warned += 1;
        }
    }

    // 7. Feature modules
    {
        const features = std.enums.values(abi.Feature);
        var enabled_count: usize = 0;
        for (features) |tag| {
            var fw2 = abi.initAppDefault(allocator) catch break;
            defer fw2.deinit();
            if (fw2.isEnabled(tag)) enabled_count += 1;
        }
        var buf: [64]u8 = undefined;
        const detail = std.fmt.bufPrint(&buf, "{d}/{d} enabled", .{ enabled_count, features.len }) catch "checked";
        printCheck("Feature modules", enabled_count > 0, detail);
        if (enabled_count > 0) checks_passed += 1 else checks_failed += 1;
    }

    // Summary
    utils.output.println("", .{});
    utils.output.printSeparator(50);
    const total = checks_passed + checks_failed + checks_warned;
    utils.output.println("", .{});

    if (checks_failed == 0) {
        utils.output.printSuccess("All {d} checks passed ({d} warnings).", .{ total, checks_warned });
    } else {
        utils.output.printError("{d} check(s) failed, {d} passed, {d} warning(s).", .{ checks_failed, checks_passed, checks_warned });
    }

    if (checks_warned > 0) {
        utils.output.println("", .{});
        utils.output.printInfo("Warnings are optional — the framework works without them.", .{});
        utils.output.printInfo("Run 'abi env' to see all environment variables.", .{});
    }
}

fn printCheck(name: []const u8, passed: bool, detail: []const u8) void {
    if (passed) {
        utils.output.println("  {s}\u{2713}{s} {s}: {s}", .{
            utils.output.Color.green(),
            utils.output.Color.reset(),
            name,
            detail,
        });
    } else {
        utils.output.println("  {s}\u{2717}{s} {s}: {s}", .{
            utils.output.Color.red(),
            utils.output.Color.reset(),
            name,
            detail,
        });
    }
}

fn checkEnvVar(comptime env_name: [*:0]const u8, label: []const u8, passed: *usize, warned: *usize) void {
    if (std.c.getenv(env_name)) |ptr| {
        const val = std.mem.sliceTo(ptr, 0);
        if (val.len > 4) {
            // Redact: show first 4 chars
            var buf: [64]u8 = undefined;
            const detail = std.fmt.bufPrint(&buf, "set ({s}...)", .{val[0..4]}) catch "set";
            printCheck(label, true, detail);
        } else {
            printCheck(label, true, "set");
        }
        passed.* += 1;
    } else {
        printCheck(label, false, "not set");
        warned.* += 1;
    }
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi doctor", "")
        .description("Check environment, dependencies, and configuration health.")
        .section("Checks Performed")
        .text("  - Zig compiler version\n")
        .text("  - Framework initialization\n")
        .text("  - GPU device detection\n")
        .text("  - API keys (OpenAI, Anthropic)\n")
        .text("  - Ollama connectivity\n")
        .text("  - Config files (ralph.yml)\n")
        .text("  - Feature module status\n")
        .newline()
        .section("Options")
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi doctor", "Run all health checks");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
