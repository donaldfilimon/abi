const std = @import("std");
const model = @import("model.zig");
const source_abi = @import("source_abi.zig");
const source_cli = @import("source_cli.zig");
const source_readme = @import("source_readme.zig");
const source_roadmap = @import("source_roadmap.zig");
const source_features = @import("source_features.zig");
const source_baseline = @import("source_baseline.zig");
const render_api_md = @import("render_api_md.zig");
const render_guides_md = @import("render_guides_md.zig");
const render_plans_md = @import("render_plans_md.zig");
const render_api_app = @import("render_api_app.zig");
const check = @import("check.zig");

pub const Options = struct {
    check: bool = false,
    api_only: bool = false,
    no_wasm: bool = false,
    help: bool = false,
};

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.environ });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const argv = try init.args.toSlice(arena.allocator());
    const opts = parseArgs(argv[1..]);

    if (opts.help) {
        printHelp();
        return;
    }

    try cwd.createDirPath(io, "docs/api");
    if (!opts.api_only) {
        try cwd.createDirPath(io, "docs/_docs");
        try cwd.createDirPath(io, "docs/api-app/data");
        try cwd.createDirPath(io, "docs/plans");
    }

    const modules = try source_abi.discoverModules(allocator, io, cwd);
    defer model.deinitModuleSlice(allocator, modules);

    const commands = try source_cli.discoverCommands(allocator, io, cwd);
    defer model.deinitCommandSlice(allocator, commands);

    const features = try source_features.discoverFeatures(allocator, io, cwd);
    defer model.deinitFeatureSlice(allocator, features);

    const readmes = try source_readme.collectReadmeSummaries(allocator, io, cwd);
    defer model.deinitReadmeSlice(allocator, readmes);

    const roadmap_data = try source_roadmap.discover(allocator);
    defer roadmap_data.deinit(allocator);

    var outputs = std.ArrayListUnmanaged(model.OutputFile).empty;
    defer {
        for (outputs.items) |output| output.deinit(allocator);
        outputs.deinit(allocator);
    }

    try render_api_md.render(allocator, modules, &outputs);

    if (!opts.api_only) {
        const meta = try source_baseline.loadBuildMeta(allocator, io, cwd);
        defer allocator.free(meta.zig_version);

        try render_guides_md.render(
            allocator,
            io,
            cwd,
            meta,
            modules,
            commands,
            features,
            readmes,
            roadmap_data.roadmap_entries,
            roadmap_data.plan_entries,
            &outputs,
        );
        try render_plans_md.render(
            allocator,
            io,
            cwd,
            roadmap_data.plan_entries,
            roadmap_data.roadmap_entries,
            &outputs,
        );
        try render_api_app.render(
            allocator,
            modules,
            commands,
            features,
            roadmap_data.roadmap_entries,
            roadmap_data.plan_entries,
            &outputs,
        );
    }

    if (opts.check) {
        try check.verifyOutputs(allocator, io, cwd, outputs.items);
        if (!opts.api_only and !opts.no_wasm) {
            try verifyWasmDrift(allocator, io, cwd);
        }
        return;
    }

    try check.writeOutputs(allocator, io, cwd, outputs.items);

    if (!opts.api_only and !opts.no_wasm) {
        try buildWasmAssets(allocator, io, cwd);
        try tryBuildComponent(allocator, io, cwd);
    }

    std.debug.print("OK: generated {d} docs artifacts\n", .{outputs.items.len});
}

fn parseArgs(args: []const [:0]const u8) Options {
    var opts = Options{};
    for (args) |arg_z| {
        const arg = std.mem.sliceTo(arg_z, 0);
        if (std.mem.eql(u8, arg, "--check")) {
            opts.check = true;
        } else if (std.mem.eql(u8, arg, "--api-only")) {
            opts.api_only = true;
        } else if (std.mem.eql(u8, arg, "--no-wasm")) {
            opts.no_wasm = true;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            opts.help = true;
        }
    }
    return opts;
}

fn printHelp() void {
    std.debug.print(
        \\Usage: zig build gendocs -- [--check] [--api-only] [--no-wasm]
        \\
        \\Generate ABI docs pipeline outputs.
        \\
        \\Modes:
        \\  --check      Validate generated outputs are up to date (no writes)
        \\  --api-only   Generate only docs/api markdown outputs
        \\  --no-wasm    Skip docs api-app wasm runtime build
        \\
        \\Artifacts:
        \\  docs/api/*.md
        \\  docs/_docs/*.md
        \\  docs/plans/*.md
        \\  docs/api-app/*
        \\
    , .{});
}

fn verifyWasmDrift(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    const generated_path = ".zig-cache/gendocs-docs-engine.wasm";
    defer cwd.deleteFile(io, generated_path) catch {};

    try runWasmBuild(allocator, io, generated_path);

    const expected = try cwd.readFileAlloc(io, "docs/api-app/data/docs_engine.wasm", allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(expected);

    const generated = try cwd.readFileAlloc(io, generated_path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(generated);

    if (!std.mem.eql(u8, expected, generated)) {
        std.debug.print("ERROR: docs/api-app/data/docs_engine.wasm is out of date\n", .{});
        return check.CheckError.DriftDetected;
    }
}

fn buildWasmAssets(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    _ = cwd;
    try runWasmBuild(allocator, io, "docs/api-app/data/docs_engine.wasm");
}

fn runWasmBuild(allocator: std.mem.Allocator, io: std.Io, emit_path: []const u8) !void {
    const emit_arg = try std.fmt.allocPrint(allocator, "-femit-bin={s}", .{emit_path});
    defer allocator.free(emit_arg);

    const argv = [_][]const u8{
        "zig",
        "build-exe",
        "tools/gendocs/wasm/exports.zig",
        "-target",
        "wasm32-freestanding",
        "-O",
        "ReleaseSmall",
        "-fno-entry",
        "-rdynamic",
        emit_arg,
    };

    try runCommand(io, &argv);
}

fn tryBuildComponent(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) !void {
    if (!commandExists(io, "wasm-tools")) {
        std.debug.print("INFO: wasm-tools not found; skipping component packaging\n", .{});
    } else {
        const argv = [_][]const u8{
            "wasm-tools",
            "component",
            "new",
            "docs/api-app/data/docs_engine.wasm",
            "-o",
            "docs/api-app/data/docs_engine.component.wasm",
        };

        runCommand(io, &argv) catch |err| {
            std.debug.print("WARN: component packaging skipped: {t}\n", .{err});
        };
    }

    // Keep WIT source co-located with generated app data for discoverability.
    const wit = try cwd.readFileAlloc(io, "tools/gendocs/wasm/wit/docs_engine.wit", allocator, .limited(64 * 1024));
    defer allocator.free(wit);
    var out = try cwd.createFile(io, "docs/api-app/data/docs_engine.wit", .{ .truncate = true });
    defer out.close(io);
    try out.writeStreamingAll(io, wit);
}

fn commandExists(io: std.Io, cmd: []const u8) bool {
    var child = std.process.spawn(io, .{
        .argv = &.{ cmd, "--version" },
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch return false;

    const term = child.wait(io) catch return false;
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn runCommand(io: std.Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });

    const term = try child.wait(io);
    switch (term) {
        .exited => |code| {
            if (code != 0) return error.CommandFailed;
        },
        else => return error.CommandFailed,
    }
}

test "parseArgs recognizes pipeline flags" {
    const args: []const [:0]const u8 = &.{
        "--check",
        "--api-only",
        "--no-wasm",
    };
    const opts = parseArgs(args);
    try std.testing.expect(opts.check);
    try std.testing.expect(opts.api_only);
    try std.testing.expect(opts.no_wasm);
}
