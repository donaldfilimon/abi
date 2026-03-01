//! LSP client CLI command.
//!
//! Talks to ZLS over JSON-RPC and exposes common LSP requests.
//!
//! Usage:
//!   abi lsp hover --path src/main.zig --line 0 --character 0
//!   abi lsp completion --path src/main.zig --line 3 --character 10
//!   abi lsp request --method textDocument/hover --params '{...}'

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = @import("../utils/io_backend.zig");

const lsp = abi.services.lsp;
const ArgParser = utils.args.ArgParser;
const output = utils.output;
const HelpBuilder = utils.help.HelpBuilder;
const common_options = utils.help.common_options;

const max_doc_bytes = 4 * 1024 * 1024;

pub const meta: command_mod.Meta = .{
    .name = "lsp",
    .description = "ZLS LSP client (request, hover, completion, definition, rename, format)",
    .kind = .group,
    .subcommands = &.{
        "request",
        "notify",
        "hover",
        "completion",
        "definition",
        "references",
        "rename",
        "format",
        "diagnostics",
        "help",
    },
    .children = &.{
        .{ .name = "request", .description = "Send an LSP request", .handler = command_mod.parserHandler(runRequestSubcommand) },
        .{ .name = "notify", .description = "Send an LSP notification", .handler = command_mod.parserHandler(runNotifySubcommand) },
        .{ .name = "hover", .description = "Get hover info at position", .handler = command_mod.parserHandler(runHoverSubcommand) },
        .{ .name = "completion", .description = "Get completion items at position", .handler = command_mod.parserHandler(runCompletionSubcommand) },
        .{ .name = "definition", .description = "Get definition at position", .handler = command_mod.parserHandler(runDefinitionSubcommand) },
        .{ .name = "references", .description = "Find references at position", .handler = command_mod.parserHandler(runReferencesSubcommand) },
        .{ .name = "rename", .description = "Rename symbol at position", .handler = command_mod.parserHandler(runRenameSubcommand) },
        .{ .name = "format", .description = "Format document and return edits", .handler = command_mod.parserHandler(runFormatSubcommand) },
        .{ .name = "diagnostics", .description = "Fetch document diagnostics", .handler = command_mod.parserHandler(runDiagnosticsSubcommand) },
    },
};

pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    if (args.len == 0) {
        printHelp(ctx.allocator);
        return;
    }
    const cmd = std.mem.sliceTo(args[0], 0);
    if (utils.args.matchesAny(cmd, &.{ "--help", "-h", "help" })) {
        printHelp(ctx.allocator);
        return;
    }
    output.printError("Unknown lsp command: {s}", .{cmd});
    if (command_mod.suggestSubcommand(meta, cmd)) |suggestion| {
        output.println("Did you mean: {s}", .{suggestion});
    }
}

fn runRequestSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var method: ?[]const u8 = null;
    var params_json: ?[]const u8 = null;
    var params_owned: ?[]u8 = null;
    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{"--method"})) |value| {
            method = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--params"})) |value| {
            params_json = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--params-file"})) |value| {
            var io_backend = cli_io.initIoBackend(allocator);
            defer io_backend.deinit();
            const io = io_backend.io();
            params_owned = try std.Io.Dir.cwd().readFileAlloc(io, value, allocator, .limited(max_doc_bytes));
            params_json = params_owned;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--path"})) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }

        const arg = parser.next().?;
        output.printError("Unexpected argument for 'request': {s}", .{arg});
        return;
    }

    if (method == null) {
        output.printError("--method is required", .{});
        return;
    }
    defer if (params_owned) |owned| allocator.free(owned);

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    if (path_opt) |path| {
        if (text_opt == null and path.len == 0) {
            output.printError("--path must be non-empty", .{});
            return;
        }
        const uri = try openDocument(allocator, io_backend.io(), &client, path, text_opt);
        defer allocator.free(uri);
    } else if (text_opt != null) {
        output.printError("--text requires --path", .{});
        return;
    }

    const resp = try client.requestRaw(method.?, params_json);
    defer allocator.free(resp.json);

    if (resp.is_error) {
        output.printWarning("LSP error response", .{});
    }
    output.println("{s}", .{resp.json});
}

fn runNotifySubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var method: ?[]const u8 = null;
    var params_json: ?[]const u8 = null;
    var params_owned: ?[]u8 = null;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{"--method"})) |value| {
            method = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--params"})) |value| {
            params_json = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--params-file"})) |value| {
            var io_backend = cli_io.initIoBackend(allocator);
            defer io_backend.deinit();
            const io = io_backend.io();
            params_owned = try std.Io.Dir.cwd().readFileAlloc(io, value, allocator, .limited(max_doc_bytes));
            params_json = params_owned;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'notify': {s}", .{arg});
        return;
    }

    if (method == null) {
        output.printError("--method is required", .{});
        return;
    }
    defer if (params_owned) |owned| allocator.free(owned);

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    try client.notifyRaw(method.?, params_json);
    output.printSuccess("Notification sent", .{});
}

fn runHoverSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    const resp_opt = try runPositionSubcommand(allocator, parser, .hover);
    if (resp_opt == null) return;
    const resp = resp_opt.?;
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runCompletionSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    const resp_opt = try runPositionSubcommand(allocator, parser, .completion);
    if (resp_opt == null) return;
    const resp = resp_opt.?;
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runDefinitionSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    const resp_opt = try runPositionSubcommand(allocator, parser, .definition);
    if (resp_opt == null) return;
    const resp = resp_opt.?;
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runReferencesSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;
    var line: ?u32 = null;
    var character: ?u32 = null;
    var include_decl: bool = true;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{ "--path", "-p" })) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--line", "-l" })) |value| {
            line = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--character", "-c" })) |value| {
            character = value;
            continue;
        }
        if (parser.consumeFlag(&.{"--include-declaration"})) {
            include_decl = true;
            continue;
        }
        if (parser.consumeFlag(&.{"--no-include-declaration"})) {
            include_decl = false;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'references': {s}", .{arg});
        return;
    }

    if (path_opt == null or line == null or character == null) {
        output.printError("--path, --line, and --character are required", .{});
        return;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    const uri = try openDocument(allocator, io_backend.io(), &client, path_opt.?, text_opt);
    defer allocator.free(uri);

    const pos = lsp.types.Position{ .line = line.?, .character = character.? };
    const resp = try client.references(uri, pos, include_decl);
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runRenameSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;
    var line: ?u32 = null;
    var character: ?u32 = null;
    var new_name: ?[]const u8 = null;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{ "--path", "-p" })) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--line", "-l" })) |value| {
            line = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--character", "-c" })) |value| {
            character = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{ "--new-name", "-n" })) |value| {
            new_name = value;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'rename': {s}", .{arg});
        return;
    }

    if (path_opt == null or line == null or character == null or new_name == null) {
        output.printError("--path, --line, --character, and --new-name are required", .{});
        return;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    const uri = try openDocument(allocator, io_backend.io(), &client, path_opt.?, text_opt);
    defer allocator.free(uri);

    const pos = lsp.types.Position{ .line = line.?, .character = character.? };
    const resp = try client.rename(uri, pos, new_name.?);
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runFormatSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;
    var tab_size: u32 = 4;
    var insert_spaces: bool = true;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{ "--path", "-p" })) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{"--tab-size"})) |value| {
            tab_size = value;
            continue;
        }
        if (parser.consumeFlag(&.{"--insert-spaces"})) {
            insert_spaces = true;
            continue;
        }
        if (parser.consumeFlag(&.{"--tabs"})) {
            insert_spaces = false;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'format': {s}", .{arg});
        return;
    }

    if (path_opt == null) {
        output.printError("--path is required", .{});
        return;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    const uri = try openDocument(allocator, io_backend.io(), &client, path_opt.?, text_opt);
    defer allocator.free(uri);

    const options = lsp.types.FormattingOptions{
        .tabSize = tab_size,
        .insertSpaces = insert_spaces,
    };
    const resp = try client.formatting(uri, options);
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

fn runDiagnosticsSubcommand(allocator: std.mem.Allocator, parser: *ArgParser) !void {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{ "--path", "-p" })) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument for 'diagnostics': {s}", .{arg});
        return;
    }

    if (path_opt == null) {
        output.printError("--path is required", .{});
        return;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    const uri = try openDocument(allocator, io_backend.io(), &client, path_opt.?, text_opt);
    defer allocator.free(uri);

    const resp = try client.diagnostics(uri);
    defer allocator.free(resp.json);
    if (resp.is_error) output.printWarning("LSP error response", .{});
    output.println("{s}", .{resp.json});
}

const PositionCommand = enum { hover, completion, definition };

fn runPositionSubcommand(
    allocator: std.mem.Allocator,
    parser: *ArgParser,
    cmd: PositionCommand,
) !?lsp.Response {
    var env_cfg = try lsp.loadConfigFromEnv(allocator);
    defer env_cfg.deinit();
    var cfg = env_cfg.config;

    var path_opt: ?[]const u8 = null;
    var text_opt: ?[]const u8 = null;
    var line: ?u32 = null;
    var character: ?u32 = null;

    while (parser.hasMore()) {
        if (parser.wantsHelp()) {
            printHelp(allocator);
            return null;
        }
        if (try applyCommonOption(parser, &cfg)) continue;
        if (try takeOptionValue(parser, &.{ "--path", "-p" })) |value| {
            path_opt = value;
            continue;
        }
        if (try takeOptionValue(parser, &.{"--text"})) |value| {
            text_opt = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--line", "-l" })) |value| {
            line = value;
            continue;
        }
        if (try takeOptionU32(parser, &.{ "--character", "-c" })) |value| {
            character = value;
            continue;
        }
        const arg = parser.next().?;
        output.printError("Unexpected argument: {s}", .{arg});
        return null;
    }

    if (path_opt == null or line == null or character == null) {
        output.printError("--path, --line, and --character are required", .{});
        return null;
    }

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    var client = try lsp.Client.init(allocator, io_backend.io(), cfg);
    defer client.deinit();

    const uri = try openDocument(allocator, io_backend.io(), &client, path_opt.?, text_opt);
    defer allocator.free(uri);

    const pos = lsp.types.Position{ .line = line.?, .character = character.? };
    const resp = switch (cmd) {
        .hover => try client.hover(uri, pos),
        .completion => try client.completion(uri, pos),
        .definition => try client.definition(uri, pos),
    };
    return resp;
}

fn applyCommonOption(parser: *ArgParser, cfg: *lsp.Config) !bool {
    if (try takeOptionValue(parser, &.{"--zls"})) |value| {
        cfg.zls_path = value;
        return true;
    }
    if (try takeOptionValue(parser, &.{"--zig"})) |value| {
        cfg.zig_exe_path = value;
        return true;
    }
    if (try takeOptionValue(parser, &.{"--root"})) |value| {
        cfg.workspace_root = value;
        return true;
    }
    if (try takeOptionValue(parser, &.{"--log-level"})) |value| {
        cfg.log_level = value;
        return true;
    }
    if (parser.consumeFlag(&.{"--no-snippets"})) {
        cfg.enable_snippets = false;
        return true;
    }
    if (parser.consumeFlag(&.{"--snippets"})) {
        cfg.enable_snippets = true;
        return true;
    }
    return false;
}

fn takeOptionValue(parser: *ArgParser, options: []const []const u8) !?[]const u8 {
    if (!parser.matches(options)) return null;
    _ = parser.next();
    return parser.next() orelse error.MissingValue;
}

fn takeOptionU32(parser: *ArgParser, options: []const []const u8) !?u32 {
    const value = try takeOptionValue(parser, options) orelse return null;
    return std.fmt.parseInt(u32, value, 10) catch error.InvalidValue;
}

fn openDocument(
    allocator: std.mem.Allocator,
    io: std.Io,
    client: *lsp.Client,
    path: []const u8,
    text_override: ?[]const u8,
) ![]u8 {
    const resolved = try lsp.resolvePath(allocator, io, client.workspaceRoot(), path);
    defer allocator.free(resolved);

    const uri = try lsp.pathToUri(allocator, resolved);

    var owned_text: ?[]u8 = null;
    const text = text_override orelse blk: {
        const contents = try std.Io.Dir.cwd().readFileAlloc(io, resolved, allocator, .limited(max_doc_bytes));
        owned_text = contents;
        break :blk contents;
    };
    defer if (owned_text) |owned| allocator.free(owned);

    try client.didOpen(.{
        .uri = uri,
        .text = text,
    });

    return uri;
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi lsp <subcommand>", "")
        .description("ZLS LSP client utilities (0-based line/character positions).")
        .section("Subcommands")
        .subcommand(.{ .name = "request", .description = "Send an LSP request" })
        .subcommand(.{ .name = "notify", .description = "Send an LSP notification" })
        .subcommand(.{ .name = "hover", .description = "Get hover info at position" })
        .subcommand(.{ .name = "completion", .description = "Get completion items at position" })
        .subcommand(.{ .name = "definition", .description = "Get definition at position" })
        .subcommand(.{ .name = "references", .description = "Find references at position" })
        .subcommand(.{ .name = "rename", .description = "Rename symbol at position" })
        .subcommand(.{ .name = "format", .description = "Format document and return edits" })
        .subcommand(.{ .name = "diagnostics", .description = "Fetch document diagnostics" })
        .newline()
        .section("Common Options")
        .option(.{ .long = "--zls", .arg = "PATH", .description = "ZLS binary path" })
        .option(.{ .long = "--zig", .arg = "PATH", .description = "Zig compiler path for ZLS" })
        .option(.{ .long = "--root", .arg = "PATH", .description = "Workspace root override" })
        .option(.{ .long = "--log-level", .arg = "LEVEL", .description = "ZLS log level" })
        .option(.{ .long = "--no-snippets", .description = "Disable completion snippets" })
        .option(common_options.help)
        .newline()
        .section("Examples")
        .example("abi lsp hover --path src/main.zig --line 0 --character 0", "Hover")
        .example("abi lsp completion --path src/main.zig --line 3 --character 10", "Completion")
        .example("abi lsp request --method textDocument/definition --params '{...}'", "Raw request");

    builder.print();
}

test {
    std.testing.refAllDecls(@This());
}
