//! Explore codebase command.

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const utils = @import("../utils/mod.zig");

pub const meta: command_mod.Meta = .{
    .name = "explore",
    .description = "Search and explore codebase",
};

const DebugWriter = struct {
    pub fn print(_: @This(), comptime fmt: []const u8, args: anytype) !void {
        std.debug.print(fmt, args);
    }
};

/// Run the explore command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);

    if (parser.wantsHelp()) {
        printHelp(allocator);
        return;
    }

    // Check if AI explore feature is enabled
    if (!abi.ai.explore.isEnabled()) {
        utils.output.printError("AI code exploration feature is disabled.", .{});
        utils.output.printInfo("Rebuild with: zig build -Denable-ai=true -Denable-explore=true", .{});
        return;
    }

    if (!parser.hasMore()) {
        utils.output.printError("No search query provided.", .{});
        printHelp(allocator);
        return;
    }

    var query: ?[]const u8 = null;
    var root_path: []const u8 = ".";
    var level: abi.ai.explore.ExploreLevel = .medium;
    var output_format: abi.ai.explore.OutputFormat = .human;
    var include_patterns = std.ArrayListUnmanaged([]const u8).empty;
    var exclude_patterns = std.ArrayListUnmanaged([]const u8).empty;
    var case_sensitive = false;
    var use_regex = false;
    var max_files: usize = 0;
    var max_depth: usize = 0;
    var timeout_ms: u64 = 0;

    while (parser.hasMore()) {
        if (parser.consumeOption(&[_][]const u8{ "--level", "-l" })) |val| {
            level = switch (std.ascii.eqlIgnoreCase(val, "quick")) {
                true => .quick,
                else => switch (std.ascii.eqlIgnoreCase(val, "medium")) {
                    true => .medium,
                    else => switch (std.ascii.eqlIgnoreCase(val, "thorough")) {
                        true => .thorough,
                        else => switch (std.ascii.eqlIgnoreCase(val, "deep")) {
                            true => .deep,
                            else => {
                                utils.output.printError("Unknown level: {s}. Use: quick, medium, thorough, deep", .{val});
                                return;
                            },
                        },
                    },
                },
            };
        } else if (parser.consumeOption(&[_][]const u8{ "--format", "-f" })) |val| {
            output_format = switch (std.ascii.eqlIgnoreCase(val, "json")) {
                true => .json,
                else => switch (std.ascii.eqlIgnoreCase(val, "compact")) {
                    true => .compact,
                    else => switch (std.ascii.eqlIgnoreCase(val, "yaml")) {
                        true => .yaml,
                        else => .human,
                    },
                },
            };
        } else if (parser.consumeOption(&[_][]const u8{ "--include", "-i" })) |val| {
            try include_patterns.append(allocator, val);
        } else if (parser.consumeOption(&[_][]const u8{ "--exclude", "-e" })) |val| {
            try exclude_patterns.append(allocator, val);
        } else if (parser.consumeFlag(&[_][]const u8{ "--case-sensitive", "-c" })) {
            case_sensitive = true;
        } else if (parser.consumeFlag(&[_][]const u8{ "--regex", "-r" })) {
            use_regex = true;
        } else if (parser.consumeOption(&[_][]const u8{"--max-files"})) |val| {
            max_files = std.fmt.parseInt(usize, val, 10) catch 0;
        } else if (parser.consumeOption(&[_][]const u8{"--max-depth"})) |val| {
            max_depth = std.fmt.parseInt(usize, val, 10) catch 0;
        } else if (parser.consumeOption(&[_][]const u8{"--timeout"})) |val| {
            timeout_ms = std.fmt.parseInt(u64, val, 10) catch 0;
        } else if (parser.consumeOption(&[_][]const u8{"--path"})) |val| {
            root_path = val;
        } else if (query == null) {
            query = parser.next();
        } else {
            utils.output.printError("Unknown argument: {s}", .{parser.next().?});
            printHelp(allocator);
            return;
        }
    }

    const search_query = query orelse {
        utils.output.printError("No search query provided.", .{});
        printHelp(allocator);
        return;
    };

    var config = abi.ai.explore.ExploreConfig.defaultForLevel(level);
    config.output_format = output_format;
    config.case_sensitive = case_sensitive;
    config.use_regex = use_regex;

    if (include_patterns.items.len > 0) {
        config.include_patterns = include_patterns.items;
    }
    if (exclude_patterns.items.len > 0) {
        config.exclude_patterns = exclude_patterns.items;
    }
    if (max_files > 0) config.max_files = max_files;
    if (max_depth > 0) config.max_depth = max_depth;
    if (timeout_ms > 0) config.timeout_ms = timeout_ms;

    var agent = try abi.ai.explore.ExploreAgent.init(allocator, config);
    defer agent.deinit();

    var timer = abi.shared.time.Timer.start() catch {
        utils.output.printError("Timer unavailable on this platform", .{});
        return;
    };
    var result = try agent.explore(root_path, search_query);
    defer result.deinit();

    const duration_ms = @divTrunc(timer.read(), std.time.ns_per_ms);

    switch (output_format) {
        .human => {
            try result.formatHuman(DebugWriter{});
        },
        .json => {
            try result.formatJSON(DebugWriter{});
        },
        .compact => {
            std.debug.print("Query: \"{s}\" | Found: {d} matches in {d}ms\n", .{
                search_query, result.matches_found, duration_ms,
            });
        },
        .yaml => {
            std.debug.print("query: \"{s}\"\n", .{search_query});
            std.debug.print("level: {t}\n", .{level});
            std.debug.print("matches_found: {d}\n", .{result.matches_found});
            std.debug.print("duration_ms: {d}\n", .{duration_ms});
        },
    }

    include_patterns.deinit(allocator);
    exclude_patterns.deinit(allocator);
}

fn printHelp(allocator: std.mem.Allocator) void {
    var builder = utils.help.HelpBuilder.init(allocator);
    defer builder.deinit();

    _ = builder
        .usage("abi explore", "[options] <query>")
        .description("Search and explore the codebase for patterns.")
        .section("Arguments")
        .text("  <query>              Search pattern or natural language query\n\n")
        .section("Options")
        .option(.{ .short = "-l", .long = "--level", .arg = "level", .description = "Exploration depth: quick, medium, thorough, deep (default: medium)" })
        .option(.{ .short = "-f", .long = "--format", .arg = "fmt", .description = "Output format: human, json, compact, yaml (default: human)" })
        .option(.{ .short = "-i", .long = "--include", .arg = "pat", .description = "Include files matching pattern (repeatable)" })
        .option(.{ .short = "-e", .long = "--exclude", .arg = "pat", .description = "Exclude files matching pattern (repeatable)" })
        .option(.{ .short = "-c", .long = "--case-sensitive", .description = "Match case sensitively" })
        .option(.{ .short = "-r", .long = "--regex", .description = "Treat query as regex pattern" })
        .option(.{ .long = "--path", .arg = "path", .description = "Root directory to search (default: .)" })
        .option(.{ .long = "--max-files", .arg = "n", .description = "Maximum files to scan" })
        .option(.{ .long = "--max-depth", .arg = "n", .description = "Maximum directory depth" })
        .option(.{ .long = "--timeout", .arg = "ms", .description = "Timeout in milliseconds" })
        .option(utils.help.common_options.help)
        .newline()
        .section("Examples")
        .example("abi explore \"HTTP handler\"", "")
        .example("abi explore -l thorough \"FIXME\"", "")
        .example("abi explore -f json \"function_name\"", "")
        .example("abi explore -i \"*.zig\" \"pub fn\"", "")
        .example("abi explore --regex \"fn\\s+\\w+\"", "");

    builder.print();
}
