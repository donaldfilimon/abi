//! Explore codebase command.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");

/// Run the explore command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    var query: ?[]const u8 = null;
    var root_path: []const u8 = ".";
    var level: abi.ai.explore.ExploreLevel = .medium;
    var output_format: abi.ai.explore.OutputFormat = .human;
    var include_patterns = std.ArrayListUnmanaged([]const u8){};
    var exclude_patterns = std.ArrayListUnmanaged([]const u8){};
    var case_sensitive = false;
    var use_regex = false;
    var max_files: usize = undefined;
    var max_depth: usize = undefined;
    var timeout_ms: u64 = undefined;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h" })) {
            printHelp();
            return;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--level", "-l" })) {
            if (i < args.len) {
                const level_str = std.mem.sliceTo(args[i], 0);
                level = switch (std.ascii.eqlIgnoreCase(level_str, "quick")) {
                    true => .quick,
                    else => switch (std.ascii.eqlIgnoreCase(level_str, "medium")) {
                        true => .medium,
                        else => switch (std.ascii.eqlIgnoreCase(level_str, "thorough")) {
                            true => .thorough,
                            else => switch (std.ascii.eqlIgnoreCase(level_str, "deep")) {
                                true => .deep,
                                else => {
                                    std.debug.print("Unknown level: {s}. Use: quick, medium, thorough, deep\n", .{level_str});
                                    return;
                                },
                            },
                        },
                    },
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--format", "-f" })) {
            if (i < args.len) {
                const format_str = std.mem.sliceTo(args[i], 0);
                output_format = switch (std.ascii.eqlIgnoreCase(format_str, "json")) {
                    true => .json,
                    else => switch (std.ascii.eqlIgnoreCase(format_str, "compact")) {
                        true => .compact,
                        else => switch (std.ascii.eqlIgnoreCase(format_str, "yaml")) {
                            true => .yaml,
                            else => .human,
                        },
                    },
                };
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--include", "-i" })) {
            if (i < args.len) {
                try include_patterns.append(allocator, std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--exclude", "-e" })) {
            if (i < args.len) {
                try exclude_patterns.append(allocator, std.mem.sliceTo(args[i], 0));
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--case-sensitive", "-c" })) {
            case_sensitive = true;
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--regex", "-r" })) {
            use_regex = true;
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{"--max-files"})) {
            if (i < args.len) {
                max_files = try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{"--max-depth"})) {
            if (i < args.len) {
                max_depth = try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{"--timeout"})) {
            if (i < args.len) {
                timeout_ms = try std.fmt.parseInt(u64, std.mem.sliceTo(args[i], 0), 10);
                i += 1;
            }
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--path")) {
            if (i < args.len) {
                root_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (query == null) {
            query = std.mem.sliceTo(arg, 0);
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            printHelp();
            return;
        }
    }

    const search_query = query orelse {
        std.debug.print("Error: No search query provided.\n", .{});
        printHelp();
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

    const start_time = try std.time.Instant.now();
    var result = try agent.explore(root_path, search_query);
    defer result.deinit();

    const end_time = try std.time.Instant.now();
    const duration_ms = @divTrunc(end_time.since(start_time), std.time.ns_per_ms);

    switch (output_format) {
        .human => {
            result.formatHuman(std.debug);
        },
        .json => {
            result.formatJSON(std.debug);
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

fn printHelp() void {
    const help_text =
        "Usage: abi explore [options] <query>\n\n" ++
        "Search and explore the codebase for patterns.\n\n" ++
        "Arguments:\n" ++
        "  <query>              Search pattern or natural language query\n\n" ++
        "Options:\n" ++
        "  -l, --level <level>  Exploration depth: quick, medium, thorough, deep (default: medium)\n" ++
        "  -f, --format <fmt>   Output format: human, json, compact, yaml (default: human)\n" ++
        "  -i, --include <pat>  Include files matching pattern (can be used multiple times)\n" ++
        "  -e, --exclude <pat>  Exclude files matching pattern (can be used multiple times)\n" ++
        "  -c, --case-sensitive Match case sensitively\n" ++
        "  -r, --regex          Treat query as regex pattern\n" ++
        "  --path <path>        Root directory to search (default: .)\n" ++
        "  --max-files <n>      Maximum files to scan\n" ++
        "  --max-depth <n>      Maximum directory depth\n" ++
        "  --timeout <ms>       Timeout in milliseconds\n" ++
        "  -h, --help           Show this help message\n\n" ++
        "Examples:\n" ++
        "  abi explore \"HTTP handler\"\n" ++
        "  abi explore -l thorough \"FIXME\"\n" ++
        "  abi explore -f json \"function_name\"\n" ++
        "  abi explore -i \"*.zig\" \"pub fn\"\n" ++
        "  abi explore --regex \"fn\\s+\\w+\"";
    std.debug.print("{s}\n", .{help_text});
}
