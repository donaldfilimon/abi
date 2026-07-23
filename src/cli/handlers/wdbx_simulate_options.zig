//! Argument/config parsing for `abi wdbx simulate`.
const std = @import("std");
const features = @import("abi").features;
const wdbx = features.wdbx;
const format = @import("wdbx_simulate_format.zig");
const diag = format.diag;

pub const MAX_RULE_FILE_BYTES = 1 * 1024 * 1024;
pub const MAX_CONFIG_FILE_BYTES = 1 * 1024 * 1024;
pub const MAX_RESUME_FILE_BYTES = 256 * 1024 * 1024;

pub const Options = struct {
    initial: std.ArrayListUnmanaged([]const u8) = .empty,
    rules: std.ArrayListUnmanaged(wdbx.multiway.Rule) = .empty,
    max_depth: ?u32 = null,
    max_states: ?u32 = null,
    max_events: ?u32 = null,
    max_payload: ?u32 = null,
    max_duration_ms: ?u64 = null,
    max_memory_bytes: ?u64 = null,
    seed: ?u64 = null,
    workers: ?u32 = null,
    format: enum { summary, json, dot } = .summary,
    output: ?[]const u8 = null,
    store: ?[]const u8 = null,
    resume_file: ?[]const u8 = null,
    resume_wdbx: ?[]const u8 = null,
    dry_run: bool = false,
    quiet: bool = false,
    verbose: bool = false,
};

pub fn parseRuleLine(allocator: std.mem.Allocator, opts: *Options, text: []const u8, origin: []const u8) !?u8 {
    const rule = wdbx.multiway.parseRule(allocator, text) catch |err| switch (err) {
        error.MissingArrow => return diag("invalid rule '{s}' ({s}): expected 'LHS->RHS'", .{ text, origin }),
        error.EmptyLhs => return diag("invalid rule '{s}' ({s}): left-hand side must be non-empty", .{ text, origin }),
        error.OutOfMemory => return error.OutOfMemory,
    };
    try opts.rules.append(allocator, rule);
    return null;
}

pub fn loadRulesFile(io: std.Io, allocator: std.mem.Allocator, opts: *Options, path: []const u8) !?u8 {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_RULE_FILE_BYTES)) catch |err| {
        return diag("cannot read rules file '{s}': {s}", .{ path, @errorName(err) });
    };
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        if (try parseRuleLine(allocator, opts, line, path)) |code| return code;
    }
    return null;
}

fn jsonUint(value: std.json.Value) ?u64 {
    return switch (value) {
        .integer => |n| if (n < 0) null else @intCast(n),
        else => null,
    };
}

pub fn loadConfigFile(io: std.Io, allocator: std.mem.Allocator, opts: *Options, path: []const u8) !?u8 {
    const content = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(MAX_CONFIG_FILE_BYTES)) catch |err| {
        return diag("cannot read config file '{s}': {s}", .{ path, @errorName(err) });
    };
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, allocator, content, .{}) catch {
        return diag("config file '{s}' is not valid JSON", .{path});
    };
    const root = switch (parsed) {
        .object => |obj| obj,
        else => return diag("config file '{s}': top level must be a JSON object", .{path}),
    };
    if (root.get("initial")) |value| {
        const arr = switch (value) {
            .array => |a| a,
            else => return diag("config '{s}': \"initial\" must be an array of strings", .{path}),
        };
        for (arr.items) |item| {
            const text = switch (item) {
                .string => |s| s,
                else => return diag("config '{s}': \"initial\" entries must be strings", .{path}),
            };
            try opts.initial.append(allocator, try allocator.dupe(u8, text));
        }
    }
    if (root.get("rules")) |value| {
        const arr = switch (value) {
            .array => |a| a,
            else => return diag("config '{s}': \"rules\" must be an array", .{path}),
        };
        for (arr.items) |item| {
            switch (item) {
                .string => |text| {
                    if (try parseRuleLine(allocator, opts, text, path)) |code| return code;
                },
                .object => |obj| {
                    const text = switch (obj.get("rule") orelse return diag("config '{s}': rule object missing \"rule\"", .{path})) {
                        .string => |s| s,
                        else => return diag("config '{s}': rule object \"rule\" must be a string", .{path}),
                    };
                    if (try parseRuleLine(allocator, opts, text, path)) |code| return code;
                    const rule = &opts.rules.items[opts.rules.items.len - 1];
                    if (obj.get("weight")) |weight_value| {
                        rule.weight = switch (weight_value) {
                            .float => |f| f,
                            .integer => |n| @floatFromInt(n),
                            else => return diag("config '{s}': rule \"weight\" must be a number", .{path}),
                        };
                    }
                    if (obj.get("family")) |family_value| {
                        rule.family = switch (family_value) {
                            .string => |s| try allocator.dupe(u8, s),
                            else => return diag("config '{s}': rule \"family\" must be a string", .{path}),
                        };
                    }
                },
                else => return diag("config '{s}': \"rules\" entries must be strings or objects", .{path}),
            }
        }
    }
    const uint_fields = .{
        .{ "max_depth", "max_depth" },
        .{ "max_states", "max_states" },
        .{ "max_events", "max_events" },
        .{ "max_payload", "max_payload" },
    };
    inline for (uint_fields) |field| {
        if (root.get(field[0])) |value| {
            const n = jsonUint(value) orelse return diag("config '{s}': \"{s}\" must be a non-negative integer", .{ path, field[0] });
            if (n > std.math.maxInt(u32)) return diag("config '{s}': \"{s}\" too large", .{ path, field[0] });
            @field(opts, field[1]) = @intCast(n);
        }
    }
    if (root.get("max_duration_ms")) |value| {
        opts.max_duration_ms = jsonUint(value) orelse return diag("config '{s}': \"max_duration_ms\" must be a non-negative integer", .{path});
    }
    if (root.get("max_memory_bytes")) |value| {
        opts.max_memory_bytes = jsonUint(value) orelse return diag("config '{s}': \"max_memory_bytes\" must be a non-negative integer", .{path});
    }
    if (root.get("seed")) |value| {
        opts.seed = jsonUint(value) orelse return diag("config '{s}': \"seed\" must be a non-negative integer", .{path});
    }
    if (root.get("workers")) |value| {
        const n = jsonUint(value) orelse return diag("config '{s}': \"workers\" must be a non-negative integer", .{path});
        if (n == 0 or n > std.math.maxInt(u32)) return diag("config '{s}': \"workers\" out of range", .{path});
        opts.workers = @intCast(n);
    }
    return null;
}

pub fn parseUintFlag(comptime T: type, args: []const []const u8, index: *usize, flag: []const u8) !?T {
    if (index.* + 1 >= args.len) {
        _ = diag("{s} requires a value", .{flag});
        return error.Reported;
    }
    index.* += 1;
    return std.fmt.parseInt(T, args[index.*], 10) catch {
        _ = diag("{s}: '{s}' is not a valid non-negative integer", .{ flag, args[index.*] });
        return error.Reported;
    };
}

pub fn stringFlag(args: []const []const u8, index: *usize, flag: []const u8) ![]const u8 {
    if (index.* + 1 >= args.len) {
        _ = diag("{s} requires a value", .{flag});
        return error.Reported;
    }
    index.* += 1;
    return args[index.*];
}

test {
    std.testing.refAllDecls(@This());
}
