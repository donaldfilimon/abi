const std = @import("std");
const wdbx = @import("wdbx");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "config",
    .summary = "Inspect and modify ABI configuration files",
    .usage = "abi config [set|get|list|validate|show] [options]",
    .details = "  --file <path>   Specify configuration file\n" ++
        "  set <key> <value>   Update a configuration entry\n" ++
        "  get <key>           Read a configuration entry\n" ++
        "  list                Print full configuration\n" ++
        "  --validate          Validate configuration schema\n" ++
        "  --summary/--no-summary  Toggle summary output\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    var config_path: ?[]const u8 = null;
    var do_validate = false;
    var do_summary = true;
    var action: ?[]const u8 = null;
    var key: ?[]const u8 = null;
    var value: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--file") and i + 1 < args.len) {
            i += 1;
            config_path = args[i];
        } else if (std.mem.eql(u8, arg, "--validate")) {
            do_validate = true;
        } else if (std.mem.eql(u8, arg, "--summary")) {
            do_summary = true;
        } else if (std.mem.eql(u8, arg, "--no-summary")) {
            do_summary = false;
        } else if (std.mem.eql(u8, arg, "show")) {
            do_summary = true;
        } else if (std.mem.eql(u8, arg, "validate")) {
            do_validate = true;
        } else if (std.mem.eql(u8, arg, "set") or std.mem.eql(u8, arg, "get") or std.mem.eql(u8, arg, "list")) {
            action = arg;
            if (std.mem.eql(u8, arg, "set") and i + 2 < args.len) {
                i += 1;
                key = args[i];
                i += 1;
                value = args[i];
            } else if (std.mem.eql(u8, arg, "get") and i + 1 < args.len) {
                i += 1;
                key = args[i];
            }
        } else if (common.isHelpToken(arg)) {
            std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
            return;
        }
    }

    var manager = try wdbx.ConfigManager.init(allocator, config_path);
    defer manager.deinit();

    if (action) |act| {
        if (std.mem.eql(u8, act, "set")) {
            if (key == null or value == null) {
                std.debug.print("config set requires <key> and <value>\n", .{});
                return;
            }
            try manager.setValue(key.?, value.?);
            try manager.save();
            std.debug.print("Set {s}={s}\n", .{ key.?, value.? });
            return;
        } else if (std.mem.eql(u8, act, "get")) {
            if (key == null) {
                std.debug.print("config get requires <key>\n", .{});
                return;
            }
            if (try manager.getValue(key.?)) |val| {
                defer allocator.free(val);
                std.debug.print("{s}={s}\n", .{ key.?, val });
            } else {
                std.debug.print("Key '{s}' not found\n", .{key.?});
            }
            return;
        } else if (std.mem.eql(u8, act, "list")) {
            const list = try manager.listAll(allocator);
            defer allocator.free(list);
            std.debug.print("{s}", .{list});
            return;
        }
    }

    if (do_validate) {
        manager.validate() catch |err| {
            std.debug.print("Config validation failed: {any}\n", .{err});
            return err;
        };
        std.debug.print("Config validation: OK\n", .{});
    }

    if (do_summary) {
        const cfg = manager.getConfig();
        std.debug.print("\nLoaded configuration from: {s}\n", .{manager.config_path});
        wdbx.ConfigUtils.printSummary(cfg);
    }
}
