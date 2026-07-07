//! CLI test aggregator — gives the `src/cli` framework a real test artifact.
//!
//! The inline tests in `arg.zig`, `registry.zig`, and `dispatch.zig` are not
//! reachable from any other test target (the CLI ships as an executable, not a
//! module), so they would never run. This aggregator pulls them in via
//! `refAllDecls` and adds focused coverage that exercises the *actual* argument
//! specs that `dispatch` walks, proving the migrated `complete`/`train` parsing
//! matches the historical hand-written parser.

const std = @import("std");
const registry = @import("cli/registry.zig");
const arg = @import("cli/arg.zig");
const dispatch = @import("cli/dispatch.zig");

fn specFor(comptime name: []const u8) []const arg.Arg {
    return comptime blk: {
        for (registry.commands) |command| {
            if (std.mem.eql(u8, command.name, name)) break :blk command.args;
        }
        @compileError("no registry command named '" ++ name ++ "'");
    };
}

fn subcommandFor(comptime command_name: []const u8, comptime subcommand_name: []const u8) registry.Command {
    return comptime blk: {
        for (registry.commands) |command| {
            if (!std.mem.eql(u8, command.name, command_name)) continue;
            for (command.subcommands) |subcommand| {
                if (std.mem.eql(u8, subcommand.name, subcommand_name)) break :blk subcommand;
            }
        }
        @compileError("no registry subcommand named '" ++ command_name ++ " " ++ subcommand_name ++ "'");
    };
}

test "registry `complete` spec parses like the legacy parseCompleteArgs" {
    const spec = specFor("complete");

    // Order-independent flags + value + positional.
    var all = try arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--model", "fable-5", "--live", "--learn", "hi" });
    defer all.deinit();
    try std.testing.expect(all.flag("live"));
    try std.testing.expect(all.flag("learn"));
    try std.testing.expect(!all.flag("confirm"));
    try std.testing.expectEqualStrings("fable-5", all.value("model").?);
    try std.testing.expectEqualStrings("hi", all.value("input").?);

    // Plain positional, flags default false.
    var plain = try arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "hello" });
    defer plain.deinit();
    try std.testing.expect(!plain.flag("learn"));
    try std.testing.expect(plain.value("model") == null);
    try std.testing.expectEqualStrings("hello", plain.value("input").?);

    // Missing input and a dangling `--model` are usage errors (→ exit 2).
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--learn" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--model" }));
}

test "registry `train` spec requires exactly one positional" {
    const spec = specFor("train");

    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "train", "hello" });
    defer ok.deinit();
    try std.testing.expectEqualStrings("hello", ok.value("input").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "train" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "train", "a", "b" }));
}

test "registry `agent` subcommands preserve focused grammar" {
    const plan = subcommandFor("agent", "plan");
    var plan_ok = try arg.parseFrom(std.testing.allocator, plan.args, &.{ "abi", "agent", "plan", "inspect" }, 3);
    defer plan_ok.deinit();
    try std.testing.expectEqualStrings("inspect", plan_ok.value("input").?);
    try std.testing.expectError(error.Usage, arg.parseFrom(std.testing.allocator, plan.args, &.{ "abi", "agent", "plan" }, 3));
    try std.testing.expectError(error.Usage, arg.parseFrom(std.testing.allocator, plan.args, &.{ "abi", "agent", "plan", "a", "b" }, 3));

    const train = subcommandFor("agent", "train");
    var train_ok = try arg.parseFrom(std.testing.allocator, train.args, &.{ "abi", "agent", "train", "abi" }, 3);
    defer train_ok.deinit();
    try std.testing.expectEqualStrings("abi", train_ok.value("profile").?);
    try std.testing.expectError(error.Usage, arg.parseFrom(std.testing.allocator, train.args, &.{ "abi", "agent", "train" }, 3));
    try std.testing.expectError(error.Usage, arg.parseFrom(std.testing.allocator, train.args, &.{ "abi", "agent", "train", "bogus" }, 3));

    const tui = subcommandFor("agent", "tui");
    var tui_ok = try arg.parseFrom(std.testing.allocator, tui.args, &.{ "abi", "agent", "tui" }, 3);
    defer tui_ok.deinit();
    try std.testing.expectEqual(@as(usize, 0), tui.args.len);
    try std.testing.expectError(error.Usage, arg.parseFrom(std.testing.allocator, tui.args, &.{ "abi", "agent", "tui", "extra" }, 3));

    const os = subcommandFor("agent", "os");
    try std.testing.expect(os.raw_handler != null);
}

test "registry `plugin` spec preserves list and run grammar" {
    const spec = specFor("plugin");

    var list = try arg.parse(std.testing.allocator, spec, &.{ "abi", "plugin", "list" });
    defer list.deinit();
    try std.testing.expectEqualStrings("list", list.value("command").?);
    try std.testing.expect(list.value("name") == null);
    try std.testing.expect(list.value("input") == null);

    var run = try arg.parse(std.testing.allocator, spec, &.{ "abi", "plugin", "run", "example-plugin", "hello", "plugin" });
    defer run.deinit();
    try std.testing.expectEqualStrings("run", run.value("command").?);
    try std.testing.expectEqualStrings("example-plugin", run.value("name").?);
    try std.testing.expectEqualStrings("hello plugin", run.value("input").?);

    var dashed = try arg.parse(std.testing.allocator, spec, &.{ "abi", "plugin", "run", "example-plugin", "--flag-like-input" });
    defer dashed.deinit();
    try std.testing.expectEqualStrings("--flag-like-input", dashed.value("input").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "plugin" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "plugin", "bogus" }));
}

test "registry `auth` spec preserves subcommand grammar" {
    const spec = specFor("auth");

    var status = try arg.parse(std.testing.allocator, spec, &.{ "abi", "auth", "status" });
    defer status.deinit();
    try std.testing.expectEqualStrings("status", status.value("command").?);
    try std.testing.expect(status.value("service") == null);

    var logout = try arg.parse(std.testing.allocator, spec, &.{ "abi", "auth", "logout" });
    defer logout.deinit();
    try std.testing.expectEqualStrings("logout", logout.value("command").?);
    try std.testing.expect(logout.value("service") == null);

    var signin = try arg.parse(std.testing.allocator, spec, &.{ "abi", "auth", "signin", "openai" });
    defer signin.deinit();
    try std.testing.expectEqualStrings("signin", signin.value("command").?);
    try std.testing.expectEqualStrings("openai", signin.value("service").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "auth" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "auth", "bogus" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "auth", "signin", "openai", "extra" }));
}

test "registry `nn` spec preserves train and sample grammar" {
    const spec = specFor("nn");

    var train_inline = try arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "train", "hello hello" });
    defer train_inline.deinit();
    try std.testing.expectEqualStrings("train", train_inline.value("command").?);
    try std.testing.expectEqualStrings("hello hello", train_inline.value("input").?);

    var train_jsonl = try arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "train", "--jsonl", "data.jsonl", "--field", "body" });
    defer train_jsonl.deinit();
    try std.testing.expectEqualStrings("data.jsonl", train_jsonl.value("jsonl").?);
    try std.testing.expectEqualStrings("body", train_jsonl.value("field").?);

    var sample = try arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "sample", "--text", "hello", "--seed", "h", "--n", "8" });
    defer sample.deinit();
    try std.testing.expectEqualStrings("sample", sample.value("command").?);
    try std.testing.expectEqualStrings("hello", sample.value("text").?);
    try std.testing.expectEqualStrings("h", sample.value("seed").?);
    try std.testing.expectEqual(@as(u64, 8), sample.uint("n").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "nn" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "bogus" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "sample", "--n", "nope" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "nn", "train", "hello", "extra" }));
}

test "registry `wdbx` advertises raw subcommands without taking over execution" {
    const command = comptime blk: {
        for (registry.commands) |candidate| {
            if (std.mem.eql(u8, candidate.name, "wdbx")) break :blk candidate;
        }
        @compileError("missing wdbx command");
    };

    try std.testing.expect(command.raw_handler != null);
    try std.testing.expect(command.handler == null);
    try std.testing.expectEqual(@as(usize, 9), command.subcommands.len);
    try std.testing.expectEqualStrings("db", command.subcommands[0].name);
    try std.testing.expectEqualStrings("api", command.subcommands[8].name);
}

test "argument-free commands reject stray tokens" {
    const spec = specFor("backends");
    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "backends" });
    defer ok.deinit();
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "backends", "extra" }));
}

test "registry `dashboard` and `tui` specs accept initial pane selection" {
    const dashboard = specFor("dashboard");
    var dashboard_ok = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--pane", "memory" });
    defer dashboard_ok.deinit();
    try std.testing.expectEqualStrings("memory", dashboard_ok.value("pane").?);

    var dashboard_numeric = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--pane", "5" });
    defer dashboard_numeric.deinit();
    try std.testing.expectEqualStrings("5", dashboard_numeric.value("pane").?);

    const tui = specFor("tui");
    var tui_ok = try arg.parse(std.testing.allocator, tui, &.{ "abi", "tui", "--pane", "wdbx" });
    defer tui_ok.deinit();
    try std.testing.expectEqualStrings("wdbx", tui_ok.value("pane").?);

    var plain = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--plain" });
    defer plain.deinit();
    try std.testing.expect(plain.flag("plain"));
    try std.testing.expect(!plain.flag("no-color"));

    var no_color = try arg.parse(std.testing.allocator, tui, &.{ "abi", "tui", "--no-color", "--pane", "scheduler" });
    defer no_color.deinit();
    try std.testing.expect(no_color.flag("no-color"));
    try std.testing.expectEqualStrings("scheduler", no_color.value("pane").?);

    var compact = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--compact", "--pane", "scheduler" });
    defer compact.deinit();
    try std.testing.expect(compact.flag("compact"));
    try std.testing.expectEqualStrings("scheduler", compact.value("pane").?);

    var once_interval = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--once", "--interval", "250" });
    defer once_interval.deinit();
    try std.testing.expect(once_interval.flag("once"));
    try std.testing.expectEqual(@as(u64, 250), once_interval.uint("interval").?);

    var json = try arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--json", "--pane", "plugins" });
    defer json.deinit();
    try std.testing.expect(json.flag("json"));
    try std.testing.expectEqualStrings("plugins", json.value("pane").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--pane" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--pane", "bogus" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "extra" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, dashboard, &.{ "abi", "dashboard", "--interval", "nope" }));
}

test "registry `scheduler` spec requires status subcommand" {
    const spec = specFor("scheduler");

    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler", "status" });
    defer ok.deinit();
    try std.testing.expectEqualStrings("status", ok.value("command").?);
    try std.testing.expectEqual(@as(usize, 1), spec[0].choices.len);
    try std.testing.expectEqualStrings("status", spec[0].choices[0]);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler", "bogus" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler", "status", "extra" }));
}

test "registry `twilio` spec requires simulate input" {
    const spec = specFor("twilio");

    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "simulate", "hello" });
    defer ok.deinit();
    try std.testing.expectEqualStrings("simulate", ok.value("command").?);
    try std.testing.expectEqualStrings("hello", ok.value("input").?);
    try std.testing.expectEqual(@as(usize, 1), spec[0].choices.len);
    try std.testing.expectEqualStrings("simulate", spec[0].choices[0]);

    var dashed = try arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "simulate", "--", "--literal" });
    defer dashed.deinit();
    try std.testing.expectEqualStrings("--literal", dashed.value("input").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "simulate" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "bogus", "hello" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "simulate", "a", "b" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "twilio", "simulate", "--literal" }));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(dispatch);
}
