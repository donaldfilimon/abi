const std = @import("std");
const build_options = @import("build_options");
const features = @import("../../features/mod.zig");

const nn = features.nn;

/// `abi nn <train|sample> ...` — a miniature character-level demo trainer.
///
/// Drives the pure-Zig `features.nn` char-LM: a tiny next-character model trained
/// by hand-derived backprop over a small in-memory corpus. This is a
/// demonstration trainer for the local pipeline, NOT a production or
/// large-language-model trainer, and it performs no network or distributed work.
///
/// Gated on `build_options.feat_nn`; with the feature compiled out the handler
/// prints the standard feature-disabled line and exits non-zero. The feature
/// path lives in private helpers so it is never analyzed in a `-Dfeat-nn=false`
/// build (where `nn` is the stub).
pub fn handleNn(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    if (build_options.feat_nn) {
        return run(allocator, args);
    } else {
        std.debug.print("nn feature is disabled in this build (build with -Dfeat-nn=true)\n", .{});
        return 1;
    }
}

fn usage() u8 {
    std.debug.print(
        \\abi nn <command> ...   (miniature character-level demo trainer)
        \\
        \\  train "<text>"                                  Train on an inline text corpus
        \\  train --jsonl <path> [--field <name>]           Train on a JSONL dataset (default field "text")
        \\  sample --text "<corpus>" --seed <char> --n <k>  Train on <corpus>, then greedily emit k chars
        \\
        \\This is a demonstration char-level trainer, not a production/LLM/distributed trainer.
        \\
    , .{});
    return 2;
}

fn run(allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    if (args.len < 3) return usage();
    const sub = args[2];
    if (std.mem.eql(u8, sub, "train")) return trainCmd(allocator, args);
    if (std.mem.eql(u8, sub, "sample")) return sampleCmd(allocator, args);
    return usage();
}

fn printReport(report: nn.TrainReport) void {
    std.debug.print(
        "nn train: initial_loss={d:.4} final_loss={d:.4} improved={s} steps={d}\n",
        .{ report.initial_loss, report.final_loss, if (report.improved) "true" else "false", report.steps },
    );
}

fn trainCmd(allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    var jsonl_path: ?[]const u8 = null;
    var field: []const u8 = "text";
    var inline_text: ?[]const u8 = null;

    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const tok = args[i];
        if (std.mem.eql(u8, tok, "--jsonl")) {
            i += 1;
            if (i >= args.len) return usage();
            jsonl_path = args[i];
        } else if (std.mem.eql(u8, tok, "--field")) {
            i += 1;
            if (i >= args.len) return usage();
            field = args[i];
        } else {
            if (inline_text != null) return usage();
            inline_text = tok;
        }
    }

    if (jsonl_path != null and inline_text != null) return usage();

    const report = blk: {
        if (jsonl_path) |path| {
            break :blk nn.trainOnJsonl(allocator, path, field, .{}) catch |err| {
                std.debug.print("error: nn train --jsonl failed: {s}\n", .{@errorName(err)});
                return 1;
            };
        } else if (inline_text) |text| {
            break :blk nn.trainOnText(allocator, text, .{}) catch |err| {
                std.debug.print("error: nn train failed: {s}\n", .{@errorName(err)});
                return 1;
            };
        } else {
            return usage();
        }
    };

    printReport(report);
    return 0;
}

fn sampleCmd(allocator: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    var text: ?[]const u8 = null;
    var seed: u8 = 0;
    var seed_set = false;
    var n: usize = 16;

    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const tok = args[i];
        if (std.mem.eql(u8, tok, "--text")) {
            i += 1;
            if (i >= args.len) return usage();
            text = args[i];
        } else if (std.mem.eql(u8, tok, "--seed")) {
            i += 1;
            if (i >= args.len or args[i].len == 0) return usage();
            seed = args[i][0];
            seed_set = true;
        } else if (std.mem.eql(u8, tok, "--n")) {
            i += 1;
            if (i >= args.len) return usage();
            n = std.fmt.parseInt(usize, args[i], 10) catch return usage();
        } else {
            return usage();
        }
    }

    const corpus = text orelse return usage();
    if (!seed_set) return usage();

    var model = nn.trainModel(allocator, corpus, .{}) catch |err| {
        std.debug.print("error: nn sample training failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer model.deinit();

    const out = nn.sample(allocator, &model, seed, n) catch |err| {
        std.debug.print("error: nn sample failed: {s}\n", .{@errorName(err)});
        return 1;
    };
    defer allocator.free(out);

    printReport(model.report);
    std.debug.print("nn sample: {s}\n", .{out});
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
