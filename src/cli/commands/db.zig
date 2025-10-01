const std = @import("std");
const modern_cli = @import("../../tools/cli/modern_cli.zig");
const errors = @import("../errors.zig");
const state_mod = @import("../state.zig");
const db_helpers = @import("../../features/database/db_helpers.zig");

fn requireState(ctx: *modern_cli.Context) errors.CommandError!*state_mod.State {
    return ctx.userData(state_mod.State) orelse errors.CommandError.RuntimeFailure;
}

fn parseVectorInput(allocator: std.mem.Allocator, raw: []const u8) errors.CommandError![]f32 {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return errors.CommandError.InvalidArgument;

    const slice = if (trimmed.len >= 2 and trimmed[0] == '[' and trimmed[trimmed.len - 1] == ']')
        std.mem.trim(u8, trimmed[1 .. trimmed.len - 1], " \t\r\n")
    else
        trimmed;

    if (slice.len == 0) return errors.CommandError.InvalidArgument;

    return db_helpers.helpers.parseVector(allocator, slice) catch {
        return errors.CommandError.InvalidArgument;
    };
}

fn insertHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const vector_raw = args.getString("vec", "");
    if (vector_raw.len == 0) return errors.CommandError.MissingArgument;

    var vector = try parseVectorInput(state.allocator, vector_raw);
    defer state.allocator.free(vector);

    const metadata_raw = args.getString("meta", "");
    const metadata_opt = if (metadata_raw.len > 0) metadata_raw else null;

    const id = state.vector_store.insert(vector, metadata_opt) catch |err| {
        return switch (err) {
            state_mod.VectorStoreError.DimensionMismatch => errors.CommandError.InvalidArgument,
            state_mod.VectorStoreError.InvalidVector => errors.CommandError.InvalidArgument,
            error.OutOfMemory => errors.CommandError.RuntimeFailure,
            else => errors.CommandError.RuntimeFailure,
        };
    };

    const dimension = state.vector_store.dimension.?;
    const count = state.vector_store.records.items.len;
    const stdout = std.io.getStdOut().writer();

    if (args.hasFlag("json")) {
        var buffer = std.ArrayList(u8).init(state.allocator);
        defer buffer.deinit();
        try std.json.stringify(
            .{
                .id = id,
                .dimension = dimension,
                .count = count,
            },
            .{},
            buffer.writer(),
        );
        try stdout.writeAll(buffer.items);
        try stdout.writeByte('\n');
    } else {
        try stdout.print(
            "Inserted vector id={d} (dimension {d}). Total stored: {d}.\n",
            .{ id, dimension, count },
        );
    }
}

fn searchHandler(ctx: *modern_cli.Context, args: *modern_cli.ParsedArgs) errors.CommandError!void {
    const state = try requireState(ctx);
    try state.consumeBudget();

    const vector_raw = args.getString("vec", "");
    if (vector_raw.len == 0) return errors.CommandError.MissingArgument;

    var query = try parseVectorInput(state.allocator, vector_raw);
    defer state.allocator.free(query);

    const k_value = args.getInteger("k", 5);
    if (k_value <= 0) return errors.CommandError.InvalidArgument;
    const k = @as(usize, @intCast(k_value));

    var results = state.vector_store.search(state.allocator, query, k) catch |err| {
        return switch (err) {
            state_mod.VectorStoreError.DimensionMismatch => errors.CommandError.InvalidArgument,
            state_mod.VectorStoreError.InvalidVector => errors.CommandError.InvalidArgument,
            error.OutOfMemory => errors.CommandError.RuntimeFailure,
            else => errors.CommandError.RuntimeFailure,
        };
    };
    defer state.allocator.free(results);

    const stdout = std.io.getStdOut().writer();

    if (args.hasFlag("json")) {
        var json_results = std.ArrayList(struct {
            id: u64,
            distance: f32,
            metadata: ?[]const u8,
        }).init(state.allocator);
        defer json_results.deinit();

        for (results) |res| {
            try json_results.append(.{
                .id = res.id,
                .distance = res.distance,
                .metadata = res.metadata,
            });
        }

        var buffer = std.ArrayList(u8).init(state.allocator);
        defer buffer.deinit();
        try std.json.stringify(
            .{
                .count = results.len,
                .requested = k,
                .results = json_results.items,
            },
            .{},
            buffer.writer(),
        );
        try stdout.writeAll(buffer.items);
        try stdout.writeByte('\n');
    } else {
        if (results.len == 0) {
            try stdout.writeAll("No vectors stored yet.\n");
            return;
        }
        try stdout.print("Top {d} matches:\n", .{results.len});
        for (results, 0..) |res, idx| {
            try stdout.print(
                "  {d}) id={d} distance={d:.4}\n",
                .{ idx + 1, res.id, res.distance },
            );
            if (res.metadata) |meta| {
                try stdout.print("      metadata: {s}\n", .{meta});
            }
        }
    }
}

pub const insert_command = modern_cli.Command{
    .name = "insert",
    .description = "Insert a vector into the in-memory WDBX store",
    .handler = insertHandler,
    .options = &.{
        .{
            .name = "vec",
            .long = "vec",
            .description = "Vector values (e.g. [0.1,0.2,0.3])",
            .arg_type = .string,
            .required = true,
        },
        .{
            .name = "meta",
            .long = "meta",
            .description = "Optional metadata payload",
            .arg_type = .string,
        },
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const search_command = modern_cli.Command{
    .name = "search",
    .description = "Search nearest neighbours",
    .handler = searchHandler,
    .options = &.{
        .{
            .name = "vec",
            .long = "vec",
            .description = "Query vector",
            .arg_type = .string,
            .required = true,
        },
        .{
            .name = "k",
            .long = "k",
            .description = "Number of neighbours to return",
            .arg_type = .integer,
        },
        .{
            .name = "json",
            .long = "json",
            .description = "Emit JSON payload",
            .arg_type = .boolean,
        },
    },
};

pub const command = modern_cli.Command{
    .name = "db",
    .description = "Vector database operations",
    .subcommands = &.{ &insert_command, &search_command },
};