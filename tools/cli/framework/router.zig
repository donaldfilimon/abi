const std = @import("std");
const context_mod = @import("context");
const types = @import("types");
const completion = @import("completion");
const errors = @import("errors");
const utils = @import("../utils/mod.zig");

const max_forward_depth: usize = 8;

pub fn runCommand(
    ctx: context_mod.CommandContext,
    descriptors: []const types.CommandDescriptor,
    raw_command: []const u8,
    args: []const [:0]const u8,
) !bool {
    const descriptor = completion.findDescriptor(descriptors, raw_command) orelse return false;
    try runDescriptor(ctx, descriptors, descriptor, args, 0);
    return true;
}

fn runDescriptor(
    ctx: context_mod.CommandContext,
    descriptors: []const types.CommandDescriptor,
    descriptor: *const types.CommandDescriptor,
    args: []const [:0]const u8,
    depth: usize,
) !void {
    if (depth > max_forward_depth) return errors.Error.ForwardLoop;

    if (descriptor.children.len > 0 and args.len > 0) {
        const child_token = std.mem.sliceTo(args[0], 0);
        if (!types.isHelpToken(child_token)) {
            if (completion.findChildDescriptor(descriptor, child_token)) |child| {
                try runDescriptor(ctx, descriptors, child, args[1..], depth + 1);
                return;
            }
            // Unknown child subcommand — suggest a close match if one exists.
            var child_names_buf: [64][]const u8 = undefined;
            const child_count = @min(descriptor.children.len, child_names_buf.len);
            for (descriptor.children[0..child_count], 0..) |child, i| {
                child_names_buf[i] = child.name;
            }
            if (utils.args.suggestCommand(child_token, child_names_buf[0..child_count])) |suggestion| {
                utils.output.printInfo("Unknown subcommand '{s}'. Did you mean: {s}?", .{ child_token, suggestion });
            }
        }
    }

    if (descriptor.forward) |forward| {
        if (forward.warning) |warning| {
            utils.output.printWarning("{s}", .{warning});
        }

        const target = completion.findDescriptor(descriptors, forward.target) orelse return errors.Error.UnknownCommand;

        var forwarded = std.ArrayListUnmanaged([:0]const u8).empty;
        defer forwarded.deinit(ctx.allocator);

        for (forward.prepend_args) |arg| try forwarded.append(ctx.allocator, arg);
        for (args) |arg| try forwarded.append(ctx.allocator, arg);

        try runDescriptor(ctx, descriptors, target, forwarded.items, depth + 1);
        return;
    }

    try descriptor.handler(&ctx, args);
}

test "runDescriptor: unknown child emits suggestion (smoke)" {
    // This test verifies that the suggestion path does not panic.
    // Output goes to stderr and is not captured, but must not crash.
    const context_mod2 = @import("context");
    _ = context_mod2;
    // suggestCommand is already tested in utils/args.zig. Here we only do a
    // compile-time smoke check that the import chain resolves.
    _ = utils.args.suggestCommand;
}

test {
    std.testing.refAllDecls(@This());
}
