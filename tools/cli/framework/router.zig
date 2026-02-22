const std = @import("std");
const context_mod = @import("context.zig");
const types = @import("types.zig");
const completion = @import("completion.zig");
const errors = @import("errors.zig");

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
        }
    }

    if (descriptor.forward) |forward| {
        if (forward.warning) |warning| {
            try ctx.stdout.print("Warning: {s}\n", .{warning});
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
