const std = @import("std");
const types = @import("types.zig");

pub fn resolveAlias(descriptors: []const types.CommandDescriptor, raw: []const u8) []const u8 {
    for (descriptors) |descriptor| {
        if (std.mem.eql(u8, raw, descriptor.name)) return descriptor.name;
        for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw, alias)) return descriptor.name;
        }
    }
    return raw;
}

pub fn findDescriptor(
    descriptors: []const types.CommandDescriptor,
    raw: []const u8,
) ?*const types.CommandDescriptor {
    return findDescriptorInSlice(descriptors, raw);
}

pub fn findChildDescriptor(
    descriptor: *const types.CommandDescriptor,
    raw: []const u8,
) ?*const types.CommandDescriptor {
    return findDescriptorInSlice(descriptor.children, raw);
}

fn findDescriptorInSlice(
    descriptors: []const types.CommandDescriptor,
    raw: []const u8,
) ?*const types.CommandDescriptor {
    for (descriptors) |*descriptor| {
        if (std.mem.eql(u8, raw, descriptor.name)) return descriptor;
        for (descriptor.aliases) |alias| {
            if (std.mem.eql(u8, raw, alias)) return descriptor;
        }
    }
    return null;
}
