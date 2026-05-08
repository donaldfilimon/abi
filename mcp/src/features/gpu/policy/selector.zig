const std = @import("std");
const catalog = @import("catalog.zig");

pub const BackendNameList = struct {
    items: [12][]const u8 = undefined,
    len: usize = 0,

    pub fn append(self: *BackendNameList, backend_name: []const u8) void {
        if (self.len >= self.items.len) return;
        for (self.items[0..self.len]) |existing| {
            if (std.mem.eql(u8, existing, backend_name)) return;
        }
        self.items[self.len] = backend_name;
        self.len += 1;
    }

    pub fn slice(self: *const BackendNameList) []const []const u8 {
        return self.items[0..self.len];
    }
};

pub const SelectionContext = struct {
    platform: catalog.PlatformClass,
    enable_gpu: bool,
    enable_web: bool,
    can_link_metal: bool = true,
    warn_if_metal_skipped: bool = false,
    allow_simulated: bool = false,
    android_primary: ?[]const u8 = null,
};

pub fn resolveAutoBackendNames(ctx: SelectionContext) BackendNameList {
    var result = BackendNameList{};

    const base_order = blk: {
        if (ctx.platform == .android) {
            if (ctx.android_primary) |primary| {
                break :blk catalog.withAndroidPrimary(primary);
            }
        }
        break :blk catalog.defaultOrder(ctx.platform);
    };

    for (base_order) |name| {
        if (std.mem.eql(u8, name, "metal") and !ctx.can_link_metal) {
            if (ctx.warn_if_metal_skipped) {
                std.log.warn(
                    "Skipping Metal backend for macOS target: required Apple frameworks were not found.",
                    .{},
                );
            }
            continue;
        }

        if (!shouldInclude(ctx, name)) continue;
        result.append(name);
    }

    return result;
}

fn shouldInclude(ctx: SelectionContext, backend_name: []const u8) bool {
    if (std.mem.eql(u8, backend_name, "webgpu") or std.mem.eql(u8, backend_name, "webgl2")) {
        return ctx.enable_web and ctx.platform == .web;
    }

    if (std.mem.eql(u8, backend_name, "simulated")) {
        return ctx.allow_simulated and (ctx.enable_gpu or ctx.enable_web);
    }

    return ctx.enable_gpu;
}

test {
    std.testing.refAllDecls(@This());
}
