const build_options = @import("build_options");

pub const ai = if (build_options.feat_ai) @import("ai/mod.zig") else @import("ai/stub.zig");
pub const wdbx = @import("wdbx/mod.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
