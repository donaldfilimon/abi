//! Public build metadata helpers.

const build_options = @import("build_options");

pub const package_version = build_options.package_version;
pub const features = @import("../features/core/feature_catalog.zig");

pub fn version() []const u8 {
    return package_version;
}

test {
    @import("std").testing.refAllDecls(@This());
}
