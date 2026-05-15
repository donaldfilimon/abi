const build_options = @import("build_options");

pub const ai = if (build_options.feat_ai) @import("ai/mod.zig") else @import("ai/stub.zig");
pub const accelerator = if (build_options.feat_accelerator) @import("accelerator/mod.zig") else @import("accelerator/stub.zig");
pub const gpu = if (build_options.feat_gpu) @import("gpu/mod.zig") else @import("gpu/stub.zig");
pub const mlir = if (build_options.feat_mlir) @import("mlir/mod.zig") else @import("mlir/stub.zig");
pub const os_control = @import("os_control/mod.zig");
pub const shaders = if (build_options.feat_shader) @import("shaders/mod.zig") else @import("shaders/stub.zig");
pub const tui = if (build_options.feat_tui) @import("tui/mod.zig") else @import("tui/stub.zig");
pub const wdbx = @import("wdbx/mod.zig");

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
