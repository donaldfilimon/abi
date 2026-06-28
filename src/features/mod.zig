const build_options = @import("build_options");

// ── AI & Data ──
pub const ai = if (build_options.feat_ai) @import("ai/mod.zig") else @import("ai/stub.zig");
pub const wdbx = if (build_options.feat_wdbx) @import("wdbx/mod.zig") else @import("wdbx/stub.zig");
pub const sea = if (build_options.feat_sea) @import("sea/mod.zig") else @import("sea/stub.zig");

// ── GPU & Compute ──
pub const gpu = if (build_options.feat_gpu) @import("gpu/mod.zig") else @import("gpu/stub.zig");
pub const accelerator = if (build_options.feat_accelerator) @import("accelerator/mod.zig") else @import("accelerator/stub.zig");
pub const shaders = if (build_options.feat_shader) @import("shaders/mod.zig") else @import("shaders/stub.zig");
pub const mlir = if (build_options.feat_mlir) @import("mlir/mod.zig") else @import("mlir/stub.zig");

// ── OS & Platform ──
pub const os_control = if (build_options.feat_os_control) @import("os_control/mod.zig") else @import("os_control/stub.zig");
pub const mobile = if (build_options.feat_mobile) @import("mobile/mod.zig") else @import("mobile/stub.zig");
pub const metrics = if (build_options.feat_metrics) @import("metrics/mod.zig") else @import("metrics/stub.zig");

// ── UI ──
pub const tui = if (build_options.feat_tui) @import("tui/mod.zig") else @import("tui/stub.zig");

// ── Utilities ──
pub const hash = if (build_options.feat_hash) @import("hash/mod.zig") else @import("hash/stub.zig");
pub const telemetry = if (build_options.feat_telemetry) @import("telemetry/mod.zig") else @import("telemetry/stub.zig");
pub const nn = if (build_options.feat_nn) @import("nn/mod.zig") else @import("nn/stub.zig");

test {
    const std = @import("std");
    // AI & Data
    std.testing.refAllDecls(ai);
    std.testing.refAllDecls(wdbx);
    std.testing.refAllDecls(sea);
    // GPU & Compute
    std.testing.refAllDecls(gpu);
    std.testing.refAllDecls(accelerator);
    std.testing.refAllDecls(shaders);
    std.testing.refAllDecls(mlir);
    // OS & Platform
    std.testing.refAllDecls(os_control);
    std.testing.refAllDecls(mobile);
    std.testing.refAllDecls(metrics);
    // UI
    std.testing.refAllDecls(tui);
    // Utilities
    std.testing.refAllDecls(hash);
    std.testing.refAllDecls(telemetry);
    std.testing.refAllDecls(nn);
    std.testing.refAllDecls(@This());
}
