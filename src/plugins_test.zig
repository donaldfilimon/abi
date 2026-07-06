//! Plugin test aggregator — gives the 16 bundled plugin `mod.zig` files a real test artifact.
//
// The inline `test { std.testing.refAllDecls(@This()); }` blocks added to every
// plugin mod/stub pair would never run under the normal build graph because
// `plugin_manager.zig` loads plugins at runtime by path, not via comptime
// `@import`. This aggregator imports every `mod.zig` so the `refAllDecls`
// coverage is actually exercised during `./build.sh check`.

const std = @import("std");

test "plugin mod.zig refAllDecls coverage" {
    _ = @import("plugins/accelerator-plugin/mod.zig");
    _ = @import("plugins/ai-plugin/mod.zig");
    _ = @import("plugins/example-plugin/mod.zig");
    _ = @import("plugins/example-wdbx-plugin/mod.zig");
    _ = @import("plugins/foundationmodels-plugin/mod.zig");
    _ = @import("plugins/gpu-plugin/mod.zig");
    _ = @import("plugins/hash-plugin/mod.zig");
    _ = @import("plugins/metrics-plugin/mod.zig");
    _ = @import("plugins/mlir-plugin/mod.zig");
    _ = @import("plugins/mobile-plugin/mod.zig");
    _ = @import("plugins/nn-plugin/mod.zig");
    _ = @import("plugins/os-control-plugin/mod.zig");
    _ = @import("plugins/sea-plugin/mod.zig");
    _ = @import("plugins/shader-plugin/mod.zig");
    _ = @import("plugins/telemetry-exporter/mod.zig");
    _ = @import("plugins/tui-plugin/mod.zig");
}

test {
    std.testing.refAllDecls(@This());
}
