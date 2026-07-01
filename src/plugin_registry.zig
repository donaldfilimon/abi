//! Generated plugin registry. DO NOT EDIT.
const Registry = @import("core/registry.zig").Registry;

pub fn registerPlugins(registry: *Registry) !void {
    try registry.registerPlugin(.{ .name = "accelerator-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-accelerator gate.", .target_feature = "accelerator", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "ai-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-ai gate.", .target_feature = "ai", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "example-plugin", .version = "0.1.0", .description = "Minimal example plugin used by registry generation tests.", .target_feature = "plugins", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "example-wdbx-plugin", .version = "0.1.0", .description = "Example WDBX plugin used by multi-plugin registry contract tests.", .target_feature = "wdbx", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "foundationmodels-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-foundationmodels gate.", .target_feature = "foundationmodels", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "gpu-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-gpu gate.", .target_feature = "gpu", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "hash-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-hash gate.", .target_feature = "hash", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "metrics-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-metrics gate.", .target_feature = "metrics", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "mlir-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-mlir gate.", .target_feature = "mlir", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "mobile-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-mobile gate.", .target_feature = "mobile", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "nn-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-nn gate.", .target_feature = "nn", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "os-control-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-os-control gate.", .target_feature = "os-control", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "sea-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-sea gate.", .target_feature = "sea", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "shader-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-shader gate.", .target_feature = "shader", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "telemetry-exporter", .version = "0.1.0", .description = "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.", .target_feature = "telemetry", .entry_point = "mod.zig" });
    try registry.registerPlugin(.{ .name = "tui-plugin", .version = "0.1.0", .description = "Example reference plugin targeting the feat-tui gate.", .target_feature = "tui", .entry_point = "mod.zig" });
}
