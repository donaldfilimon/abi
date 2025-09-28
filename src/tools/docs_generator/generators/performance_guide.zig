const std = @import("std");

pub fn generatePerformanceGuide(allocator: std.mem.Allocator) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    var writer = buffer.writer();

    try writer.writeAll("# Performance Guide\n\n");
    try writer.writeAll("These notes summarise the tuning levers we exercise in benchmarks and production deployments.\n\n");

    try writer.writeAll("## ⚡ Optimization Strategies\n\n");
    try writer.writeAll("- Prefer streaming reads when importing large WDBX shards to keep memory usage predictable.\n");
    try writer.writeAll("- Use the `simd` helpers for distance calculations to avoid unnecessary scalar fallbacks.\n");
    try writer.writeAll("- Batch HTTP requests via the async scheduler to reuse TLS sessions.\n\n");

    try writer.writeAll("## Monitoring\n\n");
    try writer.writeAll("Integrate the `perf_guard` tool to export latency histograms. Example invocation:\n\n");
    try writer.writeAll("```bash\n$ zig run src/tools/perf_guard.zig -- --listen 0.0.0.0:9001\n```\n");

    var file = try std.fs.cwd().createFile("docs/generated/PERFORMANCE_GUIDE.md", .{ .truncate = true });
    defer file.close();
    try file.writeAll(buffer.items);
}
