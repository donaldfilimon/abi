//! Tests for profiling module
//!
//! Tests only compiled when enable_profiling is true.

const std = @import("std");
const profiling = @import("profiling/mod.zig");

test "MetricsCollector initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = profiling.DEFAULT_METRICS_CONFIG;
    const collector = try profiling.MetricsCollector.init(gpa.allocator(), config, 4);
    defer collector.deinit();

    try std.testing.expectEqual(@as(usize, 4), collector.worker_stats.len);
}

test "WorkerMetrics initial state" {
    const stats = profiling.WorkerMetrics{};

    try std.testing.expectEqual(@as(u64, 0), stats.tasks_executed);
    try std.testing.expectEqual(@as(u64, 0), stats.total_execution_ns);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), stats.min_execution_ns);
    try std.testing.expectEqual(@as(u64, 0), stats.max_execution_ns);
}

test "Histogram records values correctly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const buckets = [_]u64{ 10, 50, 100, 500, 1000 };
    var histogram = try profiling.Histogram.init(gpa.allocator(), &buckets);
    defer histogram.deinit(gpa.allocator());

    histogram.record(5);
    histogram.record(20);
    histogram.record(100);
    histogram.record(2000);

    try std.testing.expectEqual(@as(u64, 4), histogram.total);
}

test "MetricsCollector records task execution" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = profiling.DEFAULT_METRICS_CONFIG;
    const collector = try profiling.MetricsCollector.init(gpa.allocator(), config, 2);
    defer collector.deinit();

    collector.recordTaskExecution(0, 1000);
    collector.recordTaskExecution(0, 2000);
    collector.recordTaskExecution(1, 1500);

    const stats = collector.getWorkerStats(0).?;
    try std.testing.expectEqual(@as(u64, 2), stats.tasks_executed);
    try std.testing.expectEqual(@as(u64, 3000), stats.total_execution_ns);
    try std.testing.expectEqual(@as(u64, 1000), stats.min_execution_ns);
    try std.testing.expectEqual(@as(u64, 2000), stats.max_execution_ns);

    const stats1 = collector.getWorkerStats(1).?;
    try std.testing.expectEqual(@as(u64, 1), stats1.tasks_executed);
    try std.testing.expectEqual(@as(u64, 1500), stats1.total_execution_ns);
}

test "MetricsSummary calculation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const config = profiling.DEFAULT_METRICS_CONFIG;
    const collector = try profiling.MetricsCollector.init(gpa.allocator(), config, 2);
    defer collector.deinit();

    collector.recordTaskExecution(0, 1000);
    collector.recordTaskExecution(1, 2000);
    collector.recordTaskExecution(0, 3000);

    const summary = collector.getSummary();
    try std.testing.expectEqual(@as(u64, 3), summary.total_tasks);
    try std.testing.expectEqual(@as(u64, 6000), summary.total_execution_ns);
    try std.testing.expectEqual(@as(u64, 2000), summary.avg_execution_ns);
    try std.testing.expectEqual(@as(u64, 1000), summary.min_execution_ns);
    try std.testing.expectEqual(@as(u64, 3000), summary.max_execution_ns);
}
