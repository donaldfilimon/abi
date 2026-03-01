//! Observability Alerting Tests — Alert manager, rules, conditions,
//! severity, metric values, and Prometheus rule builder.

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

const observability = abi.features.observability;
const AlertManager = observability.AlertManager;
const AlertManagerConfig = observability.AlertManagerConfig;
const AlertCondition = observability.AlertCondition;
const AlertSeverity = observability.AlertSeverity;
const MetricValues = observability.MetricValues;

// ============================================================================
// Alert Manager Tests
// ============================================================================

test "observability: alert manager initialization" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    const stats = manager.getStats();
    try testing.expectEqual(@as(usize, 0), stats.total_rules);
    try testing.expectEqual(@as(usize, 0), stats.firing_alerts);
}

test "observability: alert manager add rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{
        .name = "high_cpu",
        .metric = "cpu_usage",
        .threshold = 80.0,
        .severity = .warning,
    });

    const stats = manager.getStats();
    try testing.expectEqual(@as(usize, 1), stats.total_rules);
    try testing.expectEqual(@as(usize, 1), stats.active_rules);
}

test "observability: alert manager duplicate rule error" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });

    // Adding duplicate should fail
    const result = manager.addRule(.{ .name = "test_rule", .metric = "metric2", .threshold = 100.0 });
    try testing.expectError(observability.AlertError.DuplicateRule, result);
}

test "observability: alert manager remove rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });
    try testing.expectEqual(@as(usize, 1), manager.getStats().total_rules);

    try manager.removeRule("test_rule");
    try testing.expectEqual(@as(usize, 0), manager.getStats().total_rules);
}

test "observability: alert manager remove nonexistent rule error" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    const result = manager.removeRule("nonexistent");
    try testing.expectError(observability.AlertError.RuleNotFound, result);
}

test "observability: alert manager enable disable rule" {
    const allocator = testing.allocator;
    var manager = try AlertManager.init(allocator, AlertManagerConfig{});
    defer manager.deinit();

    try manager.addRule(.{ .name = "test_rule", .metric = "metric1", .threshold = 50.0 });
    try testing.expectEqual(@as(usize, 1), manager.getStats().active_rules);

    try manager.setRuleEnabled("test_rule", false);
    try testing.expectEqual(@as(usize, 0), manager.getStats().active_rules);

    try manager.setRuleEnabled("test_rule", true);
    try testing.expectEqual(@as(usize, 1), manager.getStats().active_rules);
}

test "observability: alert condition evaluation greater than" {
    try testing.expect(AlertCondition.greater_than.evaluate(100.0, 50.0));
    try testing.expect(!AlertCondition.greater_than.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.greater_than.evaluate(50.0, 50.0));
}

test "observability: alert condition evaluation less than" {
    try testing.expect(AlertCondition.less_than.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.less_than.evaluate(100.0, 50.0));
    try testing.expect(!AlertCondition.less_than.evaluate(50.0, 50.0));
}

test "observability: alert condition evaluation equal" {
    try testing.expect(AlertCondition.equal.evaluate(50.0, 50.0));
    try testing.expect(AlertCondition.equal.evaluate(50.0001, 50.0001));
    try testing.expect(!AlertCondition.equal.evaluate(50.0, 51.0));
}

test "observability: alert condition evaluation not equal" {
    try testing.expect(AlertCondition.not_equal.evaluate(50.0, 100.0));
    try testing.expect(!AlertCondition.not_equal.evaluate(50.0, 50.0));
}

test "observability: alert severity comparison" {
    try testing.expect(AlertSeverity.info.toInt() < AlertSeverity.warning.toInt());
    try testing.expect(AlertSeverity.warning.toInt() < AlertSeverity.critical.toInt());
}

test "observability: alert rule builder" {
    var builder = observability.createAlertRule("high_latency", "request_latency_ms");
    const rule = builder
        .threshold(500.0)
        .condition(.greater_than)
        .severity(.critical)
        .forDuration(60000)
        .description("High request latency detected")
        .build();

    try testing.expectEqualStrings("high_latency", rule.name);
    try testing.expectEqualStrings("request_latency_ms", rule.metric);
    try testing.expectApproxEqAbs(@as(f64, 500.0), rule.threshold, 0.001);
    try testing.expectEqual(AlertCondition.greater_than, rule.condition);
    try testing.expectEqual(AlertSeverity.critical, rule.severity);
    try testing.expectEqual(@as(u64, 60000), rule.for_duration_ms);
}

test "observability: metric values set and get" {
    const allocator = testing.allocator;
    var values = MetricValues.init();
    defer values.deinit(allocator);

    try values.set(allocator, "cpu_usage", 75.5);
    try values.set(allocator, "memory_usage", 60.0);

    try testing.expectApproxEqAbs(@as(f64, 75.5), values.get("cpu_usage").?, 0.001);
    try testing.expectApproxEqAbs(@as(f64, 60.0), values.get("memory_usage").?, 0.001);
    try testing.expect(values.get("nonexistent") == null);
}
