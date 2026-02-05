const std = @import("std");
const types = @import("types.zig");

// Alerting (stubs)
pub const AlertManager = struct {};
pub const AlertManagerConfig = struct {};
pub const AlertRule = struct {};
pub const AlertRuleBuilder = struct {};
pub const Alert = struct {};
pub const AlertCondition = struct {};
pub const AlertStats = struct {};
pub const AlertHandler = struct {};
pub const MetricValues = struct {};

pub fn createAlertRule() types.Error!AlertRule {
    return error.ObservabilityDisabled;
}
