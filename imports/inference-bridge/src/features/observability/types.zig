//! Shared types for the observability feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

/// Primary error set for observability operations.
pub const Error = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

/// Error set for monitoring lifecycle operations.
pub const MonitoringError = error{
    MonitoringDisabled,
};

/// Legacy metrics configuration placeholder.
pub const MetricsConfig = struct {};

/// Legacy metrics summary placeholder.
pub const MetricsSummary = struct {};
