//! Observability helpers: structured logging, trace identifiers, and metrics.

const std = @import("std");

pub const telemetry = @import("telemetry.zig");

pub const StructuredLogger = telemetry.StructuredLogger;
pub const StructuredEvent = telemetry.StructuredEvent;
pub const TraceId = telemetry.TraceId;
pub const TelemetrySink = telemetry.TelemetrySink;
pub const TelemetrySnapshot = telemetry.TelemetrySnapshot;
pub const LogLevel = telemetry.LogLevel;
pub const CallStatus = telemetry.CallStatus;

test {
    std.testing.refAllDecls(@This());
}
