# Observability

All agent operations should emit structured logs, metrics, and trace
identifiers. Use `std.log` for JSON-formatted logs and avoid writing secrets.
Metrics belong in `src/features/monitoring/metrics.zig` and should capture
provider latency, retry counts, and token usage.

When integrating with external systems prefer OpenTelemetry-compatible formats.
Attach correlation IDs to WDBX persistence records for audit trails. Retain only
the minimum personally identifiable information required for debugging.
