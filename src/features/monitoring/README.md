//! Monitoring Feature Overview
//!
//! Provides hooks for exporting metrics, tracing, and health checks. The module
//! currently contains a minimal implementation (`mod.zig`) that can be expanded
//! with Prometheus exporters or OpenTelemetry integration as the roadmap
//! progresses.
//!
//! To add new observability backends, create a sub‑module that implements the
//! expected metric/trace interfaces and re‑export it from `mod.zig`.
