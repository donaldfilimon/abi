//! Network Feature Overview
//!
//! Implements networking primitives such as service discovery, protocol
//! handling, scheduling, and HA (high‑availability) mechanisms. The module is
//! split into several files (`mod.zig`, `protocol.zig`, `registry.zig`,
//! `scheduler.zig`, `service_discovery.zig`, `ha.zig`). Build options control
//! which components are compiled.
//!
//! Extending the network stack typically means adding a new protocol module
//! and wiring it into `mod.zig`.
