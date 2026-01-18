//! # Network
//!
//! Distributed networking primitives for service discovery, HA, and scheduling.
//!
//! ## Features
//!
//! - **Service Discovery**: Automatic node discovery and registration
//! - **High Availability**: Leader election, failover, replication
//! - **Circuit Breaker**: Fault tolerance with exponential backoff
//! - **Rate Limiting**: Token bucket rate limiting
//! - **Retry Logic**: Configurable retry with jitter
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `mod.zig` | Public API aggregation |
//! | `protocol.zig` | Wire protocol definitions |
//! | `registry.zig` | Service registry |
//! | `scheduler.zig` | Distributed task scheduling |
//! | `service_discovery.zig` | Node discovery |
//! | `ha.zig` | High availability primitives |
//! | `circuit_breaker.zig` | Fault tolerance |
//! | `rate_limiter.zig` | Rate limiting |
//! | `retry.zig` | Retry logic |
//!
//! ## Usage
//!
//! ```zig
//! const network = @import("abi").network;
//!
//! // Service discovery
//! var discovery = try network.ServiceDiscovery.init(allocator, .{});
//! defer discovery.deinit();
//! try discovery.register("my-service", .{ .port = 8080 });
//!
//! // Circuit breaker
//! var breaker = network.CircuitBreaker.init(.{
//!     .failure_threshold = 5,
//!     .reset_timeout_ms = 30000,
//! });
//! ```
//!
//! ## Build Options
//!
//! Enable with `-Denable-network=true` (default: true).
//!
//! ## See Also
//!
//! - [Network Documentation](../../docs/network.md)
//! - [API Reference](../../API_REFERENCE.md)

