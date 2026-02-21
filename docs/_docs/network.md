---
title: "Network"
description: "Distributed compute, node discovery, and load balancing"
section: "Infrastructure"
order: 1
---

# Network

The Network module provides a full distributed compute layer with node
registration, service discovery, Raft consensus, task scheduling, load
balancing, connection pooling, circuit breakers, and unified memory management
across cluster nodes.

- **Build flag:** `-Denable-network=true` (default: enabled)
- **Namespace:** `abi.network`
- **Source:** `src/features/network/`

## Overview

The network module is the backbone of ABI's distributed computing story. It
manages a cluster of nodes that can discover each other, elect leaders via Raft
consensus, schedule tasks across the cluster, and balance load using pluggable
strategies. The module is organized into several subsystems:

- **Node Registry** -- Register, discover, and monitor compute nodes
- **Service Discovery** -- DNS-based and multicast service discovery with health checks
- **Raft Consensus** -- Leader election, log replication, snapshots, and membership changes
- **Task Scheduler** -- Distribute tasks across nodes with priority and load-aware scheduling
- **Load Balancer** -- Round-robin, least-connections, and weighted strategies
- **Connection Pool** -- Pooled, reusable connections with idle timeout and health probes
- **Circuit Breaker** -- Per-service failure tracking with open/closed/half-open state machine
- **Rate Limiter** -- Token bucket, sliding window, and fixed window algorithms
- **Retry Logic** -- Configurable retry with exponential backoff
- **Transport** -- TCP-based RPC with message framing and serialization
- **Unified Memory** -- Shared memory regions across nodes with coherence protocols
- **Linking** -- Secure inter-node channels with Thunderbolt and Internet transports
- **Failover Manager** -- Automated failover on node failure

## Quick Start

```zig
const abi = @import("abi");

// Initialize the network module via the Framework
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withNetworkDefaults()
    .build();
defer framework.deinit();

// Get the default node registry
const registry = try abi.network.defaultRegistry();

// Register nodes
try registry.register("node-1", "localhost:8080");
try registry.register("node-2", "localhost:8081");

// Update node status
_ = registry.touch("node-1");
_ = registry.setStatus("node-2", .degraded);

// List registered nodes
const nodes = registry.list();
for (nodes) |node| {
    std.debug.print("Node '{s}' at {s} - Status: {t}\n", .{
        node.id, node.address, node.status,
    });
}
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context with connect/disconnect lifecycle |
| `NetworkState` | Module-level state holding allocator, config, and registry |
| `NetworkConfig` | Cluster ID, heartbeat timeout, max nodes |
| `NodeRegistry` | Registry of compute nodes with status tracking |
| `NodeInfo` | Metadata about a registered node (id, address, status) |
| `NodeStatus` | Node health state (healthy, degraded, offline) |
| `Node` | Alias for `NodeInfo` |

### Service Discovery

| Type | Description |
|------|-------------|
| `ServiceDiscovery` | Discovery engine with pluggable backends |
| `DiscoveryConfig` | Discovery backend selection and timeouts |
| `DiscoveryBackend` | Backend type (DNS, multicast, static) |
| `ServiceInstance` | A discovered service instance |
| `ServiceStatus` | Service health state |
| `DiscoveryError` | Errors from discovery operations |

### Raft Consensus

| Type | Description |
|------|-------------|
| `RaftNode` | A single Raft participant with state machine |
| `RaftState` | Follower, candidate, or leader |
| `RaftConfig` | Election timeout, heartbeat interval, log settings |
| `LogEntry` | A single entry in the replicated log |
| `RaftPersistence` | Durable storage for Raft state |
| `RaftSnapshotManager` | Snapshot creation and installation |
| `SnapshotMetadata` | Metadata for a Raft snapshot |
| `ConfigChangeRequest` | Cluster membership change request |

### Task Scheduling

| Type | Description |
|------|-------------|
| `TaskScheduler` | Distributes tasks across compute nodes |
| `SchedulerConfig` | Max tasks, scheduling interval, strategy |
| `TaskPriority` | Task priority level |
| `TaskState` | Pending, running, completed, failed |
| `ComputeNode` | A node available for task execution |
| `LoadBalancingStrategy` | Round-robin, least-connections, weighted |
| `SchedulerStats` | Task counts and scheduling metrics |

### Load Balancing

| Type | Description |
|------|-------------|
| `LoadBalancer` | Selects nodes using a configurable strategy |
| `LoadBalancerConfig` | Strategy, health check interval |
| `LoadBalancerStrategy` | Algorithm selection |
| `NodeState` | Per-node load balancer state |
| `NodeStats` | Per-node request and latency stats |

### Connection Pool

| Type | Description |
|------|-------------|
| `ConnectionPool` | Pooled connections with health probes |
| `ConnectionPoolConfig` | Max connections, idle timeout, health interval |
| `PooledConnection` | A connection checked out from the pool |
| `ConnectionState` | Idle, active, or closed |
| `ConnectionStats` | Per-connection metrics |
| `PoolBuilder` | Fluent builder for pool configuration |

### Retry and Rate Limiting

| Type | Description |
|------|-------------|
| `RetryExecutor` | Executes operations with retry logic |
| `RetryConfig` | Max retries, backoff strategy, jitter |
| `RetryStrategy` | Fixed, exponential, or linear backoff |
| `RateLimiter` | Per-key rate limiting |
| `RateLimiterConfig` | Requests per second, burst size, algorithm |
| `RateLimitAlgorithm` | Token bucket, sliding window, fixed window |
| `TokenBucketLimiter` | Token bucket implementation |
| `SlidingWindowLimiter` | Sliding window counter implementation |
| `FixedWindowLimiter` | Fixed window counter implementation |

### Circuit Breaker

| Type | Description |
|------|-------------|
| `CircuitBreaker` | Per-service circuit breaker |
| `CircuitConfig` | Failure threshold, reset timeout |
| `CircuitState` | Closed, open, or half-open |
| `CircuitRegistry` | Collection of named circuit breakers |
| `CircuitStats` | Success/failure counts and state |

### Transport

| Type | Description |
|------|-------------|
| `TcpTransport` | TCP-based message transport |
| `TransportConfig` | Bind address, buffer sizes, timeouts |
| `MessageType` | RPC message type discriminator |
| `MessageHeader` | Wire-format message header |
| `RaftTransport` | Raft-specific transport layer |
| `RaftTransportConfig` | Peer addresses and Raft transport settings |

### Failover and Unified Memory

| Type | Description |
|------|-------------|
| `FailoverManager` | Automated failover on node failure |
| `FailoverConfig` | Detection interval, failover strategy |
| `FailoverState` | Current failover state |
| `UnifiedMemoryManager` | Cross-node shared memory |
| `UnifiedMemoryConfig` | Region size, coherence protocol |
| `MemoryRegion` | A shared memory region |
| `CoherenceProtocol` | MESI, MOESI, or directory-based |

### Linking

| Type | Description |
|------|-------------|
| `LinkManager` | Manages inter-node links |
| `Link` | A single link between two nodes |
| `SecureChannel` | Encrypted channel over a link |
| `ThunderboltTransport` | High-speed local interconnect |
| `InternetTransport` | WAN transport with NAT traversal |
| `QuicConnection` | QUIC-based connection |

### Key Functions

| Function | Description |
|----------|-------------|
| `isEnabled() bool` | Returns `true` if network is compiled in |
| `isInitialized() bool` | Returns `true` if the module has been initialized |
| `defaultRegistry() !*NodeRegistry` | Get the default node registry |
| `defaultConfig() ?NetworkConfig` | Get current network configuration |
| `retryOperation(fn, config) !RetryResult` | Execute with retry logic |
| `retryWithStrategy(fn, strategy) !RetryResult` | Execute with a specific retry strategy |
| `createRaftCluster(config) ![]RaftNode` | Bootstrap a Raft cluster |
| `generateServiceId(name) []const u8` | Generate a unique service identifier |

## Configuration

Network is configured through the `NetworkConfig` struct:

```zig
const config: abi.config.NetworkConfig = .{
    .bind_address = "0.0.0.0",
    .port = 8080,
    .discovery_enabled = true,
    .consensus_enabled = false,
    .role = .worker,          // .coordinator, .worker, or .observer
    .bootstrap_peers = &.{},
    .unified_memory = null,   // optional unified memory config
    .linking = null,          // optional linking config
};
```

The module-level `NetworkConfig` provides simpler defaults:

```zig
const net_config = abi.network.NetworkConfig{
    .cluster_id = "default",
    .heartbeat_timeout_ms = 30_000,
    .max_nodes = 256,
};
```

## CLI Commands

The `network` CLI command provides cluster management:

```bash
# Show network config and node count
zig build run -- network status

# List registered nodes
zig build run -- network list

# Register a node
zig build run -- network register node-1 localhost:8080

# Remove a node
zig build run -- network unregister node-1

# Update heartbeat timestamp
zig build run -- network touch node-1

# Set node status (healthy, degraded, offline)
zig build run -- network set-status node-1 degraded
```

## Examples

See `examples/network.zig` for a complete working example that initializes the
Framework, registers nodes, updates their status, and lists the cluster:

```bash
zig build run-network
```

## Disabling at Build Time

```bash
# Compile without network support
zig build -Denable-network=false
```

When disabled, all public functions return `error.NetworkDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures so downstream code compiles without conditional guards.

## Related

- [Gateway](gateway.html) -- API gateway with routing and rate limiting
- [Messaging](messaging.html) -- Pub/sub event bus
- [Cloud](cloud.html) -- Serverless cloud function adapters

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
