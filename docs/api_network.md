# Network API Reference
> **Codebase Status:** Synced with repository as of 2026-01-22.

**Source:** `src/network/mod.zig`

The network module provides distributed compute capabilities with node discovery, Raft consensus, task scheduling, load balancing, and fault tolerance features.

---

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with network enabled
    const config = abi.Config.init().withNetwork(true);
    var fw = try abi.Framework.init(allocator, config);
    defer fw.deinit();

    // Access network features
    if (fw.network()) |net| {
        try net.connect();
        defer net.disconnect();

        const peers = try net.discoverPeers();
        for (peers) |peer| {
            std.debug.print("Found peer: {s}\n", .{peer.id});
        }
    }
}
```

---

## Framework Context

### `Context`

Network context for Framework integration.

```zig
pub const Context = struct {
    pub fn connect(self: *Context) !void;
    pub fn disconnect(self: *Context) void;
    pub fn getState(self: *Context) State;
    pub fn discoverPeers(self: *Context) ![]NodeInfo;
    pub fn sendTask(self: *Context, node_id: []const u8, task: anytype) !void;
};
```

---

## Node Registry

### `NodeRegistry`

Central registry for cluster nodes.

```zig
pub const NodeRegistry = struct {
    pub fn init(allocator: Allocator) !NodeRegistry;
    pub fn deinit(self: *NodeRegistry) void;
    pub fn register(self: *NodeRegistry, node: NodeInfo) !void;
    pub fn unregister(self: *NodeRegistry, node_id: []const u8) !void;
    pub fn get(self: *NodeRegistry, node_id: []const u8) ?NodeInfo;
    pub fn list(self: *NodeRegistry) []NodeInfo;
    pub fn getHealthy(self: *NodeRegistry) []NodeInfo;
    pub fn updateStatus(self: *NodeRegistry, node_id: []const u8, status: NodeStatus) !void;
};
```

### `NodeInfo`

Node information structure.

```zig
pub const NodeInfo = struct {
    id: []const u8,
    address: []const u8,
    port: u16,
    status: NodeStatus,
    capabilities: []const Capability,
    metadata: ?[]const u8,
    last_heartbeat: i64,
    joined_at: i64,
};
```

### `NodeStatus`

Node health status.

```zig
pub const NodeStatus = enum {
    unknown,
    starting,
    healthy,
    degraded,
    unhealthy,
    draining,
    offline,
};
```

---

## Task Scheduling

### `TaskScheduler`

Distributed task scheduler.

```zig
pub const TaskScheduler = struct {
    pub fn init(allocator: Allocator, config: SchedulerConfig) !TaskScheduler;
    pub fn deinit(self: *TaskScheduler) void;
    pub fn submit(self: *TaskScheduler, task: TaskEnvelope) !TaskId;
    pub fn cancel(self: *TaskScheduler, task_id: TaskId) !void;
    pub fn getStatus(self: *TaskScheduler, task_id: TaskId) ?TaskState;
    pub fn getResult(self: *TaskScheduler, task_id: TaskId) !?ResultEnvelope;
    pub fn waitFor(self: *TaskScheduler, task_id: TaskId, timeout_ms: u64) !ResultEnvelope;
    pub fn getStats(self: *TaskScheduler) SchedulerStats;
};
```

### `SchedulerConfig`

Scheduler configuration.

```zig
pub const SchedulerConfig = struct {
    max_concurrent_tasks: u32 = 100,
    task_timeout_ms: u64 = 60000,
    retry_count: u32 = 3,
    retry_delay_ms: u64 = 1000,
    load_balancing: LoadBalancingStrategy = .round_robin,
    priority_enabled: bool = true,
};
```

### `TaskPriority`

Task priority levels.

```zig
pub const TaskPriority = enum {
    low,
    normal,
    high,
    critical,
};
```

### `TaskState`

Task execution state.

```zig
pub const TaskState = enum {
    pending,
    queued,
    running,
    completed,
    failed,
    cancelled,
    timeout,
};
```

### `ComputeNode`

Compute node in the scheduler.

```zig
pub const ComputeNode = struct {
    info: NodeInfo,
    current_load: f32,
    max_capacity: u32,
    running_tasks: u32,
    completed_tasks: u64,
    failed_tasks: u64,
};
```

### `LoadBalancingStrategy`

Load balancing strategies.

```zig
pub const LoadBalancingStrategy = enum {
    round_robin,
    least_connections,
    weighted_round_robin,
    random,
    resource_based,
    latency_based,
};
```

### `SchedulerStats`

Scheduler statistics.

```zig
pub const SchedulerStats = struct {
    total_tasks: u64,
    pending_tasks: u32,
    running_tasks: u32,
    completed_tasks: u64,
    failed_tasks: u64,
    avg_task_duration_ms: f64,
    throughput_per_second: f32,
};
```

---

## Task Protocol

### `TaskEnvelope`

Task wrapper for network transmission.

```zig
pub const TaskEnvelope = struct {
    id: TaskId,
    task_type: []const u8,
    payload: []const u8,
    priority: TaskPriority,
    timeout_ms: u64,
    metadata: ?[]const u8,
    created_at: i64,
};
```

### `ResultEnvelope`

Result wrapper from task execution.

```zig
pub const ResultEnvelope = struct {
    task_id: TaskId,
    status: ResultStatus,
    payload: ?[]const u8,
    error_message: ?[]const u8,
    node_id: []const u8,
    started_at: i64,
    completed_at: i64,
};
```

### `ResultStatus`

Task result status.

```zig
pub const ResultStatus = enum {
    success,
    failure,
    timeout,
    cancelled,
    rejected,
};
```

### Encoding Functions

```zig
pub fn encodeTask(task: TaskEnvelope) ![]const u8;
pub fn decodeTask(data: []const u8) !TaskEnvelope;
pub fn encodeResult(result: ResultEnvelope) ![]const u8;
pub fn decodeResult(data: []const u8) !ResultEnvelope;
```

---

## High Availability

### `HealthCheck`

Node health checking.

```zig
pub const HealthCheck = struct {
    pub fn init(allocator: Allocator, config: HealthCheckConfig) !HealthCheck;
    pub fn check(self: *HealthCheck, node: *NodeInfo) !HealthCheckResult;
    pub fn startMonitoring(self: *HealthCheck) !void;
    pub fn stopMonitoring(self: *HealthCheck) void;
};
```

### `ClusterConfig`

Cluster configuration.

```zig
pub const ClusterConfig = struct {
    cluster_name: []const u8,
    node_id: []const u8,
    seed_nodes: []const []const u8,
    heartbeat_interval_ms: u64 = 1000,
    failure_detection_timeout_ms: u64 = 5000,
    min_cluster_size: u32 = 1,
    replication_factor: u32 = 3,
};
```

### `NodeHealth`

Detailed node health information.

```zig
pub const NodeHealth = struct {
    node_id: []const u8,
    status: NodeStatus,
    cpu_usage: f32,
    memory_usage: f32,
    disk_usage: f32,
    network_latency_ms: f32,
    active_connections: u32,
    last_check: i64,
};
```

### `ClusterState`

Overall cluster state.

```zig
pub const ClusterState = enum {
    forming,
    healthy,
    degraded,
    split_brain,
    recovery,
    shutdown,
};
```

### `HealthCheckResult`

Health check result.

```zig
pub const HealthCheckResult = struct {
    healthy: bool,
    latency_ms: f32,
    checks_passed: u32,
    checks_failed: u32,
    details: ?[]const u8,
};
```

### `FailoverPolicy`

Failover policies.

```zig
pub const FailoverPolicy = enum {
    automatic,
    manual,
    quorum_based,
    none,
};
```

---

## Service Discovery

### `ServiceDiscovery`

Service discovery mechanism.

```zig
pub const ServiceDiscovery = struct {
    pub fn init(allocator: Allocator, config: DiscoveryConfig) !ServiceDiscovery;
    pub fn deinit(self: *ServiceDiscovery) void;
    pub fn register(self: *ServiceDiscovery, service: ServiceInstance) !void;
    pub fn deregister(self: *ServiceDiscovery, service_id: []const u8) !void;
    pub fn discover(self: *ServiceDiscovery, service_name: []const u8) ![]ServiceInstance;
    pub fn watch(self: *ServiceDiscovery, service_name: []const u8, callback: WatchCallback) !WatchHandle;
};
```

### `DiscoveryConfig`

Discovery configuration.

```zig
pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .mdns,
    refresh_interval_ms: u64 = 5000,
    cache_ttl_ms: u64 = 30000,
    namespace: ?[]const u8 = null,
};
```

### `DiscoveryBackend`

Discovery backends.

```zig
pub const DiscoveryBackend = enum {
    mdns,
    consul,
    etcd,
    kubernetes,
    static,
};
```

### `ServiceInstance`

Service instance information.

```zig
pub const ServiceInstance = struct {
    id: []const u8,
    name: []const u8,
    address: []const u8,
    port: u16,
    status: ServiceStatus,
    tags: []const []const u8,
    metadata: ?[]const u8,
    health_check_url: ?[]const u8,
};
```

### `ServiceStatus`

Service status.

```zig
pub const ServiceStatus = enum {
    passing,
    warning,
    critical,
    maintenance,
};
```

---

## Load Balancing

### `LoadBalancer`

Load balancer for distributing requests.

```zig
pub const LoadBalancer = struct {
    pub fn init(allocator: Allocator, config: LoadBalancerConfig) !LoadBalancer;
    pub fn deinit(self: *LoadBalancer) void;
    pub fn addNode(self: *LoadBalancer, node: NodeInfo, weight: u32) !void;
    pub fn removeNode(self: *LoadBalancer, node_id: []const u8) !void;
    pub fn getNext(self: *LoadBalancer) !*NodeInfo;
    pub fn reportSuccess(self: *LoadBalancer, node_id: []const u8, latency_ms: f32) void;
    pub fn reportFailure(self: *LoadBalancer, node_id: []const u8) void;
    pub fn getStats(self: *LoadBalancer) []NodeStats;
};
```

### `LoadBalancerConfig`

Load balancer configuration.

```zig
pub const LoadBalancerConfig = struct {
    strategy: LoadBalancerStrategy = .round_robin,
    health_check_enabled: bool = true,
    health_check_interval_ms: u64 = 5000,
    failure_threshold: u32 = 3,
    recovery_threshold: u32 = 2,
};
```

### `LoadBalancerStrategy`

Load balancer strategies.

```zig
pub const LoadBalancerStrategy = enum {
    round_robin,
    weighted_round_robin,
    least_connections,
    least_latency,
    random,
    ip_hash,
    consistent_hash,
};
```

### `NodeState`

Node state in load balancer.

```zig
pub const NodeState = enum {
    active,
    inactive,
    draining,
    failed,
};
```

### `NodeStats`

Per-node statistics.

```zig
pub const NodeStats = struct {
    node_id: []const u8,
    state: NodeState,
    weight: u32,
    active_connections: u32,
    total_requests: u64,
    failed_requests: u64,
    avg_latency_ms: f32,
    success_rate: f32,
};
```

---

## Retry Logic

### `RetryConfig`

Retry configuration.

```zig
pub const RetryConfig = struct {
    max_retries: u32 = 3,
    initial_delay_ms: u64 = 100,
    max_delay_ms: u64 = 10000,
    strategy: RetryStrategy = .exponential_backoff,
    jitter: bool = true,
    retryable_errors: []const ErrorType = &.{},
};
```

### `RetryStrategy`

Retry strategies.

```zig
pub const RetryStrategy = enum {
    fixed,
    linear,
    exponential_backoff,
    fibonacci,
};
```

### `RetryExecutor`

Execute operations with retry logic.

```zig
pub const RetryExecutor = struct {
    pub fn init(config: RetryConfig) RetryExecutor;
    pub fn execute(self: *RetryExecutor, operation: *const fn() anyerror!T) !RetryResult(T);
    pub fn executeAsync(self: *RetryExecutor, operation: *const fn() anyerror!T) !Future(RetryResult(T));
};
```

### `RetryResult`

Result of a retried operation.

```zig
pub fn RetryResult(comptime T: type) type {
    return struct {
        value: T,
        attempts: u32,
        total_delay_ms: u64,
    };
}
```

---

## Rate Limiting

### `rate_limiter`

Rate limiting for request throttling.

```zig
pub const rate_limiter = struct {
    pub const RateLimiter = struct {
        pub fn init(config: RateLimiterConfig) RateLimiter;
        pub fn acquire(self: *RateLimiter) !void;
        pub fn tryAcquire(self: *RateLimiter) bool;
        pub fn release(self: *RateLimiter) void;
        pub fn getAvailable(self: *RateLimiter) u32;
    };

    pub const RateLimiterConfig = struct {
        requests_per_second: u32 = 100,
        burst_size: u32 = 10,
        algorithm: RateLimitAlgorithm = .token_bucket,
    };

    pub const RateLimitAlgorithm = enum {
        token_bucket,
        leaky_bucket,
        sliding_window,
        fixed_window,
    };
};
```

---

## Connection Pool

### `connection_pool`

Connection pooling for network connections.

```zig
pub const connection_pool = struct {
    pub const ConnectionPool = struct {
        pub fn init(allocator: Allocator, config: PoolConfig) !ConnectionPool;
        pub fn deinit(self: *ConnectionPool) void;
        pub fn acquire(self: *ConnectionPool) !*Connection;
        pub fn release(self: *ConnectionPool, conn: *Connection) void;
        pub fn getStats(self: *ConnectionPool) PoolStats;
    };

    pub const PoolConfig = struct {
        min_connections: u32 = 1,
        max_connections: u32 = 10,
        idle_timeout_ms: u64 = 60000,
        max_lifetime_ms: u64 = 3600000,
        acquire_timeout_ms: u64 = 5000,
    };

    pub const PoolStats = struct {
        total_connections: u32,
        active_connections: u32,
        idle_connections: u32,
        waiting_requests: u32,
    };
};
```

---

## Circuit Breaker

### `circuit_breaker`

Circuit breaker for fault tolerance.

```zig
pub const circuit_breaker = struct {
    pub const CircuitBreaker = struct {
        pub fn init(config: CircuitBreakerConfig) CircuitBreaker;
        pub fn execute(self: *CircuitBreaker, operation: *const fn() anyerror!T) !T;
        pub fn getState(self: *CircuitBreaker) CircuitState;
        pub fn reset(self: *CircuitBreaker) void;
        pub fn forceOpen(self: *CircuitBreaker) void;
    };

    pub const CircuitBreakerConfig = struct {
        failure_threshold: u32 = 5,
        success_threshold: u32 = 3,
        timeout_ms: u64 = 30000,
        half_open_max_calls: u32 = 3,
    };

    pub const CircuitState = enum {
        closed,
        open,
        half_open,
    };
};
```

---

## Raft Consensus

### `raft`

Raft consensus protocol.

```zig
pub const raft = struct {
    pub const RaftNode = struct {
        pub fn init(allocator: Allocator, config: RaftConfig) !RaftNode;
        pub fn deinit(self: *RaftNode) void;
        pub fn start(self: *RaftNode) !void;
        pub fn stop(self: *RaftNode) void;
        pub fn propose(self: *RaftNode, data: []const u8) !void;
        pub fn getState(self: *RaftNode) RaftState;
        pub fn isLeader(self: *RaftNode) bool;
        pub fn getLeader(self: *RaftNode) ?[]const u8;
    };

    pub const RaftConfig = struct {
        node_id: []const u8,
        peers: []const []const u8,
        election_timeout_min_ms: u64 = 150,
        election_timeout_max_ms: u64 = 300,
        heartbeat_interval_ms: u64 = 50,
        snapshot_threshold: u64 = 10000,
    };

    pub const RaftState = enum {
        follower,
        candidate,
        leader,
    };
};
```

---

## Transport

### `transport`

Network transport layer.

```zig
pub const transport = struct {
    pub const Transport = struct {
        pub fn init(allocator: Allocator, config: TransportConfig) !Transport;
        pub fn deinit(self: *Transport) void;
        pub fn send(self: *Transport, address: []const u8, data: []const u8) !void;
        pub fn receive(self: *Transport) !Message;
        pub fn broadcast(self: *Transport, data: []const u8) !void;
    };

    pub const TransportConfig = struct {
        bind_address: []const u8,
        bind_port: u16,
        protocol: Protocol = .tcp,
        tls_enabled: bool = false,
        tls_config: ?TlsConfig = null,
    };

    pub const Protocol = enum {
        tcp,
        udp,
        quic,
    };
};
```

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-network` | true | Distributed network module |

---

## Related Documentation

- [Network Guide](network.md) - Comprehensive network guide
- [Architecture Overview](architecture/overview.md) - System architecture

---

*See also: [Framework API](api_abi.md) | [AI API](api_ai.md) | [Database API](api_database.md)*
