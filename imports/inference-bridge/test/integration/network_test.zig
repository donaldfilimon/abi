//! Integration Tests: Network Module
//!
//! Verifies network module type exports, Raft consensus types,
//! transport types, and basic API contracts without making real
//! network connections.

const std = @import("std");
const abi = @import("abi");

const network = abi.network;

// ============================================================================
// Core type availability
// ============================================================================

test "network: NodeRegistry type exists" {
    const NR = network.NodeRegistry;
    _ = NR;
}

test "network: NodeInfo type exists" {
    const NI = network.NodeInfo;
    _ = NI;
}

test "network: NodeStatus type exists" {
    const NS = network.NodeStatus;
    _ = NS;
}

test "network: NetworkError type exists" {
    const NE = network.NetworkError;
    _ = NE;
}

test "network: NetworkConfig type exists" {
    const NC = network.NetworkConfig;
    _ = NC;
}

test "network: Context type exists" {
    const Ctx = network.Context;
    _ = Ctx;
}

// ============================================================================
// Raft consensus types
// ============================================================================

test "network: RaftNode type exists" {
    const RN = network.RaftNode;
    _ = RN;
}

test "network: RaftState type exists" {
    const RS = network.RaftState;
    _ = RS;
}

test "network: RaftConfig type exists" {
    const RC = network.RaftConfig;
    _ = RC;
}

test "network: RaftError type exists" {
    const RE = network.RaftError;
    _ = RE;
}

test "network: LogEntry type exists" {
    const LE = network.LogEntry;
    _ = LE;
}

test "network: RequestVoteRequest type exists" {
    const RVR = network.RequestVoteRequest;
    _ = RVR;
}

test "network: RequestVoteResponse type exists" {
    const RVR = network.RequestVoteResponse;
    _ = RVR;
}

test "network: AppendEntriesRequest type exists" {
    const AER = network.AppendEntriesRequest;
    _ = AER;
}

test "network: AppendEntriesResponse type exists" {
    const AER = network.AppendEntriesResponse;
    _ = AER;
}

// ============================================================================
// Scheduler types
// ============================================================================

test "network: TaskScheduler type exists" {
    const TS = network.TaskScheduler;
    _ = TS;
}

test "network: SchedulerConfig type exists" {
    const SC = network.SchedulerConfig;
    _ = SC;
}

test "network: TaskPriority type exists" {
    const TP = network.TaskPriority;
    _ = TP;
}

test "network: TaskState type exists" {
    const TS = network.TaskState;
    _ = TS;
}

test "network: LoadBalancingStrategy type exists" {
    const LBS = network.LoadBalancingStrategy;
    _ = LBS;
}

// ============================================================================
// High Availability types
// ============================================================================

test "network: HealthCheck type exists" {
    const HC = network.HealthCheck;
    _ = HC;
}

test "network: ClusterConfig type exists" {
    const CC = network.ClusterConfig;
    _ = CC;
}

test "network: ClusterState type exists" {
    const CS = network.ClusterState;
    _ = CS;
}

test "network: FailoverPolicy type exists" {
    const FP = network.FailoverPolicy;
    _ = FP;
}

// ============================================================================
// Service Discovery types
// ============================================================================

test "network: ServiceDiscovery type exists" {
    const SD = network.ServiceDiscovery;
    _ = SD;
}

test "network: DiscoveryConfig type exists" {
    const DC = network.DiscoveryConfig;
    _ = DC;
}

test "network: ServiceInstance type exists" {
    const SI = network.ServiceInstance;
    _ = SI;
}

test "network: ServiceStatus type exists" {
    const SS = network.ServiceStatus;
    _ = SS;
}

// ============================================================================
// Circuit Breaker types
// ============================================================================

test "network: CircuitBreaker type exists" {
    const CB = network.CircuitBreaker;
    _ = CB;
}

test "network: CircuitConfig type exists" {
    const CC = network.CircuitConfig;
    _ = CC;
}

test "network: CircuitState type exists" {
    const CS = network.CircuitState;
    _ = CS;
}

// ============================================================================
// Transport types
// ============================================================================

test "network: TcpTransport type exists" {
    const TT = network.TcpTransport;
    _ = TT;
}

test "network: TransportConfig type exists" {
    const TC = network.TransportConfig;
    _ = TC;
}

test "network: MessageType type exists" {
    const MT = network.MessageType;
    _ = MT;
}

test "network: MessageHeader type exists" {
    const MH = network.MessageHeader;
    _ = MH;
}

// ============================================================================
// Rate Limiter types
// ============================================================================

test "network: RateLimiter type exists" {
    const RL = network.RateLimiter;
    _ = RL;
}

test "network: RateLimiterConfig type exists" {
    const RLC = network.RateLimiterConfig;
    _ = RLC;
}

test "network: RateLimitAlgorithm type exists" {
    const RLA = network.RateLimitAlgorithm;
    _ = RLA;
}

// ============================================================================
// Connection Pool types
// ============================================================================

test "network: ConnectionPool type exists" {
    const CP = network.ConnectionPool;
    _ = CP;
}

test "network: ConnectionPoolConfig type exists" {
    const CPC = network.ConnectionPoolConfig;
    _ = CPC;
}

test "network: ConnectionState type exists" {
    const CS = network.ConnectionState;
    _ = CS;
}

// ============================================================================
// Retry types
// ============================================================================

test "network: RetryConfig type exists" {
    const RC = network.RetryConfig;
    _ = RC;
}

test "network: RetryStrategy type exists" {
    const RS = network.RetryStrategy;
    _ = RS;
}

test "network: RetryExecutor type exists" {
    const RE = network.RetryExecutor;
    _ = RE;
}

// ============================================================================
// NodeRegistry basic operations
// ============================================================================

test "network: NodeRegistry init and deinit" {
    var reg = network.NodeRegistry.init(std.testing.allocator);
    defer reg.deinit();

    const nodes = reg.list();
    try std.testing.expectEqual(@as(usize, 0), nodes.len);
}

test {
    std.testing.refAllDecls(@This());
}
