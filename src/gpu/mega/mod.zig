//! Mega GPU Orchestration Module
//!
//! Provides unified cross-backend GPU orchestration with:
//! - Simultaneous CUDA + Vulkan + Metal operation
//! - Intelligent workload scheduling
//! - Learning-based optimization
//! - Real-time monitoring integration
//! - Power monitoring and eco-mode
//! - Priority workload queuing
//! - Auto-failover with circuit breaker
//! - Prometheus metrics export
//!
//! ## Overview
//!
//! The Mega module enables applications to leverage multiple GPU backends
//! simultaneously, routing workloads to the optimal backend based on:
//!
//! - Workload characteristics (compute vs memory bound)
//! - Backend availability and health
//! - Historical performance data
//! - Precision requirements
//!
//! ## Quick Start
//!
//! ```zig
//! const mega = @import("mega/mod.zig");
//!
//! // Initialize coordinator
//! var coordinator = try mega.Coordinator.init(allocator);
//! defer coordinator.deinit();
//!
//! // Define workload profile
//! const profile = mega.WorkloadProfile{
//!     .compute_intensity = 0.8,
//!     .memory_requirement_mb = 2048,
//!     .is_training = true,
//!     .category = .matrix_multiply,
//! };
//!
//! // Get scheduling decision
//! const decision = coordinator.schedule(profile);
//! std.debug.print("Scheduled to {t} backend\n", .{decision.backend_type});
//!
//! // Execute workload and record outcome
//! const start = std.time.milliTimestamp();
//! // ... execute workload on decision.backend_type ...
//! const elapsed = std.time.milliTimestamp() - start;
//! try coordinator.recordOutcome(decision, @intCast(elapsed), true);
//! ```
//!
//! ## Architecture
//!
//! The module is organized as follows:
//!
//! - `Coordinator`: Main orchestration struct managing all backends
//! - `BackendInstance`: Metadata and capabilities for each backend
//! - `WorkloadProfile`: Description of workload characteristics
//! - `ScheduleDecision`: Routing decision with confidence score
//! - `PowerMonitor`: Energy tracking and eco-mode scoring
//! - `WorkloadQueue`: Priority-based workload queuing
//! - `FailoverManager`: Circuit breaker and auto-failover
//! - `MetricsExporter`: Prometheus-compatible metrics export
//!
//! ## Thread Safety
//!
//! The Coordinator uses internal synchronization for thread-safe operation.
//! Multiple threads can call `schedule()` concurrently.

pub const coordinator = @import("coordinator.zig");
pub const scheduler = @import("scheduler.zig");
pub const hybrid = @import("hybrid.zig");
pub const queue = @import("queue.zig");
pub const failover = @import("failover.zig");
pub const power = @import("power.zig");
pub const metrics = @import("metrics.zig");

// Re-export main types
pub const Coordinator = coordinator.Coordinator;
pub const BackendInstance = coordinator.BackendInstance;
pub const WorkloadProfile = coordinator.WorkloadProfile;
pub const WorkloadCategory = coordinator.WorkloadCategory;
pub const ScheduleDecision = coordinator.ScheduleDecision;
pub const CoordinatorStats = coordinator.Coordinator.CoordinatorStats;
pub const Precision = coordinator.Precision;

// Re-export scheduler types
pub const LearningScheduler = scheduler.LearningScheduler;
pub const LearningStats = scheduler.LearningStats;
pub const Experience = scheduler.Experience;
pub const SchedulerState = scheduler.SchedulerState;
pub const QTable = scheduler.QTable;
pub const ReplayBuffer = scheduler.ReplayBuffer;

// Re-export hybrid types
pub const HybridCoordinator = hybrid.HybridCoordinator;
pub const HybridRoutingConfig = hybrid.HybridRoutingConfig;
pub const HybridWorkload = hybrid.HybridWorkload;
pub const HybridRoutingDecision = hybrid.HybridRoutingDecision;
pub const HybridDeviceType = hybrid.HybridDeviceType;
pub const HybridWorkloadType = hybrid.HybridWorkloadType;

// Re-export queue types
pub const WorkloadQueue = queue.WorkloadQueue;
pub const QueueConfig = queue.QueueConfig;
pub const EnqueueOptions = queue.EnqueueOptions;
pub const QueueStats = queue.QueueStats;
pub const Priority = queue.Priority;
pub const QueuedWorkload = queue.QueuedWorkload;
pub const WorkloadStatus = queue.WorkloadStatus;

// Re-export failover types
pub const FailoverManager = failover.FailoverManager;
pub const FailoverPolicy = failover.FailoverPolicy;
pub const FailoverEvent = failover.FailoverEvent;
pub const FailoverReason = failover.FailoverReason;
pub const FailoverStats = failover.FailoverStats;
pub const BackendHealth = failover.BackendHealth;
pub const CircuitState = failover.CircuitState;

// Re-export power monitoring types
pub const PowerMonitor = power.PowerMonitor;
pub const BackendPowerProfile = power.BackendPowerProfile;
pub const BackendEnergyStats = power.BackendEnergyStats;
pub const EcoModeConfig = power.EcoModeConfig;
pub const EnergyReport = power.EnergyReport;
pub const default_power_profiles = power.default_profiles;

// Re-export metrics exporter types
pub const MetricsExporter = metrics.MetricsExporter;
pub const BackendMetrics = metrics.BackendMetrics;
pub const HistogramValue = metrics.HistogramValue;
pub const Counter = metrics.Counter;
pub const Gauge = metrics.Gauge;
pub const default_latency_buckets = metrics.default_latency_buckets;

// Re-export convenience functions
pub const init = Coordinator.init;

test {
    // Run all tests in submodules
    _ = coordinator;
    _ = scheduler;
    _ = hybrid;
    _ = power;
    _ = queue;
    _ = failover;
    _ = metrics;
}
