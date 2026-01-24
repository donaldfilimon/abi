//! Mega GPU Orchestration Module
//!
//! Provides unified cross-backend GPU orchestration with:
//! - Simultaneous CUDA + Vulkan + Metal operation
//! - Intelligent workload scheduling
//! - Learning-based optimization
//! - Real-time monitoring integration
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
//! std.debug.print("Scheduled to {s} backend\n", .{@tagName(decision.backend_type)});
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
//!
//! ## Thread Safety
//!
//! The Coordinator uses internal synchronization for thread-safe operation.
//! Multiple threads can call `schedule()` concurrently.

pub const coordinator = @import("coordinator.zig");
pub const scheduler = @import("scheduler.zig");

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

// Re-export convenience functions
pub const init = Coordinator.init;

test {
    // Run all tests in submodules
    _ = coordinator;
    _ = scheduler;
}
