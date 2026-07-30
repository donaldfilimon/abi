//! ABI framework core: configuration, task scheduling, memory accounting, and
//! the plugin registry.
//!
//! The Rust successor to `src/core/` in the Zig tree. Depends only on
//! [`abi_foundation`].
//!
//! ## Concurrency
//!
//! The scheduler is **one-shot**: tasks run synchronously on the thread that
//! calls [`scheduler::Scheduler::run_next`] or
//! [`scheduler::Scheduler::run_all`]. There is no worker pool and no async
//! runtime. `abi scheduler status` reports `mode=one-shot`, and the Zig
//! implementation behaved the same way — its `running` counter was only ever
//! non-zero inside a single synchronous call.
//!
//! This is stated up front because `scheduler_stats` and `scheduler_info` are two
//! of the twelve frozen MCP tools. Adding a runtime would change their output and
//! break that contract, so it is a decision rather than an omission.
//!
//! [`Scheduler`] and [`MemoryTracker`] use interior mutability and are `Send +
//! Sync`, so they can be shared as `Arc<_>` across threads — which is how the AI
//! and WDBX paths took them in Zig. [`Registry`] is not synchronized; wrap it if
//! it needs to be shared.

pub mod config;
pub mod memory;
pub mod registry;
pub mod scheduler;
pub mod task;

pub use config::{AiConfig, Config, LimitConfig, LogLevel, ServerConfig};
pub use memory::{AllocationId, AllocationRecord, MemorySnapshot, MemoryTracker};
pub use registry::{
    ContextProvider, PluginCommand, PluginDescriptor, Registry, RegistryError, ResolvedCommand,
};
pub use scheduler::{Scheduler, SchedulerError, Stats};
pub use task::{TaskFn, TaskInfo, TaskPriority, TaskStatus};
