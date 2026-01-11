//! # Concurrency Utilities
//!
//! Lock-free data structures and synchronization primitives for the compute runtime.
//!
//! ## Features
//!
//! - **Lock-Free Collections**: Queue, stack, and map implementations
//! - **Work-Stealing**: LIFO for owner, FIFO for thieves
//! - **Priority Queue**: Lock-free priority-based scheduling
//! - **Sharded Map**: Reduced contention via partitioning
//! - **Backoff**: Exponential backoff with spin-loop hints
//!
//! ## Data Structures
//!
//! | Structure | Description | Use Case |
//! |-----------|-------------|----------|
//! | `WorkStealingQueue` | LIFO/FIFO hybrid | Worker thread task queues |
//! | `LockFreeQueue` | MPMC queue | High-contention producer/consumer |
//! | `LockFreeStack` | MPMC stack | Fast push/pop operations |
//! | `PriorityQueue` | Lock-free priority | Task scheduling by priority |
//! | `ShardedMap` | Partitioned map | Reduced lock contention |
//!
//! ## Usage
//!
//! ### Work-Stealing Queue
//!
//! ```zig
//! var queue = WorkStealingQueue(Task).init();
//! defer queue.deinit();
//!
//! // Owner pushes (LIFO)
//! try queue.push(task);
//! const local = queue.pop();  // Gets most recent
//!
//! // Thieves steal (FIFO)
//! const stolen = queue.steal();  // Gets oldest
//! ```
//!
//! ### Lock-Free Queue
//!
//! ```zig
//! var queue = LockFreeQueue(Message).init(allocator);
//! defer queue.deinit();
//!
//! try queue.push(msg);
//! if (queue.pop()) |m| {
//!     // Process message
//! }
//! ```
//!
//! ### Backoff
//!
//! ```zig
//! var backoff = Backoff.init();
//! while (!try_acquire_lock()) {
//!     backoff.spin();  // Spin-loop hint, then yield
//! }
//! ```
//!
//! ## See Also
//!
//! - [Compute Module](../README.md)
//! - [Runtime Engine](../runtime/README.md)

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

