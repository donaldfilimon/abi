# runtime-concurrency API Reference

> Lock-free concurrent primitives

**Source:** [`src/runtime/concurrency/mod.zig`](../../src/runtime/concurrency/mod.zig)

---

Concurrency Primitives for Runtime

This module provides lock-free and thread-safe data structures
for concurrent task execution.

## Available Types

- `WorkStealingQueue` - Owner pushes/pops from back, thieves steal from front
- `WorkQueue` - Simple FIFO queue with mutex
- `LockFreeQueue` - CAS-based queue (SPSC)
- `LockFreeStack` - CAS-based stack (WARNING: ABA susceptible)
- `LockFreeStackEBR` - ABA-safe stack with epoch-based reclamation
- `ChaseLevDeque` - Lock-free work-stealing deque (Chase-Lev algorithm)
- `MpmcQueue` - Lock-free multi-producer multi-consumer queue
- `ShardedMap` - Partitioned map reducing contention
- `PriorityQueue` - Lock-free priority queue for task scheduling
- `Backoff` - Exponential backoff for spin-wait loops
- `EpochReclamation` - Safe memory reclamation for lock-free structures

---

## API

---

*Generated automatically by `zig build gendocs`*
