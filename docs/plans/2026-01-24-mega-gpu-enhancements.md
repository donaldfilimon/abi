# Mega GPU Enhancements Design

**Date:** 2026-01-24
**Status:** Ready for Implementation

## Overview

Add four new components to the Mega GPU orchestration system using a Coordinator-Centric architecture with hybrid integration (tight integration for scheduler/eco-mode, observer-style for metrics).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Coordinator                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Scheduler  │◄─┤ PowerMonitor │  │   Metrics    │           │
│  │  (Q-learning)│  │  (eco-mode)  │  │  (observer)  │           │
│  └──────┬───────┘  └──────────────┘  └──────────────┘           │
│         │                                    ▲                   │
│         ▼                                    │ records           │
│  ┌──────────────┐  ┌──────────────┐          │                   │
│  │WorkloadQueue │──►FailoverMgr  │──────────┘                   │
│  │  (priority)  │  │(circuit brk) │                              │
│  └──────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Task 1: Create power.zig
- PowerMonitor struct with mutex
- BackendPowerProfile with TDP/idle/peak watts
- Default profiles for all backends (CUDA 250W, Vulkan 200W, Metal 60W, etc.)
- recordWorkload() to track energy consumption
- getEcoScore() for scheduler integration
- EcoModeConfig with carbon intensity
- Unit test

### Task 2: Create queue.zig
- Priority enum (critical, high, normal, low, background)
- QueuedWorkload with deadline, retry support
- WorkloadQueue with 5 priority queues
- enqueue/dequeue/cancel/complete API
- QueueStats for depth tracking
- Unit test

### Task 3: Create failover.zig
- CircuitState enum (closed, open, half_open)
- BackendHealth tracking per backend
- FailoverPolicy with thresholds and backoff
- Circuit breaker state machine
- failover() with wraparound to find next healthy backend
- Unit test

### Task 4: Create metrics.zig
- MetricsExporter with Prometheus format
- HistogramValue with standard buckets
- BackendMetrics per-backend tracking
- recordWorkload/recordFailover API
- exportPrometheus() with snapshot pattern (clone under lock, write without lock)
- Unit test

### Task 5: Update mod.zig
- Import all 4 new modules
- Re-export public types
- Add to test block

### Task 6: Verification
- Run `zig build` to verify compilation
- Run `zig build test --summary all` to verify all tests pass
- Verify no regressions (250/254 baseline)

## Key Design Decisions

1. **getOrPut not getOrPutAssumeCapacity** in metrics.zig to prevent panic
2. **Failover wraps around** priority list to find any healthy backend
3. **Snapshot pattern** in exportPrometheus to avoid blocking during I/O
4. **Each component standalone** with optional Coordinator integration
5. **Metrics are best-effort** - failures don't block workloads

## Files to Create/Modify

| File | Action | Lines (est.) |
|------|--------|--------------|
| src/gpu/mega/power.zig | Create | ~200 |
| src/gpu/mega/queue.zig | Create | ~180 |
| src/gpu/mega/failover.zig | Create | ~220 |
| src/gpu/mega/metrics.zig | Create | ~200 |
| src/gpu/mega/mod.zig | Modify | +40 |
