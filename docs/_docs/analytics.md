---
title: "Analytics"
description: "Event tracking and experiment framework"
section: "Operations"
order: 2
---

# Analytics

The analytics module provides business event tracking, session management,
funnel instrumentation, and A/B experiment assignment. Unlike
[Observability](observability.html) (system metrics, tracing, profiling),
analytics focuses on user-facing events and product usage patterns.

- **Build flag:** `-Denable-analytics=true` (default: enabled)
- **Namespace:** `abi.analytics`
- **Source:** `src/features/analytics/`

## Overview

The analytics module is built around a thread-safe `Engine` that buffers events
in memory with configurable capacity and flush semantics. It supports:

- **Event Tracking** -- Named events with optional typed properties and session association
- **Session Management** -- Atomic session ID generation and per-session tracking
- **Funnel Tracking** -- Multi-step funnels with per-step counters for conversion analysis
- **A/B Experiments** -- Deterministic variant assignment based on user ID hashing (Fnv1a_64)
- **Statistics** -- Real-time stats including buffered count, total events, and total sessions

All counters use `std.atomic.Value(u64)` for lock-free concurrent access. The
event buffer itself is protected by a mutex.

## Quick Start

```zig
const abi = @import("abi");

// Create an analytics engine
var engine = abi.analytics.Engine.init(allocator, .{
    .buffer_capacity = 1024,
    .flush_interval_ms = 5000,
});
defer engine.deinit();

// Track events
try engine.track("page_view");
try engine.track("button_click");

// Track with session context
const session_id = engine.startSession();
try engine.trackWithSession("purchase", "session-42");

// Get stats
const stats = engine.getStats();
// stats.buffered_events, stats.total_events, stats.total_sessions

// Flush buffered events
const flushed = engine.flush();
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Engine` | Core analytics engine with buffered event tracking |
| `Engine.Stats` | Snapshot of engine statistics (`buffered_events`, `total_events`, `total_sessions`) |
| `Event` | Single analytics event with name, timestamp, session ID, and properties |
| `Event.Property` | Key-value property on an event |
| `Event.Value` | Property value: `string`, `int`, `float`, or `boolean` |
| `AnalyticsConfig` | Engine configuration |
| `AnalyticsError` | Error set: `BufferFull`, `InvalidEvent`, `FlushFailed`, `AnalyticsDisabled`, `OutOfMemory` |
| `Funnel` | Multi-step funnel tracker |
| `Funnel.Step` | Individual funnel step with atomic counter |
| `Experiment` | A/B experiment with deterministic variant assignment |
| `Context` | Framework integration context wrapping an `Engine` |

### Engine Functions

| Function | Description |
|----------|-------------|
| `Engine.init(allocator, config)` | Create a new analytics engine |
| `Engine.deinit()` | Release engine resources |
| `Engine.track(name)` | Track a named event |
| `Engine.trackWithSession(name, session_id)` | Track an event associated with a session |
| `Engine.startSession()` | Atomically generate a new session ID (returns `u64`) |
| `Engine.bufferedCount()` | Number of events currently in the buffer |
| `Engine.totalEvents()` | Total events tracked (including flushed) |
| `Engine.flush()` | Clear the buffer; returns the number of events flushed |
| `Engine.getStats()` | Snapshot of current statistics |

### Funnel Functions

| Function | Description |
|----------|-------------|
| `Funnel.init(allocator, name)` | Create a named funnel |
| `Funnel.deinit()` | Release funnel resources |
| `Funnel.addStep(step_name)` | Append a step to the funnel |
| `Funnel.recordStep(step_index)` | Increment a step's counter atomically |
| `Funnel.getStepCounts(buffer)` | Copy step counts into a buffer |

### Experiment Functions

| Function | Description |
|----------|-------------|
| `Experiment.assign(user_id)` | Assign a user to a variant (deterministic via Fnv1a_64 hash) |
| `Experiment.totalAssignments()` | Total number of assignments made |

## Configuration

```zig
const config = abi.analytics.AnalyticsConfig{
    .buffer_capacity = 1024,       // Max events before auto-flush / BufferFull error
    .enable_timestamps = true,     // Attach monotonic timestamps to events
    .app_id = "my-service",        // Application identifier
    .flush_interval_ms = 5000,     // Flush interval hint (0 = manual only)
};
```

## Funnel Example

```zig
var funnel = abi.analytics.Funnel.init(allocator, "onboarding");
defer funnel.deinit();

try funnel.addStep("signup");
try funnel.addStep("verify_email");
try funnel.addStep("complete_profile");

// Record user progression
funnel.recordStep(0); // signup
funnel.recordStep(0); // another signup
funnel.recordStep(1); // verify email
funnel.recordStep(2); // complete profile

// Analyze conversion
var buf: [3]u64 = undefined;
const counts = funnel.getStepCounts(&buf);
// counts[0] = 2, counts[1] = 1, counts[2] = 1
```

## Experiment Example

```zig
var experiment = abi.analytics.Experiment{
    .name = "color_test",
    .variants = &.{ "red", "blue", "green" },
};

// Same user always gets the same variant (deterministic)
const variant = experiment.assign("user-123");
const again = experiment.assign("user-123");
// variant == again
```

## Examples

See `examples/analytics.zig` for a full working example.

```bash
zig build run-analytics
```

## Disabling at Build Time

```bash
zig build -Denable-analytics=false
```

When disabled, `abi.analytics.isEnabled()` returns `false`. Engine operations
return `error.AnalyticsDisabled`, funnels are no-ops, and experiments always
return `"control"`.

## Related

- [Observability](observability.html) -- System-level metrics and tracing (complementary to analytics)
- [Benchmarks](benchmarks.html) -- Performance measurement tooling
