---
title: analytics API
purpose: Generated API reference for analytics
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2962+08416b44f
---

# analytics

> Analytics Module

Business event tracking, session analytics, and funnel instrumentation.
Unlike observability (system metrics, tracing, profiling), analytics
focuses on user-facing events and product usage patterns.

## Features
- Custom event tracking with typed properties
- Session lifecycle management
- Funnel step recording
- A/B experiment assignment tracking
- Thread-safe event buffer with configurable flush

**Source:** [`src/features/analytics/mod.zig`](../../src/features/analytics/mod.zig)

**Build flag:** `-Dfeat_analytics=true`

---

## API

### <a id="pub-const-engine"></a>`pub const Engine`

<sup>**const**</sup> | [source](../../src/features/analytics/mod.zig#L36)

Core analytics engine. Buffers events and provides batch retrieval.

### <a id="pub-fn-track-self-engine-name-const-u8-analyticserror-void"></a>`pub fn track(self: *Engine, name: []const u8) AnalyticsError!void`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L57)

Track a named event.

### <a id="pub-fn-trackwithsession-self-engine-name-const-u8-session-id-const-u8-analyticserror-void"></a>`pub fn trackWithSession(self: *Engine, name: []const u8, session_id: ?[]const u8) AnalyticsError!void`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L62)

Track a named event associated with a session.

### <a id="pub-fn-startsession-self-engine-u64"></a>`pub fn startSession(self: *Engine) u64`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L82)

Start a new session and return its ID.

### <a id="pub-fn-bufferedcount-self-engine-usize"></a>`pub fn bufferedCount(self: *Engine) usize`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L87)

Get count of buffered events.

### <a id="pub-fn-totalevents-self-const-engine-u64"></a>`pub fn totalEvents(self: *const Engine) u64`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L94)

Get total events tracked (including flushed).

### <a id="pub-fn-flush-self-engine-usize"></a>`pub fn flush(self: *Engine) usize`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L99)

Flush all buffered events, returning the count flushed.

### <a id="pub-fn-getstats-self-engine-stats"></a>`pub fn getStats(self: *Engine) Stats`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L108)

Get a snapshot of current stats.

### <a id="pub-const-funnel"></a>`pub const Funnel`

<sup>**const**</sup> | [source](../../src/features/analytics/mod.zig#L126)

Track progression through a named funnel.

### <a id="pub-fn-addstep-self-funnel-step-name-const-u8-void"></a>`pub fn addStep(self: *Funnel, step_name: []const u8) !void`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L143)

Add a step to the funnel.

### <a id="pub-fn-recordstep-self-funnel-step-index-usize-void"></a>`pub fn recordStep(self: *Funnel, step_index: usize) void`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L148)

Record a user reaching a step.

### <a id="pub-fn-getstepcounts-self-const-funnel-buffer-u64-u64"></a>`pub fn getStepCounts(self: *const Funnel, buffer: []u64) []u64`

<sup>**fn**</sup> | [source](../../src/features/analytics/mod.zig#L155)

Get step counts for analysis.

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/features/analytics/mod.zig#L175)

Analytics context for Framework integration.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
