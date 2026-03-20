---
title: messaging API
purpose: Generated API reference for messaging
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# messaging

> Messaging Module

Topic-based pub/sub messaging with MQTT-style pattern matching,
synchronous delivery, dead letter queue, and backpressure.

Architecture:
- Topic registry with subscriber lists
- Pattern matching: `events.*` (single-level), `events.#` (multi-level)
- Bounded message queues with backpressure (returns ChannelFull)
- Dead letter queue for failed deliveries
- Synchronous delivery (publish blocks until all subscribers notified)
- RwLock for concurrent topic lookups

**Source:** [`src/features/messaging/mod.zig`](../../src/features/messaging/mod.zig)

**Build flag:** `-Dfeat_messaging=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-messagingconfig-messagingerror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: MessagingConfig) MessagingError!void`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L238)

Initialize the global messaging singleton.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L244)

Tear down the messaging subsystem and all topics/subscribers.

### <a id="pub-fn-publish-allocator-std-mem-allocator-topic-name-const-u8-payload-const-u8-messagingerror-void"></a>`pub fn publish( allocator: std.mem.Allocator, topic_name: []const u8, payload: []const u8, ) MessagingError!void`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L261)

Publish a message to a topic. Delivers synchronously to all matching
subscribers (exact match and MQTT-style wildcards `*`, `#`).

### <a id="pub-fn-subscribe-std-mem-allocator-topic-pattern-const-u8-callback-subscribercallback-user-ctx-anyopaque-messagingerror-u64"></a>`pub fn subscribe( _: std.mem.Allocator, topic_pattern: []const u8, callback: SubscriberCallback, user_ctx: ?*anyopaque, ) MessagingError!u64`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L323)

Register a callback for messages matching `topic_pattern`.
Returns a subscriber ID that can be passed to `unsubscribe`.

### <a id="pub-fn-unsubscribe-subscriber-id-u64-messagingerror-bool"></a>`pub fn unsubscribe(subscriber_id: u64) MessagingError!bool`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L368)

Remove a subscriber by ID. Returns `true` if the subscriber was found.

### <a id="pub-fn-listtopics-allocator-std-mem-allocator-messagingerror-const-u8"></a>`pub fn listTopics(allocator: std.mem.Allocator) MessagingError![][]const u8`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L396)

List all active topic names. Caller owns the returned slice.

### <a id="pub-fn-getdeadletters-allocator-std-mem-allocator-messagingerror-deadletter"></a>`pub fn getDeadLetters(allocator: std.mem.Allocator) MessagingError![]DeadLetter`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L435)

Retrieve all dead letter entries. Caller owns the returned slice.

### <a id="pub-fn-cleardeadletters-void"></a>`pub fn clearDeadLetters() void`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L461)

Discard all dead letter entries.

### <a id="pub-fn-messagingstats-messagingstats"></a>`pub fn messagingStats() MessagingStats`

<sup>**fn**</sup> | [source](../../src/features/messaging/mod.zig#L475)

Snapshot publish/deliver/fail counters and active topic/subscriber counts.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
