---
title: "Messaging"
description: "Event bus, pub/sub, and message queues"
section: "Infrastructure"
order: 3
---

# Messaging

The Messaging module provides topic-based pub/sub messaging with MQTT-style
wildcard pattern matching, synchronous delivery, dead letter queues, and
backpressure support.

- **Build flag:** `-Denable-messaging=true` (default: enabled)
- **Namespace:** `abi.messaging`
- **Source:** `src/features/messaging/`

## Overview

The messaging module is a lightweight, in-process event bus for decoupling
components. Publishers send messages to named topics, and subscribers receive
them through callback functions registered against topic patterns. The module
supports:

- **Topic-based pub/sub** -- Publish to named topics, subscribe with callbacks
- **MQTT-style wildcards** -- `*` matches a single level, `#` matches zero or more levels
- **Synchronous delivery** -- Publish blocks until all matching subscribers are notified
- **Dead letter queue** -- Messages that subscribers reject are captured for later inspection
- **Backpressure** -- Bounded message queues return `error.ChannelFull` when capacity is reached
- **Auto-created topics** -- Topics are created on first publish or subscribe
- **Subscriber management** -- Subscribe returns an ID; pass it to `unsubscribe` to remove
- **RwLock concurrency** -- Concurrent topic lookups with exclusive writes

## Quick Start

```zig
const abi = @import("abi");

// Initialize via Framework
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withMessagingDefaults()
    .build();
defer framework.deinit();

// Define a subscriber callback
fn onUserEvent(msg: abi.messaging.Message, _: ?*anyopaque) abi.messaging.DeliveryResult {
    std.debug.print("[{s}] {s}\n", .{ msg.topic, msg.payload });
    return .ok;
}

// Subscribe with MQTT wildcard pattern
const sub_id = try abi.messaging.subscribe(
    allocator,
    "users.*",     // matches users.login, users.signup, etc.
    onUserEvent,
    null,          // optional user context pointer
);

// Publish messages
try abi.messaging.publish(allocator, "users.login", "alice logged in");
try abi.messaging.publish(allocator, "users.signup", "bob signed up");

// Unsubscribe when done
_ = try abi.messaging.unsubscribe(sub_id);

// Check stats
const s = abi.messaging.messagingStats();
std.debug.print("{} published, {} delivered\n", .{
    s.total_published, s.total_delivered,
});
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context |
| `MessagingConfig` | Max channels, buffer size, persistence toggle |
| `Message` | A delivered message: topic, payload, timestamp, id |
| `Channel` | A named channel with subscriber count |
| `DeliveryResult` | Subscriber return value: `ok`, `retry`, or `discard` |
| `SubscriberCallback` | `*const fn (Message, ?*anyopaque) DeliveryResult` |
| `MessagingStats` | Aggregate counters: published, delivered, failed, topics, subscribers, dead letters |
| `TopicInfo` | Per-topic stats: name, subscriber count, published/delivered/failed counts |
| `DeadLetter` | A failed message with reason and timestamp |
| `MessagingError` | Error set: FeatureDisabled, ChannelFull, ChannelClosed, TopicNotFound, etc. |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator, config) !void` | Initialize the messaging singleton |
| `deinit() void` | Tear down all topics, subscribers, and dead letters |
| `isEnabled() bool` | Returns `true` if messaging is compiled in |
| `isInitialized() bool` | Returns `true` if the singleton is active |
| `publish(allocator, topic, payload) !void` | Publish a message to all matching subscribers |
| `subscribe(allocator, pattern, callback, ctx) !u64` | Register a subscriber; returns subscriber ID |
| `unsubscribe(subscriber_id) !bool` | Remove a subscriber by ID; returns `true` if found |
| `listTopics(allocator) ![][]const u8` | List all active topic names (caller owns slice) |
| `topicStats(topic_name) !TopicInfo` | Get per-topic statistics |
| `getDeadLetters(allocator) ![]DeadLetter` | Retrieve all dead letter entries (caller owns slice) |
| `clearDeadLetters() void` | Discard all dead letters |
| `messagingStats() MessagingStats` | Snapshot of aggregate counters |

### MQTT-style Pattern Matching

Topic patterns use dot-separated levels with two wildcard operators:

| Pattern | Matches | Does Not Match |
|---------|---------|----------------|
| `users.login` | `users.login` | `users.signup` |
| `users.*` | `users.login`, `users.signup` | `users.login.attempt` |
| `users.#` | `users.login`, `users.signup`, `users.login.attempt` | `system.health` |
| `#` | Everything | (matches all topics) |

- `*` matches exactly **one** level
- `#` matches **zero or more** levels (must be the last segment)

### Delivery Results

Subscriber callbacks return a `DeliveryResult` that controls message acknowledgement:

| Value | Behavior |
|-------|----------|
| `.ok` | Message delivered successfully; counted in `delivered` stats |
| `.retry` | Delivery failed; counted in `failed` stats |
| `.discard` | Delivery rejected; message sent to dead letter queue |

## Configuration

Messaging is configured through the `MessagingConfig` struct:

```zig
const config = abi.messaging.MessagingConfig{
    .max_channels = 256,     // maximum number of topics
    .buffer_size = 4096,     // dead letter queue capacity
    .enable_persistence = false,
};
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_channels` | 256 | Maximum number of concurrent topics |
| `buffer_size` | 4096 | Dead letter queue capacity |
| `enable_persistence` | `false` | Enable persistent message storage |

## CLI Commands

The messaging module does not have a dedicated CLI command. Use the messaging
API programmatically or through the Framework builder.

## Examples

See `examples/messaging.zig` for a complete working example that subscribes
with wildcard patterns, publishes messages, lists topics, and prints stats:

```bash
zig build run-messaging
```

## Disabling at Build Time

```bash
# Compile without messaging support
zig build -Denable-messaging=false
```

When disabled, all public functions return `error.FeatureDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures so downstream code compiles without conditional guards.

## Related

- [Network](network.html) -- Distributed compute and node management
- [Gateway](gateway.html) -- API gateway with routing and rate limiting
- [Web](web.html) -- HTTP client utilities
