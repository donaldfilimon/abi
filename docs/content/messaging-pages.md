---
title: Messaging & Pages
description: Event pub/sub with MQTT patterns and dashboard page routing
section: Modules
order: 12
---

# Messaging & Pages

ABI includes a **Messaging** module for topic-based pub/sub communication and a
**Pages** module for dashboard-style URL routing with template rendering.

---

## Messaging (`abi.messaging`)

Topic-based pub/sub messaging with MQTT-style pattern matching, synchronous
delivery, dead letter queue, and backpressure support.

**Build flag:** `-Denable-messaging=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Topic registry** | Named topics with subscriber lists |
| **Pattern matching** | MQTT-style: `*` (single-level), `#` (multi-level) |
| **Delivery** | Synchronous -- `publish()` blocks until all subscribers are notified |
| **Backpressure** | Bounded message queues; returns `ChannelFull` when capacity exceeded |
| **Dead letter queue** | Failed deliveries stored with reason and timestamp |
| **Concurrency** | `RwLock` for concurrent topic lookups |

### MQTT Pattern Matching

Subscribers can register with wildcard patterns that match topic hierarchies
using dot-separated segments:

| Pattern | Matches | Does Not Match |
|---------|---------|----------------|
| `events.user.*` | `events.user.created`, `events.user.deleted` | `events.user.profile.updated` |
| `events.#` | `events.user.created`, `events.system.shutdown`, `events` | `logs.error` |
| `events.*.created` | `events.user.created`, `events.order.created` | `events.user.profile.created` |

- `*` matches exactly one level in the topic hierarchy.
- `#` matches zero or more levels (must be the last segment).

### Subscriber Callbacks

Subscribers provide a callback function that returns a `DeliveryResult`:

| Result | Behavior |
|--------|----------|
| `.ok` | Message acknowledged, delivery complete |
| `.retry` | Message will be redelivered (up to retry limit) |
| `.discard` | Message dropped without retry |

Failed deliveries (after retries exhausted) are sent to the dead letter queue.

### API

```zig
const abi = @import("abi");
const messaging = abi.messaging;

// Initialize
try messaging.init(allocator, .{
    .max_topics = 256,
    .max_subscribers_per_topic = 64,
});
defer messaging.deinit();

// Define a subscriber callback
fn onUserEvent(msg: messaging.Message, ctx: ?*anyopaque) messaging.DeliveryResult {
    _ = ctx;
    std.log.info("[{s}] {s}", .{ msg.topic, msg.payload });
    return .ok;
}

// Subscribe to a topic pattern
const sub_id = try messaging.subscribe("events.user.*", onUserEvent, null);

// Publish a message (blocks until all subscribers notified)
try messaging.publish(allocator, "events.user.created", "{\"id\": 42}");

// Unsubscribe
_ = try messaging.unsubscribe(sub_id);

// List active topics
const topics = try messaging.listTopics(allocator);
defer {
    for (topics) |t| allocator.free(t);
    allocator.free(topics);
}

// Inspect topic stats
const info = try messaging.topicStats("events.user.created");
std.log.info("subscribers={d} published={d}", .{
    info.subscriber_count, info.messages_published,
});

// Inspect dead letter queue
const dead = try messaging.getDeadLetters(allocator);
defer allocator.free(dead);
for (dead) |dl| {
    std.log.warn("DLQ: topic={s} reason={s}", .{ dl.message.topic, dl.reason });
}

// Clear dead letter queue
messaging.clearDeadLetters();

// Global stats
const s = messaging.messagingStats();
std.log.info("published={d} delivered={d} failed={d} dlq={d}", .{
    s.total_published, s.total_delivered, s.total_failed, s.dead_letter_count,
});
```

### Types

| Type | Description |
|------|-------------|
| `MessagingConfig` | Configuration: `max_topics`, `max_subscribers_per_topic` |
| `Message` | Delivered message: topic, payload, timestamp, id |
| `DeliveryResult` | Callback return: `.ok`, `.retry`, `.discard` |
| `SubscriberCallback` | Function pointer: `fn (Message, ?*anyopaque) DeliveryResult` |
| `TopicInfo` | Per-topic stats: subscriber_count, messages_published/delivered/failed |
| `DeadLetter` | Failed message with reason and timestamp |
| `Channel` | Topic descriptor: name, subscriber_count |
| `MessagingStats` | Aggregate stats: total_published, total_delivered, total_failed, dead_letter_count |
| `MessagingError` | Error set: `ChannelFull`, `TopicNotFound`, `SubscriberNotFound`, etc. |

---

## Pages (`abi.pages`)

Dashboard and UI page management with radix-tree URL routing, template variable
substitution, and support for static and dynamic content.

**Build flag:** `-Denable-pages=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Radix tree router** | O(path_segments) page matching with params (`{id}`) and wildcards (`*`) |
| **Templates** | Single-pass `{{variable}}` substitution with default values |
| **Content types** | Static (literal HTML/text) and template (rendered with variables) |
| **Per-page flags** | Auth requirement, cache TTL, layout selection |
| **Metadata** | Up to 4 custom key-value pairs per page |
| **Concurrency** | `RwLock` for concurrent page lookups |

### Content Types

Pages support two content modes:

| Mode | Description |
|------|-------------|
| **Static** | Literal content returned as-is. Best for fixed HTML, JSON, or text. |
| **Template** | Content with `{{variable}}` placeholders replaced at render time. |

### API

```zig
const abi = @import("abi");
const pages = abi.pages;

// Initialize
try pages.init(allocator, .{
    .max_pages = 512,
    .default_layout = "main",
    .default_cache_ttl_ms = 60_000,
});
defer pages.deinit();

// Add a static page
try pages.addPage(.{
    .path = "/dashboard",
    .title = "Dashboard",
    .content = .{ .static = "<h1>Welcome</h1><p>System status: OK</p>" },
    .layout = "main",
});

// Add a template page with variables
try pages.addPage(.{
    .path = "/users/{id}",
    .title = "User Profile",
    .content = .{
        .template = .{
            .source = "<h1>{{name}}</h1><p>Role: {{role}}</p>",
            .default_vars = .{
                .{ .key = "name", .value = "Unknown" },
                .{ .key = "role", .value = "viewer" },
            } ++ .{.{}} ** 6,
            .var_count = 2,
        },
    },
    .require_auth = true,
});

// Match a URL to a page
if (try pages.matchPage("/users/42")) |match| {
    if (match.getParam("id")) |user_id| {
        std.log.info("matched user page: id={s}", .{user_id});
    }
}

// Render a page with template variables
var vars = [_]pages.TemplateVar{
    .{ .key = "name", .value = "Alice" },
    .{ .key = "role", .value = "admin" },
} ++ .{.{}} ** 6;
var result = try pages.renderPage(allocator, "/users/42", &vars, 2);
defer result.deinit(allocator);
// result.body contains: "<h1>Alice</h1><p>Role: admin</p>"

// Look up a page without rendering
if (pages.getPage("/dashboard")) |page| {
    std.log.info("page: {s} (auth={any})", .{ page.title, page.require_auth });
}

// List all registered pages
const all = pages.listPages();
for (all) |page| {
    std.log.info("  {s} -> {s}", .{ page.path, page.title });
}

// Remove a page
_ = try pages.removePage("/dashboard");

// Stats
const s = pages.stats();
std.log.info("pages={d} renders={d} static={d} template={d}", .{
    s.total_pages, s.total_renders, s.static_pages, s.template_pages,
});
```

### Types

| Type | Description |
|------|-------------|
| `PagesConfig` | Configuration: `max_pages`, `default_layout`, `template_cache`, `default_cache_ttl_ms` |
| `Page` | Page definition: path, title, content, layout, method, require_auth, cache_ttl_ms, metadata |
| `PageContent` | Union: `.static` (literal) or `.template` (with variables) |
| `TemplateRef` | Template source with up to 8 default variable bindings |
| `TemplateVar` | Key-value pair for template substitution |
| `PageMatch` | Matched page with extracted path parameters |
| `RenderResult` | Rendered output: title, body, layout (call `deinit()` to free) |
| `PagesStats` | Aggregate: total_pages, total_renders, static_pages, template_pages |
| `PagesError` | Error set: `PageNotFound`, `DuplicatePage`, `TooManyPages`, `TemplateError`, etc. |

---

## Disabling at Build Time

```bash
# Disable messaging
zig build -Denable-messaging=false

# Disable pages
zig build -Denable-pages=false
```

When disabled, all public functions return `error.FeatureDisabled` with zero
binary overhead.
