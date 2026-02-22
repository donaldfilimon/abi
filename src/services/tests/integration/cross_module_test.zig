//! Cross-Module Integration Tests
//!
//! Tests that exercise interactions between feature modules:
//! - Cache + Storage: cache-aside pattern, invalidation
//! - Search: index documents, query, update (document-store integration)
//! - Messaging + Gateway: pub/sub with route dispatch concepts
//!
//! Each test initializes the real module singletons, exercises their APIs
//! in combination, and tears them down. Feature-disabled builds skip
//! gracefully via `error.SkipZigTest`.

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");

// Access modules through abi (real or stub depending on build flags)
const cache = abi.cache;
const storage = abi.storage;
const search = abi.search;
const messaging = abi.messaging;
const gateway = abi.gateway;

// ============================================================================
// Cache + Storage Integration
// ============================================================================

test "cross-module: cache-aside pattern with storage backend" {
    if (!build_options.enable_cache or !build_options.enable_storage)
        return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Init both modules
    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();
    try cache.init(allocator, .{ .max_entries = 64, .eviction_policy = .lru });
    defer cache.deinit();

    // 1. Store data in storage
    try storage.putObject(allocator, "doc/readme.txt", "Hello from storage");

    // 2. Cache miss — fetch from storage, populate cache
    const cached = try cache.get("doc/readme.txt");
    try std.testing.expect(cached == null); // not cached yet

    const from_storage = try storage.getObject(allocator, "doc/readme.txt");
    defer allocator.free(from_storage);
    try std.testing.expectEqualStrings("Hello from storage", from_storage);

    // Populate cache with storage result
    try cache.put("doc/readme.txt", from_storage);

    // 3. Cache hit — avoid storage round-trip
    const hit = try cache.get("doc/readme.txt");
    try std.testing.expect(hit != null);
    try std.testing.expectEqualStrings("Hello from storage", hit.?);

    // 4. Verify stats reflect the hit/miss pattern
    const c_stats = cache.stats();
    try std.testing.expect(c_stats.hits >= 1);
    try std.testing.expect(c_stats.misses >= 1);
    try std.testing.expectEqual(@as(u32, 1), c_stats.entries);
}

test "cross-module: cache invalidation clears stale storage data" {
    if (!build_options.enable_cache or !build_options.enable_storage)
        return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try storage.init(allocator, .{ .backend = .memory });
    defer storage.deinit();
    try cache.init(allocator, .{ .max_entries = 64, .eviction_policy = .lru });
    defer cache.deinit();

    // Store and cache initial value
    try storage.putObject(allocator, "config.json", "{\"v\":1}");
    try cache.put("config.json", "{\"v\":1}");

    // Update storage (simulating external write)
    try storage.putObject(allocator, "config.json", "{\"v\":2}");

    // Cache still has stale data
    const stale = try cache.get("config.json");
    try std.testing.expect(stale != null);
    try std.testing.expectEqualStrings("{\"v\":1}", stale.?);

    // Invalidate cache entry
    const deleted = try cache.delete("config.json");
    try std.testing.expect(deleted);

    // Re-fetch from storage gets fresh data
    const fresh = try storage.getObject(allocator, "config.json");
    defer allocator.free(fresh);
    try std.testing.expectEqualStrings("{\"v\":2}", fresh);

    // Re-populate cache
    try cache.put("config.json", fresh);
    const refreshed = try cache.get("config.json");
    try std.testing.expect(refreshed != null);
    try std.testing.expectEqualStrings("{\"v\":2}", refreshed.?);
}

// ============================================================================
// Search + Document Integration
// ============================================================================

test "cross-module: index and search documents via search engine" {
    if (!build_options.enable_search) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try search.init(allocator, .{});
    defer search.deinit();

    // Create an index (acts as a document store + search engine)
    _ = try search.createIndex(allocator, "articles");

    // Index documents
    try search.indexDocument("articles", "doc1", "Zig is a systems programming language");
    try search.indexDocument("articles", "doc2", "Rust focuses on memory safety");
    try search.indexDocument("articles", "doc3", "Zig comptime enables powerful metaprogramming");

    // Query — should find Zig-related documents
    const results = try search.query(allocator, "articles", "Zig programming");
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    // Top result should be about Zig
    const top = results[0];
    try std.testing.expect(top.score > 0);
}

test "cross-module: update document re-indexes in search" {
    if (!build_options.enable_search) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try search.init(allocator, .{});
    defer search.deinit();

    _ = try search.createIndex(allocator, "docs");

    // Index original
    try search.indexDocument("docs", "page1", "introduction to databases");

    // Query for original content
    const before = try search.query(allocator, "docs", "databases");
    defer allocator.free(before);
    try std.testing.expect(before.len == 1);

    // Update the same doc_id with new content (re-index)
    try search.indexDocument("docs", "page1", "guide to distributed caching");

    // Old query should no longer match
    const after_old = try search.query(allocator, "docs", "databases");
    defer allocator.free(after_old);
    try std.testing.expect(after_old.len == 0);

    // New query should match
    const after_new = try search.query(allocator, "docs", "caching");
    defer allocator.free(after_new);
    try std.testing.expect(after_new.len == 1);
}

// ============================================================================
// Messaging + Gateway Integration
// ============================================================================

var msg_received_count: u32 = 0;
var msg_last_topic: []const u8 = "";
var msg_last_payload: []const u8 = "";

fn integrationCallback(msg: messaging.Message, _: ?*anyopaque) messaging.DeliveryResult {
    msg_received_count += 1;
    msg_last_topic = msg.topic;
    msg_last_payload = msg.payload;
    return .ok;
}

test "cross-module: messaging pub/sub with gateway route awareness" {
    if (!build_options.enable_messaging or !build_options.enable_gateway)
        return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();
    try gateway.init(allocator, gateway.GatewayConfig.defaults());
    defer gateway.deinit();

    // Register a gateway route
    try gateway.addRoute(.{
        .path = "/api/events",
        .method = .POST,
        .upstream = "http://events-service:8080",
    });

    // Subscribe to events topic
    msg_received_count = 0;
    const sub_id = try messaging.subscribe(
        allocator,
        "api.events.#",
        integrationCallback,
        null,
    );

    // Simulate: gateway matches a route, then publishes a message
    const match = try gateway.matchRoute("/api/events", .POST);
    try std.testing.expect(match != null);

    // Dispatch message based on matched route
    try messaging.publish(allocator, "api.events.created", "new-event-payload");

    // Verify subscriber received the message
    try std.testing.expectEqual(@as(u32, 1), msg_received_count);
    try std.testing.expectEqualStrings("api.events.created", msg_last_topic);
    try std.testing.expectEqualStrings("new-event-payload", msg_last_payload);

    // Verify stats from both modules
    const gw_stats = gateway.stats();
    try std.testing.expect(gw_stats.total_requests >= 1);
    const msg_stats = messaging.messagingStats();
    try std.testing.expect(msg_stats.total_published >= 1);
    try std.testing.expect(msg_stats.total_delivered >= 1);

    // Cleanup subscriber
    _ = try messaging.unsubscribe(sub_id);
}

test "cross-module: gateway circuit breaker + messaging notification" {
    if (!build_options.enable_messaging or !build_options.enable_gateway)
        return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try gateway.init(allocator, .{
        .circuit_breaker = .{
            .failure_threshold = 2,
            .reset_timeout_ms = 5000,
            .half_open_max_requests = 1,
        },
    });
    defer gateway.deinit();
    try messaging.init(allocator, messaging.MessagingConfig.defaults());
    defer messaging.deinit();

    // Trip the circuit breaker on an upstream
    gateway.recordUpstreamResult("events-backend", false);
    gateway.recordUpstreamResult("events-backend", false);
    const cb_state = gateway.getCircuitState("events-backend");
    try std.testing.expectEqual(gateway.CircuitBreakerState.open, cb_state);

    // Publish a notification about the circuit breaker trip
    try messaging.publish(allocator, "system.circuit.open", "events-backend");

    const mstats = messaging.messagingStats();
    try std.testing.expect(mstats.total_published >= 1);

    // Reset and verify
    gateway.resetCircuit("events-backend");
    try std.testing.expectEqual(
        gateway.CircuitBreakerState.closed,
        gateway.getCircuitState("events-backend"),
    );
}

test {
    std.testing.refAllDecls(@This());
}
