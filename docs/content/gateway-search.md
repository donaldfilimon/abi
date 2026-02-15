---
title: Gateway & Search
description: API gateway with rate limiting and full-text BM25 search
section: Modules
order: 11
---

# Gateway & Search

ABI includes an **API Gateway** for routing, rate limiting, and circuit breaking,
and a **Search** engine with inverted indexes and BM25 relevance scoring.

---

## Gateway (`abi.gateway`)

A full API gateway with radix-tree route matching, three rate limiting algorithms,
circuit breaker state machine, middleware chain, and latency tracking.

**Build flag:** `-Denable-gateway=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Radix tree router** | O(path_segments) matching with path params (`{id}`) and wildcards (`*`) |
| **Rate limiting** | Token bucket, sliding window histogram, fixed window |
| **Circuit breaker** | State machine: closed -> open -> half_open -> closed |
| **Latency histogram** | 7 buckets for p50/p99 estimation |
| **Concurrency** | `RwLock` for concurrent route lookups |

### Rate Limiting Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| Token bucket | Tokens replenish at a fixed rate; requests consume tokens | Bursty traffic with steady average |
| Sliding window | Histogram-based window that smoothly slides over time | Accurate per-second limits |
| Fixed window | Simple counter reset at fixed intervals | Low overhead, coarse limits |

### Circuit Breaker States

```
closed ──(failures exceed threshold)──> open
  ^                                       |
  |                              (timeout expires)
  |                                       v
  └──(probe succeeds)──── half_open ──(probe fails)──> open
```

The circuit breaker tracks per-upstream success/failure counts. When failures
exceed the configured threshold, the circuit opens and rejects requests for a
cooldown period. After the timeout, a single probe request is allowed through
(half-open state).

### API

```zig
const abi = @import("abi");
const gateway = abi.gateway;

// Initialize
try gateway.init(allocator, .{
    .max_routes = 256,
});
defer gateway.deinit();

// Add routes with optional rate limiting
try gateway.addRoute(.{
    .path = "/api/users/{id}",
    .method = .GET,
    .upstream = "http://user-service:8080",
    .timeout_ms = 5_000,
    .rate_limit = .{
        .algorithm = .token_bucket,
        .requests_per_second = 100,
        .burst_size = 20,
    },
});

try gateway.addRoute(.{
    .path = "/api/files/*",
    .method = .GET,
    .upstream = "http://file-service:8081",
});

// Match a request to a route
if (try gateway.matchRoute("/api/users/42", .GET)) |match| {
    // Extract path parameters
    if (match.getParam("id")) |user_id| {
        std.log.info("user_id = {s}", .{user_id});
    }
}

// Check rate limit before forwarding
const limit = gateway.checkRateLimit("/api/users/{id}");
if (!limit.allowed) {
    std.log.warn("rate limited, retry in {d}ms", .{limit.reset_after_ms});
}

// Record upstream result for circuit breaker
gateway.recordUpstreamResult("http://user-service:8080", true);

// Query circuit breaker state
const cb_state = gateway.getCircuitState("http://user-service:8080");
// cb_state is one of: .closed, .open, .half_open

// Remove a route
_ = try gateway.removeRoute("/api/files/*");

// Gateway statistics
const s = gateway.stats();
std.log.info("requests={d} rate_limited={d} circuit_trips={d}", .{
    s.total_requests, s.rate_limited_count, s.circuit_breaker_trips,
});
```

### Types

| Type | Description |
|------|-------------|
| `GatewayConfig` | Configuration: `max_routes`, default timeouts |
| `Route` | Route definition: path, method, upstream, timeout, rate_limit, middlewares |
| `HttpMethod` | Enum: `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, `HEAD`, `OPTIONS` |
| `MatchResult` | Matched route with extracted path parameters |
| `RateLimitConfig` | Algorithm selection and parameters (requests/second, burst) |
| `RateLimitAlgorithm` | Enum: `.token_bucket`, `.sliding_window`, `.fixed_window` |
| `RateLimitResult` | Result: `allowed`, `remaining`, `reset_after_ms` |
| `CircuitBreakerState` | Enum: `.closed`, `.open`, `.half_open` |
| `GatewayStats` | Aggregate stats: total_requests, active_routes, rate_limited, circuit trips |
| `MiddlewareType` | Enum: `auth`, `rate_limit`, `circuit_breaker`, `access_log`, `cors`, etc. |
| `GatewayError` | Error set: `RouteNotFound`, `RateLimitExceeded`, `CircuitOpen`, `InvalidRoute`, etc. |

---

## Search (`abi.search`)

Full-text search with inverted indexes, BM25 relevance scoring, tokenization,
and snippet generation.

**Build flag:** `-Denable-search=true` (enabled by default)

### Architecture

| Component | Description |
|-----------|-------------|
| **Named indexes** | SwissMap of index name to inverted index |
| **Inverted index** | Term -> PostingList (doc_id, term_frequency, positions) |
| **BM25 scoring** | IDF x TF component with Lucene IDF variant (always non-negative) |
| **Tokenizer** | Lowercase normalization + stop word removal |
| **Snippets** | Window with highest match density around query terms |

### BM25 Parameters

The default BM25 parameters follow standard information retrieval conventions:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.2 | Term frequency saturation; higher values increase TF influence |
| `b` | 0.75 | Document length normalization; 0 = no normalization, 1 = full |

The IDF (inverse document frequency) component uses the Lucene variant which
guarantees non-negative scores even when a term appears in more than half the
documents.

### API

```zig
const abi = @import("abi");
const search = abi.search;

// Initialize
try search.init(allocator, .{
    .max_indexes = 64,
});
defer search.deinit();

// Create a named index
const idx = try search.createIndex(allocator, "articles");

// Index documents
try search.indexDocument("articles", "doc-1", "The quick brown fox jumps");
try search.indexDocument("articles", "doc-2", "A lazy dog sleeps in the sun");
try search.indexDocument("articles", "doc-3", "Quick foxes are brown and fast");

// Search with BM25 scoring (returns results sorted by relevance)
const results = try search.query(allocator, "articles", "quick brown fox");
defer allocator.free(results);

for (results) |hit| {
    std.log.info("doc={s} score={d:.3} snippet={s}", .{
        hit.doc_id, hit.score, hit.snippet,
    });
}

// Delete a document from the index
_ = try search.deleteDocument("articles", "doc-2");

// Delete an entire index
try search.deleteIndex("articles");

// Aggregate stats
const s = search.stats();
std.log.info("indexes={d} documents={d} terms={d}", .{
    s.total_indexes, s.total_documents, s.total_terms,
});
```

### Stop Words

The tokenizer removes common English stop words before indexing and querying.
The stop word list includes determiners, prepositions, pronouns, auxiliary verbs,
and conjunctions (approximately 50 words). This improves both index size and
query relevance by focusing on content-bearing terms.

### Types

| Type | Description |
|------|-------------|
| `SearchConfig` | Configuration: `max_indexes`, BM25 parameters |
| `SearchIndex` | Index metadata: name, doc_count, size_bytes |
| `SearchResult` | Hit: doc_id, BM25 score, context snippet |
| `SearchStats` | Aggregate: total_indexes, total_documents, total_terms |
| `SearchError` | Error set: `IndexNotFound`, `InvalidQuery`, `IndexAlreadyExists`, `DocumentNotFound`, etc. |

---

## Disabling at Build Time

```bash
# Disable gateway
zig build -Denable-gateway=false

# Disable search
zig build -Denable-search=false
```

When disabled, all public functions return `error.FeatureDisabled` with zero
binary overhead.
