---
title: "Web"
description: "HTTP utilities and web middleware"
section: "Infrastructure"
order: 5
---

# Web

The Web module provides an HTTP client with JSON support, persona-based chat
routing, weather API integration, and thread-safe global client management.

- **Build flag:** `-Denable-web=true` (default: enabled)
- **Namespace:** `abi.web`
- **Source:** `src/features/web/`

## Overview

The web module wraps Zig's standard library HTTP client with convenient
utilities for common web operations. It serves as the outbound HTTP layer of
the ABI framework, complementing the inbound [Gateway](gateway.html) module.

Key capabilities:

- **HTTP Client** -- Synchronous GET and POST requests with configurable timeouts, redirects, and response size limits
- **JSON Utilities** -- Parse JSON responses and check HTTP status codes
- **Persona Chat Handlers** -- Request/response types and route definitions for the AI persona API
- **Weather Client** -- Integration with the Open-Meteo weather API for coordinate-based forecasts
- **Thread Safety** -- Global client protected by a mutex; per-thread `Context` for isolated usage
- **Request Options** -- Configurable user agent, redirect limits, extra headers, and max response bytes

## Quick Start

```zig
const abi = @import("abi");

// Initialize via Framework
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withWebDefaults()
    .build();
defer framework.deinit();

// Make an HTTP GET request
const response = try abi.web.get(allocator, "https://api.example.com/data");
defer abi.web.freeResponse(allocator, response);

if (abi.web.isSuccessStatus(response.status)) {
    // Parse JSON response
    var parsed = try abi.web.parseJsonValue(allocator, response);
    defer parsed.deinit();
    // Use parsed.value...
}
```

### POST Request with JSON

```zig
const body = "{\"message\": \"hello\"}";
const response = try abi.web.postJson(allocator, "https://api.example.com/chat", body);
defer abi.web.freeResponse(allocator, response);
```

### Request Options

```zig
const response = try abi.web.getWithOptions(allocator, url, .{
    .max_response_bytes = 10 * 1024 * 1024,  // 10MB limit
    .user_agent = "my-app/1.0",
    .follow_redirects = true,
    .redirect_limit = 5,
    .extra_headers = &.{
        .{ .name = "Authorization", .value = "Bearer token" },
    },
});
```

### Using the Context API

For Framework integration or per-thread isolation, use the `Context` struct:

```zig
var ctx = try abi.web.Context.init(allocator, .{});
defer ctx.deinit();

const response = try ctx.get("https://api.example.com/data");
defer ctx.freeResponse(response);

var json = try ctx.parseJsonValue(response);
defer json.deinit();
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context wrapping an HTTP client |
| `HttpClient` | Low-level HTTP client with GET/POST methods |
| `RequestOptions` | Timeout, user agent, redirect limit, extra headers, max response bytes |
| `Response` | HTTP response: status code and body bytes |
| `JsonValue` | Alias for `std.json.Value` |
| `ParsedJson` | Alias for `std.json.Parsed(JsonValue)` |
| `WeatherClient` | Open-Meteo weather API client |
| `WeatherConfig` | Weather API configuration |
| `WebError` | Error set: `WebDisabled` |

### Chat Handlers

| Type | Description |
|------|-------------|
| `ChatHandler` | Handler for persona-based chat requests |
| `ChatRequest` | Chat request: content, user ID, session ID, persona, context, max tokens, temperature |
| `ChatResponse` | Chat response: content, persona, confidence, latency, code blocks, references |
| `ChatResult` | Raw result: status code and body |

### Persona Routing

| Type | Description |
|------|-------------|
| `PersonaRouter` | Maps URL paths to AI persona handlers |
| `Route` | A route: path, method, description, auth requirement |
| `RouteContext` | Request context for route handling |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator) !void` | Initialize the global HTTP client (thread-safe) |
| `deinit() void` | Tear down the global client |
| `isEnabled() bool` | Returns `true` if web is compiled in |
| `isInitialized() bool` | Returns `true` if the global client is active |
| `get(allocator, url) !Response` | HTTP GET request |
| `getWithOptions(allocator, url, options) !Response` | HTTP GET with custom options |
| `postJson(allocator, url, body) !Response` | HTTP POST with JSON body |
| `freeResponse(allocator, response) void` | Free a response body |
| `parseJsonValue(allocator, response) !ParsedJson` | Parse response body as JSON |
| `isSuccessStatus(status) bool` | Check if an HTTP status is 2xx |

### Error Handling

The module uses `WebError` and standard HTTP errors:

| Error | Description |
|-------|-------------|
| `WebDisabled` | Web feature is disabled at build time |
| `InvalidUrl` | URL parsing failed |
| `RequestFailed` | HTTP request failed |
| `ConnectionFailed` | Network connection failed |
| `ResponseTooLarge` | Response exceeds `max_response_bytes` |
| `Timeout` | Request timed out |
| `ReadFailed` | Error reading response body |

## Configuration

Web is configured through the `WebConfig` struct:

```zig
const config = abi.config.WebConfig{
    .bind_address = "127.0.0.1",
    .port = 3000,
    .cors_enabled = true,
    .timeout_ms = 30000,
    .max_body_size = 10 * 1024 * 1024,  // 10MB
    .rate_limit = .{},                   // disabled by default
};
```

For production deployments with rate limiting enabled:

```zig
const config = abi.config.WebConfig.productionDefaults();
// bind_address = "0.0.0.0", rate limiting enabled (100 req/min, 20 burst)
```

| Field | Default | Description |
|-------|---------|-------------|
| `bind_address` | `"127.0.0.1"` | HTTP server bind address |
| `port` | 3000 | HTTP server port |
| `cors_enabled` | `true` | Enable CORS headers |
| `timeout_ms` | 30000 | Request timeout in milliseconds |
| `max_body_size` | 10MB | Maximum request body size |
| `rate_limit` | disabled | Rate limiting configuration |

## CLI Commands

The web module does not have a dedicated CLI command. Use the web API
programmatically or through the Framework builder.

## Examples

See `examples/web.zig` for a complete working example demonstrating chat
handler types, persona routing, and HTTP client configuration:

```bash
zig build run-web
```

## Disabling at Build Time

```bash
# Compile without web support
zig build -Denable-web=false
```

When disabled, all public functions return `error.WebDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures -- including full stub implementations of `ChatHandler`,
`PersonaRouter`, and `RouteContext` -- so downstream code compiles without
conditional guards.

## Related

- [Gateway](gateway.html) -- API gateway for inbound request routing
- [Pages](pages.html) -- Dashboard UI pages
- [Cloud](cloud.html) -- Serverless cloud function adapters
- [Network](network.html) -- Distributed compute and node management

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
