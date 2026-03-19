---
title: web API
purpose: Generated API reference for web
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# web

> Web Module - HTTP Client and Web Utilities

This module provides HTTP client functionality, weather API integration,
and profile API handlers for the ABI framework. It wraps Zig's standard
library HTTP client with convenient utilities for common web operations.

## Features

- **HTTP Client**: Synchronous HTTP client with configurable options
- GET and POST requests with JSON support
- Configurable timeouts, redirects, and response size limits
- Thread-safe global client with mutex protection

- **Weather Client**: Integration with Open-Meteo weather API
- Coordinate-based weather forecasts
- Location validation and URL building

- **Profile API**: HTTP handlers and routes for AI profile system
- Chat request/response handlers
- REST API routes with OpenAPI documentation
- Health check and metrics endpoints

## Usage Example

```zig
const web = @import("abi").web;

// Initialize the web module
try web.init(allocator);
defer web.deinit();

// Make an HTTP GET request
const response = try web.get(allocator, "https://api.example.com/data");
defer web.freeResponse(allocator, response);

if (web.isSuccessStatus(response.status)) {
// Parse JSON response
var parsed = try web.parseJsonValue(allocator, response);
defer parsed.deinit();
// Use parsed.value...
}
```

## Using the Context API

For Framework integration, use the Context struct:

```zig
const cfg = config_module.WebConfig{};
var ctx = try web.Context.init(allocator, cfg);
defer ctx.deinit();

const response = try ctx.get("https://api.example.com/data");
defer ctx.freeResponse(response);
```

## POST Request with JSON

```zig
const body = "{\"message\": \"hello\"}";
const response = try web.postJson(allocator, "https://api.example.com/chat", body);
defer web.freeResponse(allocator, response);
```

## Request Options

```zig
const response = try web.getWithOptions(allocator, url, .{
.max_response_bytes = 10 * 1024 * 1024,  // 10MB limit
.user_agent = "my-app/1.0",
.follow_redirects = true,
.redirect_limit = 5,
.extra_headers = &.{
.{ .name = "Authorization", .value = "Bearer token" },
},
});
```

## Error Handling

The module uses `HttpError` for HTTP-specific errors:
- `InvalidUrl`: URL parsing failed
- `InvalidRequest`: Request configuration is invalid
- `RequestFailed`: HTTP request failed
- `ConnectionFailed`: Network connection failed
- `ResponseTooLarge`: Response exceeds max_response_bytes
- `Timeout`: Request timed out
- `ReadFailed`: Error reading response body

## Feature Flag

This module is controlled by `-Dfeat-web=true` (default: enabled).
When disabled, all operations return `error.WebDisabled`.

## Thread Safety

The global `init()`/`deinit()` functions use mutex protection for
thread-safe access to the default client. The `Context` struct should
be used per-thread or with external synchronization.

**Source:** [`src/features/web/mod.zig`](../../src/features/web/mod.zig)

**Build flag:** `-Dfeat_web=true`

---

## API

### <a id="pub-const-weberror"></a>`pub const WebError`

<sup>**const**</sup> | [source](../../src/features/web/mod.zig#L145)

Errors specific to the web module.

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/features/web/mod.zig#L165)

Web Context for Framework integration.

Wraps the HTTP client functionality to provide a consistent interface
with other ABI modules. This is the preferred API for Framework users
as it integrates with the unified configuration system.

## Example

```zig
var ctx = try web.Context.init(allocator, config);
defer ctx.deinit();

const response = try ctx.get("https://api.example.com/data");
defer ctx.freeResponse(response);

var json = try ctx.parseJsonValue(response);
defer json.deinit();
```

### <a id="pub-fn-get-self-context-url-const-u8-response"></a>`pub fn get(self: *Context, url: []const u8) !Response`

<sup>**fn**</sup> | [source](../../src/features/web/mod.zig#L193)

Perform an HTTP GET request.

### <a id="pub-fn-getwithoptions-self-context-url-const-u8-options-requestoptions-response"></a>`pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response`

<sup>**fn**</sup> | [source](../../src/features/web/mod.zig#L201)

Perform an HTTP GET request with options.

### <a id="pub-fn-postjson-self-context-url-const-u8-body-const-u8-response"></a>`pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response`

<sup>**fn**</sup> | [source](../../src/features/web/mod.zig#L209)

Perform an HTTP POST request with JSON body.

### <a id="pub-fn-freeresponse-self-context-response-response-void"></a>`pub fn freeResponse(self: *Context, response: Response) void`

<sup>**fn**</sup> | [source](../../src/features/web/mod.zig#L217)

Free a response body.

### <a id="pub-fn-parsejsonvalue-self-context-response-response-parsedjson"></a>`pub fn parseJsonValue(self: *Context, response: Response) !ParsedJson`

<sup>**fn**</sup> | [source](../../src/features/web/mod.zig#L222)

Parse a JSON response.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
