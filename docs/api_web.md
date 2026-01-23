# web API Reference

> Web utilities and HTTP support

**Source:** [`src/web/mod.zig`](../../src/web/mod.zig)

---

Web feature helpers for HTTP and weather client access.

This module provides:
- HTTP client for making requests
- Weather API client
- Persona API handlers and routes

---

## API

### `pub const Context`

<sup>**type**</sup>

Web Context for Framework integration.
Wraps the HTTP client functionality to provide a consistent interface with other modules.

### `pub fn get(self: *Context, url: []const u8) !Response`

<sup>**fn**</sup>

Perform an HTTP GET request.

### `pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response`

<sup>**fn**</sup>

Perform an HTTP GET request with options.

### `pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response`

<sup>**fn**</sup>

Perform an HTTP POST request with JSON body.

### `pub fn freeResponse(self: *Context, response: Response) void`

<sup>**fn**</sup>

Free a response body.

### `pub fn parseJsonValue(self: *Context, response: Response) !ParsedJson`

<sup>**fn**</sup>

Parse a JSON response.

---

*Generated automatically by `zig build gendocs`*
