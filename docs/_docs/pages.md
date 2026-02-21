---
title: "Pages"
description: "Dashboard UI pages with URL path routing"
section: "Infrastructure"
order: 4
---

# Pages

The Pages module provides dashboard and UI page management with radix-tree URL
routing, `{{variable}}` template rendering, path parameter extraction, and
per-page auth and cache settings.

- **Build flag:** `-Denable-pages=true` (default: enabled)
- **Namespace:** `abi.pages`
- **Source:** `src/features/pages/`

## Overview

The pages module lets you register URL-routed pages that serve static content
or render templates with variable substitution. It shares the same radix tree
implementation as the [Gateway](gateway.html) module (via
`services/shared/utils/radix_tree.zig`), giving O(path-segments) route matching
with support for path parameters and wildcards.

Key capabilities:

- **Radix-tree URL routing** -- O(path-segments) page matching with `{param}` and `*` wildcards
- **Static and template content** -- Pages can serve literal HTML or templates with `{{variable}}` substitution
- **Template rendering** -- Single-pass `{{key}}` placeholder replacement with default and override variables
- **Per-page settings** -- Auth requirement flag, cache TTL, layout selection, metadata
- **Concurrent access** -- RwLock protects page lookups; atomic counter for render stats
- **Framework integration** -- Context struct for lifecycle management

## Quick Start

```zig
const abi = @import("abi");

// Initialize via Framework
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withPagesDefaults()
    .build();
defer framework.deinit();

// Register a static page
try abi.pages.addPage(.{
    .path = "/",
    .title = "Home",
    .content = .{ .static = "<h1>Welcome</h1><p>Dashboard home</p>" },
});

// Register a template page with path parameter
try abi.pages.addPage(.{
    .path = "/users/{id}",
    .title = "User Profile",
    .content = .{
        .template = .{
            .source = "<h1>{{username}}</h1><p>Role: {{role}}</p>",
            .default_vars = blk: {
                var vars: [8]abi.pages.TemplateVar = [_]abi.pages.TemplateVar{.{}} ** 8;
                vars[0] = .{ .key = "username", .value = "guest" };
                vars[1] = .{ .key = "role", .value = "viewer" };
                break :blk vars;
            },
            .var_count = 2,
        },
    },
    .require_auth = true,
});

// Match a URL and extract parameters
if (try abi.pages.matchPage("/users/42")) |match| {
    std.debug.print("Page: {s}\n", .{match.page.title});
    if (match.getParam("id")) |id| {
        std.debug.print("User ID: {s}\n", .{id});
    }
}

// Render a page with variable overrides
var result = try abi.pages.renderPage(allocator, "/", &.{});
defer result.deinit(allocator);
std.debug.print("{s}\n", .{result.body});
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context |
| `PagesConfig` | Max pages, default layout, template cache settings, default cache TTL |
| `Page` | A registered page: path, title, content, layout, method, auth, cache TTL, metadata |
| `PageContent` | Union of `.static` (literal string) or `.template` (template ref with vars) |
| `TemplateRef` | Template source string with up to 8 default variables |
| `TemplateVar` | Key-value pair for template substitution |
| `HttpMethod` | GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS |
| `MetadataEntry` | Key-value metadata attached to a page |
| `PageMatch` | Matched page with extracted path parameters |
| `RenderResult` | Rendered page: title, body (possibly owned), layout |
| `PagesStats` | Total pages, total renders, static vs. template counts |
| `PagesError` | Error set: FeatureDisabled, PageNotFound, DuplicatePage, TooManyPages, etc. |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator, config) !void` | Initialize the pages singleton |
| `deinit() void` | Tear down all pages and free state |
| `isEnabled() bool` | Returns `true` if pages is compiled in |
| `isInitialized() bool` | Returns `true` if the singleton is active |
| `addPage(page) !void` | Register a page at a URL path |
| `removePage(path) !bool` | Remove a page by path; rebuilds radix tree |
| `getPage(path) ?Page` | Look up a page by exact path (no pattern matching) |
| `matchPage(path) !?PageMatch` | Match a URL against the radix tree with parameter extraction |
| `renderPage(allocator, path, vars) !RenderResult` | Render a page's template with variable substitution |
| `listPages() []const Page` | Return registered pages |
| `stats() PagesStats` | Snapshot of page and render counts |

### Template Rendering

Templates use `{{key}}` placeholders that are replaced during rendering:

```
<h1>Hello, {{username}}!</h1>
<p>Your role is {{role}}.</p>
```

Variables are resolved in this order:

1. Override variables passed to `renderPage()`
2. Default variables defined in the `TemplateRef`

If a key is not found in either source, an empty string is substituted.

### Path Parameters

Pages registered with `{param}` segments extract values during matching:

| Registered Path | Request URL | Extracted Parameters |
|-----------------|-------------|---------------------|
| `/users/{id}` | `/users/42` | `id = "42"` |
| `/posts/{slug}/comments` | `/posts/hello-world/comments` | `slug = "hello-world"` |
| `/files/*` | `/files/readme.txt` | wildcard match |

Access parameters through the `PageMatch` struct:

```zig
if (match.getParam("id")) |id| {
    // use id
}
```

## Configuration

Pages is configured through the `PagesConfig` struct:

```zig
const config = abi.pages.PagesConfig{
    .max_pages = 256,
    .default_layout = "default",
    .enable_template_cache = true,
    .template_cache_size = 64,
    .default_cache_ttl_ms = 0,    // 0 = no caching
};
```

| Field | Default | Description |
|-------|---------|-------------|
| `max_pages` | 256 | Maximum number of registered pages |
| `default_layout` | `"default"` | Default layout name for pages |
| `enable_template_cache` | `true` | Cache compiled templates |
| `template_cache_size` | 64 | Number of cached templates |
| `default_cache_ttl_ms` | 0 | Default cache TTL (0 = disabled) |

## CLI Commands

The pages module does not have a dedicated CLI command. Use the pages API
programmatically or through the Framework builder.

## Examples

See `examples/pages.zig` for a complete working example that registers static
and template pages, matches URLs, extracts path parameters, and renders
content:

```bash
zig build run-pages
```

## Disabling at Build Time

```bash
# Compile without pages support
zig build -Denable-pages=false
```

When disabled, all public functions return `error.FeatureDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures so downstream code compiles without conditional guards.

## Related

- [Gateway](gateway.html) -- API gateway sharing the same radix tree router
- [Web](web.html) -- HTTP client utilities and persona routing
- [Cloud](cloud.html) -- Serverless cloud function adapters

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
