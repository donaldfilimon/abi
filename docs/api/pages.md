---
title: pages API
purpose: Generated API reference for pages
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# pages

> Pages Module

Dashboard/UI pages with radix-tree URL routing, template rendering,
and support for both static and dynamic content.

Architecture:
- Radix tree for O(path_segments) page matching with params and wildcards
- Single-pass {{variable}} template substitution
- Per-page flags for auth requirement and cache TTL
- RwLock for concurrent page lookups

**Source:** [`src/features/observability/pages/mod.zig`](../../src/features/observability/pages/mod.zig)

**Build flag:** `-Dfeat_pages=true`

---

## API

### <a id="pub-fn-init-allocator-std-mem-allocator-config-pagesconfig-pageserror-void"></a>`pub fn init(allocator: std.mem.Allocator, config: PagesConfig) PagesError!void`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L245)

Initialize the pages module singleton for URL-routed dashboard/UI pages.

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L251)

Tear down the pages module, freeing all registered pages.

### <a id="pub-fn-addpage-page-page-pageserror-void"></a>`pub fn addPage(page: Page) PagesError!void`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L268)

Register a page at a URL path. Supports path parameters (`{id}`)
and wildcard segments (`*`).

### <a id="pub-fn-removepage-path-const-u8-pageserror-bool"></a>`pub fn removePage(path: []const u8) PagesError!bool`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L321)

Remove a page by its registered path. Returns `true` if found.

### <a id="pub-fn-getpage-path-const-u8-page"></a>`pub fn getPage(path: []const u8) ?Page`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L348)

Look up a page by exact path (no pattern matching).

### <a id="pub-fn-matchpage-path-const-u8-pageserror-pagematch"></a>`pub fn matchPage(path: []const u8) PagesError!?PageMatch`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L363)

Match an incoming URL path against the radix tree.
Returns the page and extracted path parameters, or `null`.

### <a id="pub-fn-renderpage-allocator-std-mem-allocator-path-const-u8-vars-const-templatevar-pageserror-renderresult"></a>`pub fn renderPage( allocator: std.mem.Allocator, path: []const u8, vars: []const TemplateVar, ) PagesError!RenderResult`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L386)

Render a page's template, substituting `{{variable}}` placeholders with
the provided parameters and path captures.

### <a id="pub-fn-listpages-const-page"></a>`pub fn listPages() []const Page`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L443)

Return all registered pages (slice into internal storage).

### <a id="pub-fn-stats-pagesstats"></a>`pub fn stats() PagesStats`

<sup>**fn**</sup> | [source](../../src/features/observability/pages/mod.zig#L451)

Snapshot page count, render count, and cache hit ratio.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
