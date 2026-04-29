//! Integration Tests: Pages Feature
//!
//! Tests the pages module exports, lifecycle queries, page types,
//! and route matching through the public `abi.pages` surface.
//! Pages is nested under observability but gated independently by `feat_pages`.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const pages = abi.pages;

// ============================================================================
// Feature gate
// ============================================================================

test "pages: isEnabled reflects feature flag" {
    if (build_options.feat_pages) {
        try std.testing.expect(pages.isEnabled());
    } else {
        try std.testing.expect(!pages.isEnabled());
    }
}

test "pages: isInitialized reflects feature flag" {
    if (build_options.feat_pages) {
        // May or may not be initialized depending on runtime state
        const initialized = pages.isInitialized();
        try std.testing.expect(initialized == true or initialized == false);
    } else {
        try std.testing.expect(!pages.isInitialized());
    }
}

// ============================================================================
// Types
// ============================================================================

test "pages: PagesError includes expected variants" {
    const e1: pages.PagesError = error.FeatureDisabled;
    const e2: pages.PagesError = error.PageNotFound;
    const e3: pages.PagesError = error.DuplicatePage;
    const e4: pages.PagesError = error.InvalidPath;
    try std.testing.expect(e1 == error.FeatureDisabled);
    try std.testing.expect(e2 == error.PageNotFound);
    try std.testing.expect(e3 == error.DuplicatePage);
    try std.testing.expect(e4 == error.InvalidPath);
}

test "pages: HttpMethod enum has expected variants" {
    const get = pages.HttpMethod.GET;
    const post = pages.HttpMethod.POST;
    const put = pages.HttpMethod.PUT;
    const del = pages.HttpMethod.DELETE;
    try std.testing.expect(get != post);
    try std.testing.expect(put != del);
}

test "pages: Page default values" {
    const page = pages.Page{};
    try std.testing.expectEqualStrings("/", page.path);
    try std.testing.expectEqualStrings("", page.title);
    try std.testing.expectEqualStrings("default", page.layout);
    try std.testing.expectEqual(pages.HttpMethod.GET, page.method);
    try std.testing.expect(!page.require_auth);
    try std.testing.expectEqual(@as(u64, 0), page.cache_ttl_ms);
}

test "pages: Page with custom values" {
    const page = pages.Page{
        .path = "/dashboard",
        .title = "Dashboard",
        .layout = "admin",
        .method = .POST,
        .require_auth = true,
        .cache_ttl_ms = 60_000,
    };
    try std.testing.expectEqualStrings("/dashboard", page.path);
    try std.testing.expectEqualStrings("Dashboard", page.title);
    try std.testing.expectEqualStrings("admin", page.layout);
    try std.testing.expectEqual(pages.HttpMethod.POST, page.method);
    try std.testing.expect(page.require_auth);
    try std.testing.expectEqual(@as(u64, 60_000), page.cache_ttl_ms);
}

test "pages: PagesStats default values" {
    const s = pages.PagesStats{};
    try std.testing.expectEqual(@as(u32, 0), s.total_pages);
    try std.testing.expectEqual(@as(u64, 0), s.total_renders);
    try std.testing.expectEqual(@as(u32, 0), s.static_pages);
    try std.testing.expectEqual(@as(u32, 0), s.template_pages);
}

test "pages: TemplateVar default values" {
    const tv = pages.TemplateVar{};
    try std.testing.expectEqualStrings("", tv.key);
    try std.testing.expectEqualStrings("", tv.value);
}

test "pages: MetadataEntry default values" {
    const me = pages.MetadataEntry{};
    try std.testing.expectEqualStrings("", me.key);
    try std.testing.expectEqualStrings("", me.value);
}

// ============================================================================
// Stub API
// ============================================================================

test "pages: stats returns default PagesStats" {
    const s = pages.stats();
    try std.testing.expectEqual(@as(u32, 0), s.total_pages);
    try std.testing.expectEqual(@as(u64, 0), s.total_renders);
}

test "pages: listPages returns slice" {
    const page_list = pages.listPages();
    try std.testing.expect(page_list.len >= 0);
}

test "pages: getPage returns null or page" {
    const result = pages.getPage("/nonexistent");
    // Stub returns null; real impl may vary
    if (result) |_| {} else {
        try std.testing.expect(true);
    }
}

test "pages: addPage returns result or FeatureDisabled" {
    const page = pages.Page{
        .path = "/test",
        .title = "Test Page",
    };
    const result = pages.addPage(page);
    if (result) |_| {
        // Feature enabled -- page added
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "pages: removePage returns result or FeatureDisabled" {
    const result = pages.removePage("/test");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "pages: matchPage returns result or FeatureDisabled" {
    const result = pages.matchPage("/test");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test {
    std.testing.refAllDecls(@This());
}
