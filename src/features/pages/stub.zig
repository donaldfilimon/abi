//! Pages Stub Module
//!
//! API-compatible no-op implementations when pages feature is disabled.

const std = @import("std");
const core_config = @import("../../core/config/content.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const PagesConfig = core_config.PagesConfig;

pub const PagesError = error{
    FeatureDisabled,
    PageNotFound,
    DuplicatePage,
    TooManyPages,
    InvalidPath,
    TemplateError,
    OutOfMemory,
};

pub const HttpMethod = enum { GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS };

pub const MetadataEntry = struct {
    key: []const u8 = "",
    value: []const u8 = "",
};

pub const TemplateVar = struct {
    key: []const u8 = "",
    value: []const u8 = "",
};

pub const TemplateRef = struct {
    source: []const u8 = "",
    default_vars: [8]TemplateVar = [_]TemplateVar{.{}} ** 8,
    var_count: u8 = 0,
};

pub const PageContent = union(enum) {
    static: []const u8,
    template: TemplateRef,
};

pub const Page = struct {
    path: []const u8 = "/",
    title: []const u8 = "",
    content: PageContent = .{ .static = "" },
    layout: []const u8 = "default",
    method: HttpMethod = .GET,
    require_auth: bool = false,
    cache_ttl_ms: u64 = 0,
    metadata: [4]MetadataEntry = [_]MetadataEntry{.{}} ** 4,
    metadata_count: u8 = 0,
};

pub const PageMatch = struct {
    page: Page,
    params: [8]Param = [_]Param{.{}} ** 8,
    param_count: u8 = 0,

    pub const Param = struct {
        name: []const u8 = "",
        value: []const u8 = "",
    };

    pub fn getParam(self: *const PageMatch, name: []const u8) ?[]const u8 {
        _ = self;
        _ = name;
        return null;
    }
};

pub const RenderResult = struct {
    title: []const u8 = "",
    body: []u8 = &.{},
    layout: []const u8 = "default",
    body_owned: bool = false,

    pub fn deinit(self: *RenderResult, allocator: std.mem.Allocator) void {
        if (self.body_owned) {
            allocator.free(self.body);
        }
        self.* = undefined;
    }
};

pub const PagesStats = struct {
    total_pages: u32 = 0,
    total_renders: u64 = 0,
    static_pages: u32 = 0,
    template_pages: u32 = 0,
};

pub const Context = stub_context.StubContextWithConfig(PagesConfig);

pub fn init(_: std.mem.Allocator, _: PagesConfig) PagesError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn addPage(_: Page) PagesError!void {
    return error.FeatureDisabled;
}
pub fn removePage(_: []const u8) PagesError!bool {
    return error.FeatureDisabled;
}
pub fn getPage(_: []const u8) ?Page {
    return null;
}
pub fn matchPage(_: []const u8) PagesError!?PageMatch {
    return error.FeatureDisabled;
}
pub fn renderPage(_: std.mem.Allocator, _: []const u8, _: []const TemplateVar) PagesError!RenderResult {
    return error.FeatureDisabled;
}
pub fn listPages() []const Page {
    return &.{};
}
pub fn stats() PagesStats {
    return .{};
}

test {
    std.testing.refAllDecls(@This());
}
