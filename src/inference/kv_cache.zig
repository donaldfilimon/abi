//! Paged KV-cache.
//!
//! Block-based memory management for key-value caches used in transformer
//! inference. Allocates fixed-size pages and maps sequence IDs to page lists,
//! enabling efficient memory reuse across requests.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    num_pages: u32 = 10000,
    page_size: u32 = 16,
    num_layers: u32 = 32,
    num_heads: u32 = 32,
    head_dim: u32 = 128,
};

const Page = struct {
    data: []f32,
    used_tokens: u32,
    seq_id: ?u64,
};

pub const PagedKVCache = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,
    pages: []Page,
    free_pages: std.ArrayListUnmanaged(u32),
    seq_pages: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(u32)),
    total_pages: u32,

    pub fn init(allocator: Allocator, config: Config) !Self {
        const page_data_size = config.page_size * config.num_heads * config.head_dim * 2;
        const pages = try allocator.alloc(Page, config.num_pages);

        var free_pages = std.ArrayListUnmanaged(u32).empty;
        errdefer free_pages.deinit(allocator);
        for (pages, 0..) |*page, i| {
            page.data = try allocator.alloc(f32, page_data_size);
            @memset(page.data, 0.0);
            page.used_tokens = 0;
            page.seq_id = null;
            try free_pages.append(allocator, @intCast(i));
        }

        return .{
            .allocator = allocator,
            .config = config,
            .pages = pages,
            .free_pages = free_pages,
            .seq_pages = .empty,
            .total_pages = config.num_pages,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.pages) |page| {
            self.allocator.free(page.data);
        }
        self.allocator.free(self.pages);
        self.free_pages.deinit(self.allocator);

        var it = self.seq_pages.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.seq_pages.deinit(self.allocator);
    }

    pub fn allocate(self: *Self, seq_id: u64, num_tokens: u32) !bool {
        const pages_needed = (num_tokens + self.config.page_size - 1) / self.config.page_size;
        if (self.free_pages.items.len < pages_needed) return false;

        const entry = try self.seq_pages.getOrPut(self.allocator, seq_id);
        if (!entry.found_existing) {
            entry.value_ptr.* = .empty;
        }

        for (0..pages_needed) |_| {
            const page_id = self.free_pages.pop();
            self.pages[page_id].seq_id = seq_id;
            self.pages[page_id].used_tokens = 0;
            try entry.value_ptr.append(self.allocator, page_id);
        }

        return true;
    }

    pub fn free(self: *Self, seq_id: u64) void {
        if (self.seq_pages.fetchRemove(seq_id)) |kv| {
            var page_list = kv.value;
            for (page_list.items) |page_id| {
                self.pages[page_id].seq_id = null;
                self.pages[page_id].used_tokens = 0;
                self.free_pages.append(self.allocator, page_id) catch |err| {
                    std.log.err("KV cache page {d} leaked — free list append failed: {t}", .{ page_id, err });
                };
            }
            page_list.deinit(self.allocator);
        }
    }

    pub fn getUtilization(self: *const Self) f32 {
        const used: f32 = @floatFromInt(self.total_pages - @as(u32, @intCast(self.free_pages.items.len)));
        const total: f32 = @floatFromInt(self.total_pages);
        return used / total;
    }

    pub fn activeSequences(self: *const Self) usize {
        return self.seq_pages.count();
    }
};

test "kv cache allocate and free" {
    const allocator = std.testing.allocator;

    var cache = try PagedKVCache.init(allocator, .{
        .num_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
    });
    defer cache.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cache.getUtilization(), 1e-5);

    const ok = try cache.allocate(1, 32);
    try std.testing.expect(ok);
    try std.testing.expect(cache.getUtilization() > 0.0);
    try std.testing.expectEqual(@as(usize, 1), cache.activeSequences());

    cache.free(1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cache.getUtilization(), 1e-5);
    try std.testing.expectEqual(@as(usize, 0), cache.activeSequences());
}

test "kv cache out of memory" {
    const allocator = std.testing.allocator;

    var cache = try PagedKVCache.init(allocator, .{
        .num_pages = 2,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
    });
    defer cache.deinit();

    const ok1 = try cache.allocate(1, 32);
    try std.testing.expect(ok1);

    const ok2 = try cache.allocate(2, 16);
    try std.testing.expect(!ok2);
}
