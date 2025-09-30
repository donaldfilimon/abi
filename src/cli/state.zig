const std = @import("std");const framework_runtime = @import("../framework/runtime.zig");const framework_config = @import("../framework/config.zig");const errors = @import("errors.zig");pub const SearchResult = struct {    id: u64,    distance: f32,    metadata: ?[]const u8,};pub const VectorStoreError = error{    DimensionMismatch,};pub const VectorRecord = struct {    id: u64,    values: []f32,    metadata: ?[]u8,};pub const VectorStore = struct {    allocator: std.mem.Allocator,    records: std.ArrayList(VectorRecord),    dimension: ?usize = null,    next_id: u64 = 1,    pub fn init(allocator: std.mem.Allocator) VectorStore {        return .{            .allocator = allocator,            .records = std.ArrayList(VectorRecord).init(allocator),        };    }    pub fn deinit(self: *VectorStore) void {        for (self.records.items) |record| {            self.allocator.free(record.values);            if (record.metadata) |meta| {                self.allocator.free(meta);            }        }        self.records.deinit();    }    pub fn insert(self: *VectorStore, values: []const f32, metadata: ?[]const u8) !u64 {
        if (values.len == 0) return error.InvalidVector;
        if (self.dimension) |dim| {
            if (values.len != dim) return VectorStoreError.DimensionMismatch;
        } else {
            self.dimension = values.len;
        }

        const stored_values = try self.allocator.dupe(f32, values);
        var stored_values_needs_free = true;
        errdefer if (stored_values_needs_free) self.allocator.free(stored_values);

        var stored_metadata: ?[]u8 = null;
        var stored_metadata_needs_free = false;
        if (metadata) |meta| {
            const duplicated_metadata = try self.allocator.dupe(u8, meta);
            stored_metadata = duplicated_metadata;
            stored_metadata_needs_free = true;
        }
        errdefer if (stored_metadata_needs_free) self.allocator.free(stored_metadata.?);

        const id = self.next_id;
        self.next_id += 1;
        try self.records.append(.{
            .id = id,
            .values = stored_values,
            .metadata = stored_metadata,
        });
        stored_values_needs_free = false;
        stored_metadata_needs_free = false;

        return id;
    }
pub fn search(self: *VectorStore, allocator: std.mem.Allocator, query: []const f32, k: usize) ![]SearchResult {        if (self.dimension == null or self.records.items.len == 0) {            return allocator.alloc(SearchResult, 0);        }        if (query.len != self.dimension.?) {            return VectorStoreError.DimensionMismatch;        }        const total = self.records.items.len;        var temp = try allocator.alloc(SearchResult, total);        errdefer allocator.free(temp);        for (self.records.items, 0..) |record, idx| {            temp[idx] = .{                .id = record.id,                .distance = distanceSquared(query, record.values),                .metadata = if (record.metadata) |meta| meta else null,            };        }        insertionSort(temp);        const limit = std.math.min(k, temp.len);        var out = try allocator.alloc(SearchResult, limit);        std.mem.copy(SearchResult, out, temp[0..limit]);        allocator.free(temp);        return out;    }};fn distanceSquared(a: []const f32, b: []const f32) f32 {    var sum: f64 = 0;    for (a, b) |lhs, rhs| {        const diff = @as(f64, lhs) - @as(f64, rhs);        sum += diff * diff;    }    return @floatCast(sum);}fn insertionSort(slice: []SearchResult) void {    var i: usize = 1;    while (i < slice.len) : (i += 1) {        const key = slice[i];        var j = i;        while (j > 0 and key.distance < slice[j - 1].distance) : (j -= 1) {            slice[j] = slice[j - 1];        }        slice[j] = key;    }}pub const RateLimiter = struct {    max_actions: usize,    actions: usize = 0,    pub fn init(max_actions: usize) RateLimiter {        return .{ .max_actions = max_actions };    }    pub fn tryConsume(self: *RateLimiter) bool {        if (self.actions >= self.max_actions) return false;        self.actions += 1;        return true;    }};pub const State = struct {    allocator: std.mem.Allocator,    framework: framework_runtime.Framework,    vector_store: VectorStore,    rate_limiter: RateLimiter,    pub fn init(allocator: std.mem.Allocator) !State {        var framework = try framework_runtime.Framework.init(allocator, .{});        errdefer framework.deinit();        return State{            .allocator = allocator,            .framework = framework,            .vector_store = VectorStore.init(allocator),            .rate_limiter = RateLimiter.init(128),        };    }    pub fn deinit(self: *State) void {        self.vector_store.deinit();        self.framework.deinit();    }    pub fn consumeBudget(self: *State) !void {        if (!self.rate_limiter.tryConsume()) {            return errors.CommandError.RateLimited;        }    }};pub fn allFeatures() []const framework_config.Feature {    return &std.meta.tags(framework_config.Feature) ** 1;}

test "VectorStore insert frees metadata on append failure" {
    var failing_state = std.testing.FailingAllocator.init(std.testing.allocator, .{
        .fail_index = 2,
    });
    const failing_alloc = failing_state.allocator();

    var store = VectorStore.init(failing_alloc);
    defer store.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const metadata = "{"label":"test"}";

    try std.testing.expectError(error.OutOfMemory, store.insert(&values, metadata));
    try std.testing.expectEqual(@as(usize, 0), store.records.items.len);
    try std.testing.expect(failing_state.has_induced_failure);
    try std.testing.expectEqual(failing_state.allocated_bytes, failing_state.freed_bytes);
}

