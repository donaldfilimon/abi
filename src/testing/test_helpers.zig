const std = @import("std");

pub const TestAllocator = struct {
    backing: std.mem.Allocator,
    alloc_count: usize = 0,
    dealloc_count: usize = 0,
    total_allocated: usize = 0,
    total_freed: usize = 0,

    pub fn init(backing_allocator: std.mem.Allocator) TestAllocator {
        return .{ .backing = backing_allocator };
    }

    pub fn deinit(self: *TestAllocator) void {
        if (self.alloc_count != self.dealloc_count) {
            std.debug.print(
                "ALLOCATION LEAK: {d} allocs, {d} deallocs, {d} bytes leaked\n",
                .{ self.alloc_count, self.dealloc_count, self.total_allocated - self.total_freed },
            );
        }
    }

    pub fn alloc(self: *TestAllocator, comptime T: type, count: usize) ![]T {
        const buf = try self.backing.alloc(T, count);
        self.alloc_count += 1;
        self.total_allocated += buf.len * @sizeOf(T);
        return buf;
    }

    pub fn free(self: *TestAllocator, buf: []const u8) void {
        self.dealloc_count += 1;
        self.total_freed += buf.len;
        self.backing.free(buf);
    }

    pub fn dupe(self: *TestAllocator, comptime T: type, items: []const T) ![]T {
        const result = try self.backing.dupe(T, items);
        self.alloc_count += 1;
        self.total_allocated += result.len * @sizeOf(T);
        return result;
    }
};

pub const TempDir = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    files: std.ArrayListUnmanaged(TempFile),

    counter: usize = 0,

    pub fn create(allocator: std.mem.Allocator) !TempDir {
        const path = try std.fmt.allocPrint(allocator, "/tmp/zig-test-0", .{});
        return .{
            .allocator = allocator,
            .path = path,
            .files = .empty,
        };
    }

    pub fn deinit(self: *TempDir) void {
        for (self.files.items) |*file| {
            file.deinit();
        }
        self.files.deinit(self.allocator);
        self.allocator.free(self.path);
    }

    pub fn createFile(self: *TempDir, name: []const u8, content: []const u8) !*const TempFile {
        const full_path = try std.fs.path.join(self.allocator, &.{ self.path, name });
        errdefer self.allocator.free(full_path);

        const file_content = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(file_content);

        const file = try self.files.addOne(self.allocator);
        file.* = .{
            .allocator = self.allocator,
            .path = full_path,
            .content = file_content,
        };
        return file;
    }
};

pub const TempFile = struct {
    allocator: std.mem.Allocator,
    path: []const u8,
    content: []const u8 = "",

    pub fn deinit(self: *TempFile) void {
        self.allocator.free(self.path);
        self.allocator.free(self.content);
    }

    pub fn readAll(self: *const TempFile) []const u8 {
        return self.content;
    }
};

pub fn assertAlmostEqual(a: f64, b: f64, epsilon: f64) !void {
    const diff = @abs(a - b);
    if (diff > epsilon) {
        return error.AssertionFailed;
    }
}

pub fn assertContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        return error.AssertionFailed;
    }
}

pub const BenchResult = struct {
    elapsed_ms: f64,
};

pub fn bench(comptime label: []const u8, fn_run: anytype) BenchResult {
    const start = std.time.Instant.now() catch @panic("no timer");
    fn_run();
    const end = std.time.Instant.now() catch @panic("no timer");
    const elapsed_ns = end.since(start);
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    std.debug.print("bench [{s}]: {d:.3}ms\n", .{ label, elapsed_ms });
    return .{ .elapsed_ms = elapsed_ms };
}

pub const MockConnector = struct {
    alloc: std.mem.Allocator,
    responses: std.ArrayListUnmanaged([]const u8),
    call_count: usize = 0,
    initialized: bool = false,

    pub fn init(a: std.mem.Allocator) MockConnector {
        return .{
            .alloc = a,
            .responses = .empty,
        };
    }

    pub fn deinit(self: *MockConnector) void {
        for (self.responses.items) |resp| {
            self.alloc.free(resp);
        }
        self.responses.deinit(self.alloc);
    }

    pub fn initialize(self: *MockConnector) !void {
        self.initialized = true;
    }

    pub fn addResponse(self: *MockConnector, response: []const u8) !void {
        const owned = try self.alloc.dupe(u8, response);
        try self.responses.append(self.alloc, owned);
    }

    pub fn send(self: *MockConnector, _input: []const u8) ![]const u8 {
        _ = _input;
        if (!self.initialized) return error.NotInitialized;
        if (self.call_count >= self.responses.items.len) {
            return error.NoMoreResponses;
        }
        const resp = self.responses.items[self.call_count];
        self.call_count += 1;
        return resp;
    }

    pub fn close(self: *MockConnector) void {
        self.initialized = false;
    }
};

pub const MockStorage = struct {
    alloc: std.mem.Allocator,
    data: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn init(a: std.mem.Allocator) MockStorage {
        return .{ .alloc = a };
    }

    pub fn deinit(self: *MockStorage) void {
        var it = self.data.iterator();
        while (it.next()) |entry| {
            self.alloc.free(entry.key_ptr.*);
            self.alloc.free(entry.value_ptr.*);
        }
        self.data.deinit(self.alloc);
    }

    pub fn put(self: *MockStorage, key: []const u8, value: []const u8) !void {
        const owned_key = try self.alloc.dupe(u8, key);
        errdefer self.alloc.free(owned_key);
        const owned_val = try self.alloc.dupe(u8, value);
        errdefer self.alloc.free(owned_val);

        const result = try self.data.getOrPut(self.alloc, owned_key);
        if (result.found_existing) {
            self.alloc.free(owned_key);
            self.alloc.free(result.value_ptr.*);
        }
        result.key_ptr.* = owned_key;
        result.value_ptr.* = owned_val;
    }

    pub fn get(self: *const MockStorage, key: []const u8) ?[]const u8 {
        return self.data.get(key);
    }

    pub fn remove(self: *MockStorage, key: []const u8) bool {
        if (self.data.fetchRemove(key)) |entry| {
            self.alloc.free(entry.key);
            self.alloc.free(entry.value);
            return true;
        }
        return false;
    }

    pub fn count(self: *const MockStorage) usize {
        return self.data.count();
    }

    pub fn clear(self: *MockStorage) void {
        var it = self.data.iterator();
        while (it.next()) |entry| {
            self.alloc.free(entry.key_ptr.*);
            self.alloc.free(entry.value_ptr.*);
        }
        self.data.clearRetainingCapacity();
    }
};

test "TestAllocator tracks allocations" {
    var tracker = TestAllocator.init(std.testing.allocator);
    defer tracker.deinit();

    const buf = try tracker.alloc(u8, 64);
    defer tracker.free(buf);

    try std.testing.expect(tracker.alloc_count == 1);
    try std.testing.expect(tracker.dealloc_count == 0);
}

test "TempDir create and cleanup" {
    var tmp = try TempDir.create(std.testing.allocator);
    defer tmp.deinit();

    try std.testing.expect(tmp.path.len > 0);
}

test "TempFile create and read" {
    var tmp = try TempDir.create(std.testing.allocator);
    defer tmp.deinit();

    const file = try tmp.createFile("test.txt", "hello world");
    try std.testing.expectEqualStrings("hello world", file.readAll());
}

test "assertAlmostEqual" {
    try assertAlmostEqual(1.0, 1.0001, 0.001);
    try std.testing.expectError(
        error.AssertionFailed,
        assertAlmostEqual(1.0, 2.0, 0.001),
    );
}

test "assertContains" {
    try assertContains("hello world", "world");
    try std.testing.expectError(
        error.AssertionFailed,
        assertContains("hello world", "missing"),
    );
}

test "MockConnector lifecycle" {
    var connector = MockConnector.init(std.testing.allocator);
    defer connector.deinit();

    try std.testing.expectError(error.NotInitialized, connector.send("test"));

    try connector.initialize();
    try connector.addResponse("response 1");
    try connector.addResponse("response 2");

    const r1 = try connector.send("input 1");
    try std.testing.expectEqualStrings("response 1", r1);

    const r2 = try connector.send("input 2");
    try std.testing.expectEqualStrings("response 2", r2);

    try std.testing.expectError(error.NoMoreResponses, connector.send("input 3"));

    connector.close();
}

test "MockStorage operations" {
    var storage = MockStorage.init(std.testing.allocator);
    defer storage.deinit();

    try storage.put("key1", "value1");
    try storage.put("key2", "value2");

    try std.testing.expectEqualStrings("value1", storage.get("key1").?);
    try std.testing.expectEqual(@as(usize, 2), storage.count());

    try std.testing.expect(storage.remove("key1"));
    try std.testing.expect(storage.get("key1") == null);
    try std.testing.expectEqual(@as(usize, 1), storage.count());

    storage.clear();
    try std.testing.expectEqual(@as(usize, 0), storage.count());
}

test {
    std.testing.refAllDecls(@This());
}
