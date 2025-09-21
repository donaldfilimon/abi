//! Unit tests for the JSON component.

const std = @import("std");
const testing = std.testing;

test "JSON-like data structure creation" {
    std.debug.print("[DEBUG] Starting JSON-like data structure creation test\n", .{});

    const Data = struct {
        @"test": i32,
        name: []const u8,
        value: f64,
    };

    std.debug.print("[DEBUG] Creating JSON-like data structure...\n", .{});
    const data = Data{
        .@"test" = 123,
        .name = "test_data",
        .value = 45.67,
    };
    std.debug.print("[DEBUG] ✓ Data structure created successfully\n", .{});

    std.debug.print("[DEBUG] Verifying field values:\n", .{});

    try testing.expectEqual(@as(i32, 123), data.@"test");
    std.debug.print("[DEBUG] ✓ 'test' field: {d}\n", .{data.@"test"});

    try testing.expectEqualStrings("test_data", data.name);
    std.debug.print("[DEBUG] ✓ 'name' field: {s}\n", .{data.name});

    try testing.expectEqual(@as(f64, 45.67), data.value);
    std.debug.print("[DEBUG] ✓ 'value' field: {d:.2}\n", .{data.value});

    std.debug.print("[DEBUG] JSON-like data structure creation test completed successfully\n", .{});
}

test "JSON-like data structure with optional fields" {
    std.debug.print("[DEBUG] Starting JSON-like data structure with optional fields test\n", .{});

    const OptionalData = struct {
        id: ?u32,
        name: []const u8,
        metadata: ?[]const u8,
    };

    std.debug.print("[DEBUG] Creating data structure with optional fields...\n", .{});
    const data = OptionalData{
        .id = 42,
        .name = "optional_test",
        .metadata = null,
    };
    std.debug.print("[DEBUG] ✓ Optional data structure created\n", .{});

    std.debug.print("[DEBUG] Verifying optional field handling:\n", .{});

    try testing.expect(data.id != null);
    std.debug.print("[DEBUG] ✓ ID field is not null\n", .{});

    try testing.expectEqual(@as(u32, 42), data.id.?);
    std.debug.print("[DEBUG] ✓ ID value: {d}\n", .{data.id.?});

    try testing.expectEqualStrings("optional_test", data.name);
    std.debug.print("[DEBUG] ✓ Name field: {s}\n", .{data.name});

    try testing.expect(data.metadata == null);
    std.debug.print("[DEBUG] ✓ Metadata field is null as expected\n", .{});

    std.debug.print("[DEBUG] JSON-like data structure with optional fields test completed successfully\n", .{});
}

test "JSON-like data structure with arrays" {
    std.debug.print("[DEBUG] Starting JSON-like data structure with arrays test\n", .{});

    const ArrayData = struct {
        numbers: []const i32,
        strings: []const []const u8,
    };

    std.debug.print("[DEBUG] Creating test data arrays...\n", .{});
    const numbers = [_]i32{ 1, 2, 3, 4, 5 };
    const strings = [_][]const u8{ "one", "two", "three" };
    std.debug.print("[DEBUG] ✓ Test arrays created\n", .{});

    std.debug.print("[DEBUG] Creating data structure with arrays...\n", .{});
    const data = ArrayData{
        .numbers = &numbers,
        .strings = &strings,
    };
    std.debug.print("[DEBUG] ✓ Array data structure created\n", .{});

    std.debug.print("[DEBUG] Verifying array contents:\n", .{});

    try testing.expectEqual(@as(usize, 5), data.numbers.len);
    std.debug.print("[DEBUG] ✓ Numbers array length: {d}\n", .{data.numbers.len});

    try testing.expectEqual(@as(usize, 3), data.strings.len);
    std.debug.print("[DEBUG] ✓ Strings array length: {d}\n", .{data.strings.len});

    try testing.expectEqual(@as(i32, 1), data.numbers[0]);
    std.debug.print("[DEBUG] ✓ First number: {d}\n", .{data.numbers[0]});

    try testing.expectEqual(@as(i32, 5), data.numbers[4]);
    std.debug.print("[DEBUG] ✓ Last number: {d}\n", .{data.numbers[4]});

    try testing.expectEqualStrings("one", data.strings[0]);
    std.debug.print("[DEBUG] ✓ First string: {s}\n", .{data.strings[0]});

    try testing.expectEqualStrings("three", data.strings[2]);
    std.debug.print("[DEBUG] ✓ Last string: {s}\n", .{data.strings[2]});

    std.debug.print("[DEBUG] JSON-like data structure with arrays test completed successfully\n", .{});
}

test "JSON-like data structure serialization simulation" {
    std.debug.print("[DEBUG] Starting JSON-like data structure serialization simulation test\n", .{});

    const Data = struct {
        id: u32,
        name: []const u8,
        active: bool,

        fn toJsonString(self: @This(), allocator: std.mem.Allocator) ![]u8 {
            std.debug.print("[DEBUG] Serializing data structure to JSON string...\n", .{});
            return std.fmt.allocPrint(allocator, "{{\"id\":{d},\"name\":\"{s}\",\"active\":{}}}", .{ self.id, self.name, self.active });
        }
    };

    std.debug.print("[DEBUG] Creating test data structure...\n", .{});
    const data = Data{
        .id = 123,
        .name = "test_item",
        .active = true,
    };
    std.debug.print("[DEBUG] ✓ Test data created: id={d}, name={s}, active={}\n", .{ data.id, data.name, data.active });

    std.debug.print("[DEBUG] Serializing to JSON...\n", .{});
    const json_string = try data.toJsonString(testing.allocator);
    defer testing.allocator.free(json_string);
    std.debug.print("[DEBUG] ✓ JSON string created: {s}\n", .{json_string});

    std.debug.print("[DEBUG] Verifying JSON content:\n", .{});

    const has_id = std.mem.indexOf(u8, json_string, "\"id\":123") != null;
    std.debug.print("[DEBUG] Contains ID field: {}\n", .{has_id});
    try testing.expect(has_id);

    const has_name = std.mem.indexOf(u8, json_string, "\"name\":\"test_item\"") != null;
    std.debug.print("[DEBUG] Contains name field: {}\n", .{has_name});
    try testing.expect(has_name);

    const has_active = std.mem.indexOf(u8, json_string, "\"active\":true") != null;
    std.debug.print("[DEBUG] Contains active field: {}\n", .{has_active});
    try testing.expect(has_active);

    std.debug.print("[DEBUG] JSON-like data structure serialization simulation test completed successfully\n", .{});
}
