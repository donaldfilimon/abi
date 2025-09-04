//! Compile-Time Reflection and Code Generation for WDBX-AI
//!
//! This module leverages Zig's powerful compile-time capabilities to:
//! - Generate boilerplate code automatically
//! - Validate data structures at compile time
//! - Create serialization/deserialization functions
//! - Generate test code and documentation

const std = @import("std");
const print = std.debug.print;
const testing = std.testing;

/// Generate a toString function for any struct at compile time
pub fn generateToString(comptime T: type) fn (T) []const u8 {
    return struct {
        fn toString(self: T) []const u8 {
            _ = self;
            comptime var result: []const u8 = @typeName(T) ++ " { ";

            inline for (std.meta.fields(T)) |field| {
                result = result ++ field.name ++ ": ";
                switch (@typeInfo(field.type)) {
                    .Int, .Float => result = result ++ "{" ++ field.name ++ "}, ",
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            result = result ++ "\"{" ++ field.name ++ "}\", ";
                        } else {
                            result = result ++ "{" ++ field.name ++ "}, ";
                        }
                    },
                    .Bool => result = result ++ "{" ++ field.name ++ "}, ",
                    else => result = result ++ "{" ++ field.name ++ "}, ",
                }
            }

            result = result ++ " }";
            return result;
        }
    }.toString;
}

/// Generate an equals function for any struct at compile time
pub fn generateEquals(comptime T: type) fn (T, T) bool {
    return struct {
        fn equals(a: T, b: T) bool {
            inline for (std.meta.fields(T)) |field| {
                const field_a = @field(a, field.name);
                const field_b = @field(b, field.name);

                switch (@typeInfo(field.type)) {
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            if (!std.mem.eql(u8, field_a, field_b)) return false;
                        } else {
                            if (field_a != field_b) return false;
                        }
                    },
                    else => {
                        if (field_a != field_b) return false;
                    },
                }
            }
            return true;
        }
    }.equals;
}

/// Generate a hash function for any struct at compile time
pub fn generateHash(comptime T: type) fn (T) u64 {
    return struct {
        fn hash(self: T) u64 {
            var hasher = std.hash.Wyhash.init(0);

            inline for (std.meta.fields(T)) |field| {
                const field_value = @field(self, field.name);

                switch (@typeInfo(field.type)) {
                    .Int => hasher.update(std.mem.asBytes(&field_value)),
                    .Float => hasher.update(std.mem.asBytes(&field_value)),
                    .Bool => hasher.update(&[_]u8{if (field_value) 1 else 0}),
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            hasher.update(field_value);
                        } else {
                            hasher.update(std.mem.asBytes(&field_value));
                        }
                    },
                    else => hasher.update(std.mem.asBytes(&field_value)),
                }
            }

            return hasher.final();
        }
    }.hash;
}

/// Generate JSON serialization functions at compile time
pub fn generateJsonSerialization(comptime T: type) type {
    return struct {
        pub fn toJson(allocator: std.mem.Allocator, value: T) ![]u8 {
            var result = std.ArrayList(u8).init(allocator);
            defer result.deinit();

            try result.append('{');

            inline for (std.meta.fields(T), 0..) |field, i| {
                if (i > 0) try result.appendSlice(", ");

                try result.append('"');
                try result.appendSlice(field.name);
                try result.appendSlice("\": ");

                const field_value = @field(value, field.name);

                switch (@typeInfo(field.type)) {
                    .Int, .Float => {
                        const str = try std.fmt.allocPrint(allocator, "{}", .{field_value});
                        defer allocator.free(str);
                        try result.appendSlice(str);
                    },
                    .Bool => {
                        try result.appendSlice(if (field_value) "true" else "false");
                    },
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            try result.append('"');
                            try result.appendSlice(field_value);
                            try result.append('"');
                        } else {
                            const str = try std.fmt.allocPrint(allocator, "{}", .{field_value});
                            defer allocator.free(str);
                            try result.appendSlice(str);
                        }
                    },
                    else => {
                        const str = try std.fmt.allocPrint(allocator, "{}", .{field_value});
                        defer allocator.free(str);
                        try result.appendSlice(str);
                    },
                }
            }

            try result.append('}');
            return result.toOwnedSlice();
        }

        pub fn fromJson(allocator: std.mem.Allocator, json_str: []const u8) !T {
            _ = allocator;
            // Simplified JSON parsing - in production use a proper JSON parser
            var result: T = undefined;

            // For demonstration, initialize with default values
            inline for (std.meta.fields(T)) |field| {
                switch (@typeInfo(field.type)) {
                    .Int => @field(result, field.name) = 0,
                    .Float => @field(result, field.name) = 0.0,
                    .Bool => @field(result, field.name) = false,
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            @field(result, field.name) = "";
                        }
                    },
                    else => {},
                }
            }

            _ = json_str; // TODO: Implement actual JSON parsing
            return result;
        }
    };
}

/// Validate struct constraints at compile time
pub fn validateStruct(comptime T: type) void {
    // Check for common naming conventions
    inline for (std.meta.fields(T)) |field| {
        if (std.mem.startsWith(u8, field.name, "_")) {
            @compileError("Field names should not start with underscore: " ++ field.name);
        }
    }

    // Note: Advanced pointer validation removed due to TypeInfo compatibility issues
    // In production, implement specific validation for your use case
}

/// Generate builder pattern for structs
pub fn generateBuilder(comptime T: type) type {
    return struct {
        const Self = @This();
        const BuilderType = T;

        data: T,

        pub fn init() Self {
            var result: T = undefined;

            // Initialize with zero values
            inline for (std.meta.fields(T)) |field| {
                switch (@typeInfo(field.type)) {
                    .Int => @field(result, field.name) = 0,
                    .Float => @field(result, field.name) = 0.0,
                    .Bool => @field(result, field.name) = false,
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            @field(result, field.name) = "";
                        }
                    },
                    else => {},
                }
            }

            return .{ .data = result };
        }

        // Simplified setter methods - complex metaprogramming removed for compatibility
        pub fn setField(self: *Self, comptime field_name: []const u8, value: anytype) *Self {
            @field(self.data, field_name) = value;
            return self;
        }

        pub fn build(self: Self) T {
            return self.data;
        }
    };
}

/// Generate test functions for a struct
pub fn generateTests(comptime T: type) type {
    return struct {
        pub fn testBasicOperations() !void {
            // Create test instance
            var instance: T = undefined;

            // Initialize with test values
            inline for (std.meta.fields(T)) |field| {
                switch (@typeInfo(field.type)) {
                    .Int => @field(instance, field.name) = 42,
                    .Float => @field(instance, field.name) = 3.14,
                    .Bool => @field(instance, field.name) = true,
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            @field(instance, field.name) = "test";
                        }
                    },
                    else => {},
                }
            }

            // Test generated functions
            const equals_fn = generateEquals(T);
            const hash_fn = generateHash(T);

            try testing.expect(equals_fn(instance, instance));
            const hash_val = hash_fn(instance);
            try testing.expect(hash_val != 0);
        }

        pub fn testJsonSerialization() !void {
            const JsonSerializer = generateJsonSerialization(T);

            var instance: T = undefined;
            inline for (std.meta.fields(T)) |field| {
                switch (@typeInfo(field.type)) {
                    .Int => @field(instance, field.name) = 123,
                    .Float => @field(instance, field.name) = 45.6,
                    .Bool => @field(instance, field.name) = true,
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            @field(instance, field.name) = "hello";
                        }
                    },
                    else => {},
                }
            }

            const json = try JsonSerializer.toJson(testing.allocator, instance);
            defer testing.allocator.free(json);

            try testing.expect(json.len > 0);
            try testing.expect(std.mem.indexOf(u8, json, "{") != null);
            try testing.expect(std.mem.indexOf(u8, json, "}") != null);
        }
    };
}

/// Example usage macro
pub fn enhanceStruct(comptime T: type) type {
    // Validate the struct at compile time
    validateStruct(T);

    return struct {
        pub const BaseType = T;
        pub const toString = generateToString(T);
        pub const equals = generateEquals(T);
        pub const hash = generateHash(T);
        pub const JsonSerializer = generateJsonSerialization(T);
        pub const Builder = generateBuilder(T);
        pub const Tests = generateTests(T);

        pub fn create() T {
            var result: T = undefined;
            inline for (std.meta.fields(T)) |field| {
                switch (@typeInfo(field.type)) {
                    .Int => @field(result, field.name) = 0,
                    .Float => @field(result, field.name) = 0.0,
                    .Bool => @field(result, field.name) = false,
                    .Pointer => |ptr_info| {
                        if (ptr_info.child == u8) {
                            @field(result, field.name) = "";
                        }
                    },
                    else => {},
                }
            }
            return result;
        }
    };
}

// Example struct for testing
const Person = struct {
    name: []const u8,
    age: u32,
    active: bool,
};

const EnhancedPerson = enhanceStruct(Person);

test "compile-time reflection" {
    // Test enhanced struct functionality
    var person = EnhancedPerson.create();
    person.name = "Alice";
    person.age = 30;
    person.active = true;

    var person2 = person;
    person2.age = 31;

    try testing.expect(EnhancedPerson.equals(person, person));
    try testing.expect(!EnhancedPerson.equals(person, person2));

    const hash1 = EnhancedPerson.hash(person);
    const hash2 = EnhancedPerson.hash(person2);
    try testing.expect(hash1 != hash2);
}

test "json serialization" {
    const person: Person = .{
        .name = "Bob",
        .age = 25,
        .active = false,
    };

    const json = try EnhancedPerson.JsonSerializer.toJson(testing.allocator, person);
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "Bob") != null);
    try testing.expect(std.mem.indexOf(u8, json, "25") != null);
    try testing.expect(std.mem.indexOf(u8, json, "false") != null);
}

test "builder pattern" {
    var builder = EnhancedPerson.Builder.init();
    const person = builder
        .setField("name", "Charlie")
        .setField("age", @as(u32, 35))
        .setField("active", true)
        .build();

    try testing.expectEqualStrings("Charlie", person.name);
    try testing.expectEqual(@as(u32, 35), person.age);
    try testing.expect(person.active);
}
