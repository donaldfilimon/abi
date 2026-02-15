//! Common Code Generation Utilities
//!
//! Shared utilities used by all code generators for string building,
//! type mapping, and expression formatting.

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");

/// String builder for generating source code.
pub const CodeWriter = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayListUnmanaged(u8),
    indent_level: u32,
    indent_string: []const u8,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .output = .empty,
            .indent_level = 0,
            .indent_string = "    ", // 4 spaces
        };
    }

    pub fn deinit(self: *Self) void {
        self.output.deinit(self.allocator);
    }

    /// Write a string.
    pub fn write(self: *Self, s: []const u8) !void {
        try self.output.appendSlice(self.allocator, s);
    }

    /// Write a formatted string.
    pub fn writeFmt(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        var buf: [4096]u8 = undefined;
        const formatted = std.fmt.bufPrint(&buf, fmt, args) catch |err| switch (err) {
            error.NoSpaceLeft => {
                // If buffer is too small, fall back to allocating
                const text = try std.fmt.allocPrint(self.allocator, fmt, args);
                defer self.allocator.free(text);
                try self.output.appendSlice(self.allocator, text);
                return;
            },
        };
        try self.output.appendSlice(self.allocator, formatted);
    }

    /// Write a newline.
    pub fn newline(self: *Self) !void {
        try self.write("\n");
    }

    /// Write indentation.
    pub fn writeIndent(self: *Self) !void {
        var i: u32 = 0;
        while (i < self.indent_level) : (i += 1) {
            try self.write(self.indent_string);
        }
    }

    /// Write an indented line.
    pub fn writeLine(self: *Self, s: []const u8) !void {
        try self.writeIndent();
        try self.write(s);
        try self.newline();
    }

    /// Write a formatted indented line.
    pub fn writeLineFmt(self: *Self, comptime fmt: []const u8, args: anytype) !void {
        try self.writeIndent();
        try self.writeFmt(fmt, args);
        try self.newline();
    }

    /// Increase indentation.
    pub fn indent(self: *Self) void {
        self.indent_level += 1;
    }

    /// Decrease indentation.
    pub fn dedent(self: *Self) void {
        if (self.indent_level > 0) {
            self.indent_level -= 1;
        }
    }

    /// Get the generated code.
    pub fn getCode(self: *Self) ![]const u8 {
        return self.output.toOwnedSlice(self.allocator);
    }

    /// Get current output as a slice (does not transfer ownership).
    pub fn getCurrentOutput(self: *Self) []const u8 {
        return self.output.items;
    }
};

/// Common type names across backends.
pub const TypeNames = struct {
    bool_name: []const u8,
    i8_name: []const u8,
    i16_name: []const u8,
    i32_name: []const u8,
    i64_name: []const u8,
    u8_name: []const u8,
    u16_name: []const u8,
    u32_name: []const u8,
    u64_name: []const u8,
    f16_name: []const u8,
    f32_name: []const u8,
    f64_name: []const u8,
    void_name: []const u8,

    // Vector type format (e.g., "vec{d}" or "float{d}")
    vec_float_fmt: []const u8,
    vec_int_fmt: []const u8,
    vec_uint_fmt: []const u8,

    pub fn getScalarName(self: TypeNames, scalar: types.ScalarType) []const u8 {
        return switch (scalar) {
            .bool_ => self.bool_name,
            .i8 => self.i8_name,
            .i16 => self.i16_name,
            .i32 => self.i32_name,
            .i64 => self.i64_name,
            .u8 => self.u8_name,
            .u16 => self.u16_name,
            .u32 => self.u32_name,
            .u64 => self.u64_name,
            .f16 => self.f16_name,
            .f32 => self.f32_name,
            .f64 => self.f64_name,
        };
    }

    /// GLSL type names.
    pub const glsl = TypeNames{
        .bool_name = "bool",
        .i8_name = "int",
        .i16_name = "int",
        .i32_name = "int",
        .i64_name = "int64_t",
        .u8_name = "uint",
        .u16_name = "uint",
        .u32_name = "uint",
        .u64_name = "uint64_t",
        .f16_name = "float16_t",
        .f32_name = "float",
        .f64_name = "double",
        .void_name = "void",
        .vec_float_fmt = "vec",
        .vec_int_fmt = "ivec",
        .vec_uint_fmt = "uvec",
    };

    /// WGSL type names.
    pub const wgsl = TypeNames{
        .bool_name = "bool",
        .i8_name = "i32",
        .i16_name = "i32",
        .i32_name = "i32",
        .i64_name = "i64",
        .u8_name = "u32",
        .u16_name = "u32",
        .u32_name = "u32",
        .u64_name = "u64",
        .f16_name = "f16",
        .f32_name = "f32",
        .f64_name = "f64",
        .void_name = "void",
        .vec_float_fmt = "vec",
        .vec_int_fmt = "vec",
        .vec_uint_fmt = "vec",
    };

    /// CUDA type names.
    pub const cuda = TypeNames{
        .bool_name = "bool",
        .i8_name = "int8_t",
        .i16_name = "int16_t",
        .i32_name = "int",
        .i64_name = "int64_t",
        .u8_name = "uint8_t",
        .u16_name = "uint16_t",
        .u32_name = "unsigned int",
        .u64_name = "uint64_t",
        .f16_name = "half",
        .f32_name = "float",
        .f64_name = "double",
        .void_name = "void",
        .vec_float_fmt = "float",
        .vec_int_fmt = "int",
        .vec_uint_fmt = "uint",
    };

    /// Metal type names.
    pub const msl = TypeNames{
        .bool_name = "bool",
        .i8_name = "int8_t",
        .i16_name = "int16_t",
        .i32_name = "int",
        .i64_name = "int64_t",
        .u8_name = "uint8_t",
        .u16_name = "uint16_t",
        .u32_name = "uint",
        .u64_name = "uint64_t",
        .f16_name = "half",
        .f32_name = "float",
        .f64_name = "double",
        .void_name = "void",
        .vec_float_fmt = "float",
        .vec_int_fmt = "int",
        .vec_uint_fmt = "uint",
    };
};

/// Common operator symbols.
pub const OperatorSymbols = struct {
    /// Get the symbol for a binary operator.
    pub fn binaryOp(op: expr.BinaryOp) []const u8 {
        return switch (op) {
            .add => " + ",
            .sub => " - ",
            .mul => " * ",
            .div => " / ",
            .mod => " % ",
            .eq => " == ",
            .ne => " != ",
            .lt => " < ",
            .le => " <= ",
            .gt => " > ",
            .ge => " >= ",
            .and_ => " && ",
            .or_ => " || ",
            .xor => " ^^ ",
            .bit_and => " & ",
            .bit_or => " | ",
            .bit_xor => " ^ ",
            .shl => " << ",
            .shr => " >> ",
            else => " ?? ", // Function-like ops
        };
    }

    /// Get the symbol for a unary operator.
    pub fn unaryOp(op: expr.UnaryOp) []const u8 {
        return switch (op) {
            .neg => "-",
            .not => "!",
            .bit_not => "~",
            else => "", // Function-like ops
        };
    }

    /// Check if binary op is an infix operator.
    pub fn isInfixBinaryOp(op: expr.BinaryOp) bool {
        return op.isInfix();
    }

    /// Check if unary op is a prefix operator.
    pub fn isPrefixUnaryOp(op: expr.UnaryOp) bool {
        return op.isPrefix();
    }
};

/// Common built-in function mappings.
pub const BuiltinMapping = struct {
    glsl: []const u8,
    wgsl: []const u8,
    cuda: []const u8,
    msl: []const u8,

    pub fn get(self: BuiltinMapping, target: Target) []const u8 {
        return switch (target) {
            .glsl => self.glsl,
            .wgsl => self.wgsl,
            .cuda => self.cuda,
            .msl => self.msl,
        };
    }

    pub const Target = enum { glsl, wgsl, cuda, msl };
};

/// Built-in function name mappings.
pub const builtins = struct {
    pub const barrier = BuiltinMapping{
        .glsl = "barrier()",
        .wgsl = "workgroupBarrier()",
        .cuda = "__syncthreads()",
        .msl = "threadgroup_barrier(mem_flags::mem_threadgroup)",
    };

    pub const memory_barrier = BuiltinMapping{
        .glsl = "memoryBarrier()",
        .wgsl = "storageBarrier()",
        .cuda = "__threadfence()",
        .msl = "threadgroup_barrier(mem_flags::mem_device)",
    };

    pub const atomic_add = BuiltinMapping{
        .glsl = "atomicAdd",
        .wgsl = "atomicAdd",
        .cuda = "atomicAdd",
        .msl = "atomic_fetch_add_explicit",
    };

    pub const atomic_sub = BuiltinMapping{
        .glsl = "atomicAdd", // Use atomicAdd with negative value
        .wgsl = "atomicSub",
        .cuda = "atomicSub",
        .msl = "atomic_fetch_sub_explicit",
    };

    pub const atomic_min = BuiltinMapping{
        .glsl = "atomicMin",
        .wgsl = "atomicMin",
        .cuda = "atomicMin",
        .msl = "atomic_fetch_min_explicit",
    };

    pub const atomic_max = BuiltinMapping{
        .glsl = "atomicMax",
        .wgsl = "atomicMax",
        .cuda = "atomicMax",
        .msl = "atomic_fetch_max_explicit",
    };

    pub const clamp = BuiltinMapping{
        .glsl = "clamp",
        .wgsl = "clamp",
        .cuda = "clamp", // Need to define
        .msl = "clamp",
    };

    pub const mix = BuiltinMapping{
        .glsl = "mix",
        .wgsl = "mix",
        .cuda = "lerp", // Different name
        .msl = "mix",
    };

    pub const smoothstep = BuiltinMapping{
        .glsl = "smoothstep",
        .wgsl = "smoothstep",
        .cuda = "smoothstep",
        .msl = "smoothstep",
    };
};

/// Built-in variable name mappings.
pub const builtinVars = struct {
    pub const global_invocation_id = BuiltinMapping{
        .glsl = "gl_GlobalInvocationID",
        .wgsl = "global_invocation_id",
        .cuda = "globalInvocationId", // Computed
        .msl = "thread_position_in_grid",
    };

    pub const local_invocation_id = BuiltinMapping{
        .glsl = "gl_LocalInvocationID",
        .wgsl = "local_invocation_id",
        .cuda = "threadIdx", // vec3 -> make_uint3
        .msl = "thread_position_in_threadgroup",
    };

    pub const workgroup_id = BuiltinMapping{
        .glsl = "gl_WorkGroupID",
        .wgsl = "workgroup_id",
        .cuda = "blockIdx",
        .msl = "threadgroup_position_in_grid",
    };

    pub const local_invocation_index = BuiltinMapping{
        .glsl = "gl_LocalInvocationIndex",
        .wgsl = "local_invocation_index",
        .cuda = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)",
        .msl = "thread_index_in_threadgroup",
    };

    pub const num_workgroups = BuiltinMapping{
        .glsl = "gl_NumWorkGroups",
        .wgsl = "num_workgroups",
        .cuda = "gridDim",
        .msl = "threadgroups_per_grid",
    };
};

/// Escape a string for use in generated code.
pub fn escapeString(allocator: std.mem.Allocator, s: []const u8) ![]const u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
    for (s) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => try result.append(allocator, c),
        }
    }
    return result.toOwnedSlice(allocator);
}

/// Check if a name is a valid identifier.
pub fn isValidIdentifier(name: []const u8) bool {
    if (name.len == 0) return false;

    // First character must be letter or underscore
    const first = name[0];
    if (!std.ascii.isAlphabetic(first) and first != '_') return false;

    // Rest must be alphanumeric or underscore
    for (name[1..]) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_') return false;
    }

    return true;
}

/// Generate a unique identifier from a base name.
pub fn makeUniqueId(allocator: std.mem.Allocator, base: []const u8, counter: u32) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}_{d}", .{ base, counter });
}

// ============================================================================
// Tests
// ============================================================================

test "CodeWriter basic usage" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var writer = CodeWriter.init(allocator);
    defer writer.deinit();

    try writer.writeLine("// Header");
    try writer.writeLine("void main() {");
    writer.indent();
    try writer.writeLineFmt("int x = {d};", .{42});
    writer.dedent();
    try writer.writeLine("}");

    const code = try writer.getCode();
    try std.testing.expect(std.mem.indexOf(u8, code, "int x = 42;") != null);
}

test "isValidIdentifier" {
    try std.testing.expect(isValidIdentifier("foo"));
    try std.testing.expect(isValidIdentifier("_bar"));
    try std.testing.expect(isValidIdentifier("foo_123"));
    try std.testing.expect(!isValidIdentifier(""));
    try std.testing.expect(!isValidIdentifier("123foo"));
    try std.testing.expect(!isValidIdentifier("foo-bar"));
}

test "escapeString" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const escaped = try escapeString(allocator, "hello\nworld");
    try std.testing.expectEqualStrings("hello\\nworld", escaped);
}
