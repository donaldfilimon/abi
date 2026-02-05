//! GPU Kernel DSL Type System
//!
//! Defines the type system for the portable kernel DSL, including scalar types,
//! vector types, matrix types, and address spaces for GPU memory.

const std = @import("std");

/// Scalar types supported by the kernel IR.
/// These map to native GPU types across all backends.
pub const ScalarType = enum {
    bool_,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,

    /// Returns the size in bytes of this scalar type.
    pub fn byteSize(self: ScalarType) u8 {
        return switch (self) {
            .bool_ => 1,
            .i8, .u8 => 1,
            .i16, .u16, .f16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }

    /// Returns true if this is a floating-point type.
    pub fn isFloat(self: ScalarType) bool {
        return switch (self) {
            .f16, .f32, .f64 => true,
            else => false,
        };
    }

    /// Returns true if this is a signed integer type.
    pub fn isSigned(self: ScalarType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    /// Returns true if this is an unsigned integer type.
    pub fn isUnsigned(self: ScalarType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    /// Returns true if this is any integer type.
    pub fn isInteger(self: ScalarType) bool {
        return self.isSigned() or self.isUnsigned();
    }

    /// Returns the name of this type for display/debugging.
    pub fn name(self: ScalarType) []const u8 {
        return switch (self) {
            .bool_ => "bool",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .f16 => "f16",
            .f32 => "f32",
            .f64 => "f64",
        };
    }
};

/// Vector types (vec2, vec3, vec4).
/// Standard GPU vector types used for SIMD operations.
pub const VectorType = struct {
    element: ScalarType,
    size: u8, // 2, 3, or 4

    pub fn init(element: ScalarType, size: u8) VectorType {
        std.debug.assert(size >= 2 and size <= 4);
        return .{ .element = element, .size = size };
    }

    /// Returns the size in bytes of this vector type.
    pub fn byteSize(self: VectorType) usize {
        return @as(usize, self.element.byteSize()) * @as(usize, self.size);
    }

    /// Common vector type constructors
    pub fn vec2(element: ScalarType) VectorType {
        return init(element, 2);
    }

    pub fn vec3(element: ScalarType) VectorType {
        return init(element, 3);
    }

    pub fn vec4(element: ScalarType) VectorType {
        return init(element, 4);
    }

    /// Returns the name of this type for display/debugging.
    pub fn name(self: VectorType) []const u8 {
        const base = self.element.name();
        return switch (self.size) {
            2 => switch (self.element) {
                .f32 => "vec2",
                .i32 => "ivec2",
                .u32 => "uvec2",
                else => "vec2",
            },
            3 => switch (self.element) {
                .f32 => "vec3",
                .i32 => "ivec3",
                .u32 => "uvec3",
                else => "vec3",
            },
            4 => switch (self.element) {
                .f32 => "vec4",
                .i32 => "ivec4",
                .u32 => "uvec4",
                else => "vec4",
            },
            else => base,
        };
    }
};

/// Matrix types (mat2x2, mat3x3, mat4x4, etc.).
/// Used for linear algebra operations on GPU.
pub const MatrixType = struct {
    element: ScalarType,
    rows: u8,
    cols: u8,

    pub fn init(element: ScalarType, rows: u8, cols: u8) MatrixType {
        std.debug.assert(rows >= 2 and rows <= 4);
        std.debug.assert(cols >= 2 and cols <= 4);
        return .{ .element = element, .rows = rows, .cols = cols };
    }

    /// Returns the size in bytes of this matrix type.
    pub fn byteSize(self: MatrixType) usize {
        return @as(usize, self.element.byteSize()) * @as(usize, self.rows) * @as(usize, self.cols);
    }

    /// Common matrix type constructors
    pub fn mat2(element: ScalarType) MatrixType {
        return init(element, 2, 2);
    }

    pub fn mat3(element: ScalarType) MatrixType {
        return init(element, 3, 3);
    }

    pub fn mat4(element: ScalarType) MatrixType {
        return init(element, 4, 4);
    }

    /// Returns true if this is a square matrix.
    pub fn isSquare(self: MatrixType) bool {
        return self.rows == self.cols;
    }
};

/// GPU address spaces.
/// Different memory regions with different performance characteristics.
pub const AddressSpace = enum {
    /// Per-thread local memory (registers/private).
    private,
    /// Shared memory within a workgroup (fast, limited size).
    workgroup,
    /// Device memory (global storage buffers, read-write).
    storage,
    /// Constant/uniform buffers (read-only, cached).
    uniform,

    /// Returns the name of this address space for display/debugging.
    pub fn name(self: AddressSpace) []const u8 {
        return switch (self) {
            .private => "private",
            .workgroup => "workgroup",
            .storage => "storage",
            .uniform => "uniform",
        };
    }
};

/// Array type for fixed or runtime-sized arrays.
pub const ArrayType = struct {
    element: *const Type,
    /// Size of the array. null means runtime-sized (e.g., storage buffer).
    size: ?usize,

    pub fn init(element: *const Type, size: ?usize) ArrayType {
        return .{ .element = element, .size = size };
    }

    /// Returns true if this is a runtime-sized array.
    pub fn isRuntimeSized(self: ArrayType) bool {
        return self.size == null;
    }
};

/// Pointer type with address space qualification.
pub const PtrType = struct {
    pointee: *const Type,
    address_space: AddressSpace,

    pub fn init(pointee: *const Type, address_space: AddressSpace) PtrType {
        return .{ .pointee = pointee, .address_space = address_space };
    }
};

/// Complete type representation.
/// Union of all possible types in the DSL.
pub const Type = union(enum) {
    scalar: ScalarType,
    vector: VectorType,
    matrix: MatrixType,
    array: ArrayType,
    ptr: PtrType,
    void_: void,

    /// Returns true if this type can be used in arithmetic operations.
    pub fn isNumeric(self: Type) bool {
        return switch (self) {
            .scalar => |s| s != .bool_,
            .vector => true,
            .matrix => true,
            else => false,
        };
    }

    /// Returns the scalar element type if this is a compound type.
    pub fn elementType(self: Type) ?ScalarType {
        return switch (self) {
            .scalar => |s| s,
            .vector => |v| v.element,
            .matrix => |m| m.element,
            .array => |a| a.element.elementType(),
            .ptr => |p| p.pointee.elementType(),
            .void_ => null,
        };
    }

    /// Convenience constructors for common types
    pub fn f32Type() Type {
        return .{ .scalar = .f32 };
    }

    pub fn i32Type() Type {
        return .{ .scalar = .i32 };
    }

    pub fn u32Type() Type {
        return .{ .scalar = .u32 };
    }

    pub fn boolType() Type {
        return .{ .scalar = .bool_ };
    }

    pub fn vec2f32() Type {
        return .{ .vector = VectorType.vec2(.f32) };
    }

    pub fn vec3f32() Type {
        return .{ .vector = VectorType.vec3(.f32) };
    }

    pub fn vec4f32() Type {
        return .{ .vector = VectorType.vec4(.f32) };
    }

    pub fn vec3u32() Type {
        return .{ .vector = VectorType.vec3(.u32) };
    }

    pub fn voidType() Type {
        return .{ .void_ = {} };
    }
};

/// Parameter access mode for buffer bindings.
pub const AccessMode = enum {
    read_only,
    write_only,
    read_write,

    pub fn name(self: AccessMode) []const u8 {
        return switch (self) {
            .read_only => "read",
            .write_only => "write",
            .read_write => "read_write",
        };
    }

    pub fn canRead(self: AccessMode) bool {
        return self == .read_only or self == .read_write;
    }

    pub fn canWrite(self: AccessMode) bool {
        return self == .write_only or self == .read_write;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ScalarType.byteSize" {
    try std.testing.expectEqual(@as(u8, 1), ScalarType.bool_.byteSize());
    try std.testing.expectEqual(@as(u8, 4), ScalarType.f32.byteSize());
    try std.testing.expectEqual(@as(u8, 8), ScalarType.f64.byteSize());
    try std.testing.expectEqual(@as(u8, 4), ScalarType.i32.byteSize());
}

test "ScalarType.isFloat" {
    try std.testing.expect(ScalarType.f32.isFloat());
    try std.testing.expect(ScalarType.f64.isFloat());
    try std.testing.expect(!ScalarType.i32.isFloat());
    try std.testing.expect(!ScalarType.bool_.isFloat());
}

test "VectorType.byteSize" {
    const vec3f = VectorType.vec3(.f32);
    try std.testing.expectEqual(@as(usize, 12), vec3f.byteSize());

    const vec4i = VectorType.vec4(.i32);
    try std.testing.expectEqual(@as(usize, 16), vec4i.byteSize());
}

test "Type.isNumeric" {
    try std.testing.expect(Type.f32Type().isNumeric());
    try std.testing.expect(Type.i32Type().isNumeric());
    try std.testing.expect(Type.vec3f32().isNumeric());
    try std.testing.expect(!Type.boolType().isNumeric());
    try std.testing.expect(!Type.voidType().isNumeric());
}

test "AccessMode.canRead" {
    try std.testing.expect(AccessMode.read_only.canRead());
    try std.testing.expect(AccessMode.read_write.canRead());
    try std.testing.expect(!AccessMode.write_only.canRead());
}
