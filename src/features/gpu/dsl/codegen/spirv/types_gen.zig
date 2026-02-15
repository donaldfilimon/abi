//! SPIR-V Type Generation
//!
//! Methods for generating SPIR-V type declarations with caching.

const std = @import("std");
const constants = @import("constants.zig");
const dsl_types = @import("../../types.zig");

pub const OpCode = constants.OpCode;
pub const StorageClass = constants.StorageClass;

/// Key for type caching.
pub const TypeKey = struct {
    tag: enum { void_, bool_, int, float, vector, matrix, array, runtime_array, ptr, struct_, function },
    data: u64,
    extra: u32,
};

/// Type generation mixin for SpirvGenerator.
/// Provides methods for creating and caching SPIR-V types.
pub fn TypeGenMixin(comptime Self: type) type {
    return struct {
        pub fn typeFromIR(self: *Self, ty: dsl_types.Type) !u32 {
            return switch (ty) {
                .scalar => |s| try self.scalarTypeFromIR(s),
                .vector => |v| try self.getVectorType(try self.scalarTypeFromIR(v.element), v.size),
                .array => |a| {
                    const elem = try self.typeFromIR(a.element.*);
                    if (a.size) |size| {
                        return try self.getArrayType(elem, @intCast(size));
                    } else {
                        return try self.getRuntimeArrayType(elem);
                    }
                },
                .ptr => |p| {
                    const pointee = try self.typeFromIR(p.pointee.*);
                    const storage_class: StorageClass = switch (p.address_space) {
                        .private => .Private,
                        .workgroup => .Workgroup,
                        .storage => .StorageBuffer,
                        .uniform => .Uniform,
                    };
                    return try self.getPointerType(pointee, storage_class);
                },
                .void_ => try self.getVoidType(),
                .matrix => |m| {
                    const vec = try self.getVectorType(try self.scalarTypeFromIR(m.element), m.rows);
                    return try self.getMatrixType(vec, m.cols);
                },
            };
        }

        pub fn scalarTypeFromIR(self: *Self, s: dsl_types.ScalarType) !u32 {
            return switch (s) {
                .bool_ => try self.getBoolType(),
                .i8, .i16, .i32 => try self.getIntType(32, true),
                .i64 => try self.getIntType(64, true),
                .u8, .u16, .u32 => try self.getIntType(32, false),
                .u64 => try self.getIntType(64, false),
                .f16 => try self.getFloatType(16),
                .f32 => try self.getFloatType(32),
                .f64 => try self.getFloatType(64),
            };
        }

        pub fn getTypeSize(self: *Self, ty: dsl_types.Type) u32 {
            _ = self;
            return switch (ty) {
                .scalar => |s| @as(u32, s.byteSize()),
                .vector => |v| @as(u32, v.element.byteSize()) * @as(u32, v.size),
                else => 4,
            };
        }

        pub fn getVoidType(self: *Self) !u32 {
            const key = TypeKey{ .tag = .void_, .data = 0, .extra = 0 };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeVoid, &.{id});
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getBoolType(self: *Self) !u32 {
            const key = TypeKey{ .tag = .bool_, .data = 0, .extra = 0 };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeBool, &.{id});
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getIntType(self: *Self, width: u32, signed: bool) !u32 {
            const key = TypeKey{ .tag = .int, .data = width, .extra = if (signed) 1 else 0 };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeInt, &.{ id, width, if (signed) @as(u32, 1) else 0 });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getFloatType(self: *Self, width: u32) !u32 {
            const key = TypeKey{ .tag = .float, .data = width, .extra = 0 };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeFloat, &.{ id, width });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getVectorType(self: *Self, element_type: u32, count: u8) !u32 {
            const key = TypeKey{ .tag = .vector, .data = element_type, .extra = count };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeVector, &.{ id, element_type, count });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getMatrixType(self: *Self, column_type: u32, column_count: u8) !u32 {
            const key = TypeKey{ .tag = .matrix, .data = column_type, .extra = column_count };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeMatrix, &.{ id, column_type, column_count });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getArrayType(self: *Self, element_type: u32, length: u32) !u32 {
            const length_const = try self.getConstantU32(length);
            const key = TypeKey{ .tag = .array, .data = element_type, .extra = length };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeArray, &.{ id, element_type, length_const });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getRuntimeArrayType(self: *Self, element_type: u32) !u32 {
            const key = TypeKey{ .tag = .runtime_array, .data = element_type, .extra = 0 };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypeRuntimeArray, &.{ id, element_type });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getPointerType(self: *Self, pointee_type: u32, storage_class: StorageClass) !u32 {
            const key = TypeKey{ .tag = .ptr, .data = pointee_type, .extra = @intFromEnum(storage_class) };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.type_section, .OpTypePointer, &.{ id, @intFromEnum(storage_class), pointee_type });
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getFunctionType(self: *Self, return_type: u32, param_types: []const u32) !u32 {
            const key = TypeKey{ .tag = .function, .data = return_type, .extra = @intCast(param_types.len) };
            if (self.type_ids.get(key)) |id| return id;

            const id = self.allocId();
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, id);
            try operands.append(self.allocator, return_type);
            try operands.appendSlice(self.allocator, param_types);
            try self.emitOp(&self.type_section, .OpTypeFunction, operands.items);
            try self.type_ids.put(self.allocator, key, id);
            return id;
        }
    };
}
