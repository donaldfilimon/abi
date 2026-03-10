//! SPIR-V Type Code Generation
//!
//! Type generation and encoding logic extracted from generator.zig.
//! Provides functions for creating and caching SPIR-V type declarations.
//! All functions take a pointer to SpirvGenerator as the first argument.

const std = @import("std");
const constants = @import("constants");
const types_gen = @import("types_gen");
const constants_gen = @import("constants_gen");
const instruction_emit = @import("instruction_emit");
const dsl_types = @import("../../types");

pub const OpCode = constants.OpCode;
pub const StorageClass = constants.StorageClass;
pub const TypeKey = types_gen.TypeKey;
pub const ConstKey = constants_gen.ConstKey;

/// The generator type — imported lazily to avoid circular deps.
const SpirvGenerator = @import("generator").SpirvGenerator;

pub fn typeFromIR(self: *SpirvGenerator, ty: dsl_types.Type) !u32 {
    return switch (ty) {
        .scalar => |s| try scalarTypeFromIR(self, s),
        .vector => |v| try getVectorType(self, try scalarTypeFromIR(self, v.element), v.size),
        .array => |a| {
            const elem = try typeFromIR(self, a.element.*);
            if (a.size) |size| {
                return try getArrayType(self, elem, @intCast(size));
            } else {
                return try getRuntimeArrayType(self, elem);
            }
        },
        .ptr => |p| {
            const pointee = try typeFromIR(self, p.pointee.*);
            const storage_class: StorageClass = switch (p.address_space) {
                .private => .Private,
                .workgroup => .Workgroup,
                .storage => .StorageBuffer,
                .uniform => .Uniform,
            };
            return try getPointerType(self, pointee, storage_class);
        },
        .void_ => try getVoidType(self),
        .matrix => |m| {
            const vec = try getVectorType(self, try scalarTypeFromIR(self, m.element), m.rows);
            return try getMatrixType(self, vec, m.cols);
        },
    };
}

pub fn scalarTypeFromIR(self: *SpirvGenerator, s: dsl_types.ScalarType) !u32 {
    return switch (s) {
        .bool_ => try getBoolType(self),
        .i8, .i16, .i32 => try getIntType(self, 32, true),
        .i64 => try getIntType(self, 64, true),
        .u8, .u16, .u32 => try getIntType(self, 32, false),
        .u64 => try getIntType(self, 64, false),
        .f16 => try getFloatType(self, 16),
        .f32 => try getFloatType(self, 32),
        .f64 => try getFloatType(self, 64),
    };
}

pub fn getTypeSize(_: *SpirvGenerator, ty: dsl_types.Type) u32 {
    return switch (ty) {
        .scalar => |s| @as(u32, s.byteSize()),
        .vector => |v| @as(u32, v.element.byteSize()) * @as(u32, v.size),
        else => 4,
    };
}

pub fn getVoidType(self: *SpirvGenerator) !u32 {
    const key = TypeKey{ .tag = .void_, .data = 0, .extra = 0 };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeVoid, &.{id});
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getBoolType(self: *SpirvGenerator) !u32 {
    const key = TypeKey{ .tag = .bool_, .data = 0, .extra = 0 };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeBool, &.{id});
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getIntType(self: *SpirvGenerator, width: u32, signed: bool) !u32 {
    const key = TypeKey{ .tag = .int, .data = width, .extra = if (signed) 1 else 0 };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeInt, &.{ id, width, if (signed) @as(u32, 1) else 0 });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getFloatType(self: *SpirvGenerator, width: u32) !u32 {
    const key = TypeKey{ .tag = .float, .data = width, .extra = 0 };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeFloat, &.{ id, width });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getVectorType(self: *SpirvGenerator, element_type: u32, count: u8) !u32 {
    const key = TypeKey{ .tag = .vector, .data = element_type, .extra = count };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeVector, &.{ id, element_type, count });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getMatrixType(self: *SpirvGenerator, column_type: u32, column_count: u8) !u32 {
    const key = TypeKey{ .tag = .matrix, .data = column_type, .extra = column_count };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeMatrix, &.{ id, column_type, column_count });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getArrayType(self: *SpirvGenerator, element_type: u32, length: u32) !u32 {
    const const_codegen_mod = @import("const_codegen");
    const length_const = try const_codegen_mod.getConstantU32(self, length);
    const key = TypeKey{ .tag = .array, .data = element_type, .extra = length };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeArray, &.{ id, element_type, length_const });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getRuntimeArrayType(self: *SpirvGenerator, element_type: u32) !u32 {
    const key = TypeKey{ .tag = .runtime_array, .data = element_type, .extra = 0 };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeRuntimeArray, &.{ id, element_type });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getPointerType(self: *SpirvGenerator, pointee_type: u32, storage_class: StorageClass) !u32 {
    const key = TypeKey{ .tag = .ptr, .data = pointee_type, .extra = @intFromEnum(storage_class) };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.type_section, .OpTypePointer, &.{ id, @intFromEnum(storage_class), pointee_type });
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

pub fn getFunctionType(self: *SpirvGenerator, return_type: u32, param_types: []const u32) !u32 {
    const key = TypeKey{ .tag = .function, .data = return_type, .extra = @intCast(param_types.len) };
    if (self.type_ids.get(key)) |id| return id;

    const id = self.allocId();
    var operands = std.ArrayListUnmanaged(u32).empty;
    defer operands.deinit(self.allocator);
    try operands.append(self.allocator, id);
    try operands.append(self.allocator, return_type);
    try operands.appendSlice(self.allocator, param_types);
    try instruction_emit.emitOp(self, &self.type_section, .OpTypeFunction, operands.items);
    try self.type_ids.put(self.allocator, key, id);
    return id;
}

test {
    std.testing.refAllDecls(@This());
}
