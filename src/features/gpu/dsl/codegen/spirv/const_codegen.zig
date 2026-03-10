//! SPIR-V Constant Code Generation
//!
//! Constant generation logic extracted from generator.zig.
//! Provides functions for creating and caching SPIR-V constants.
//! All functions take a pointer to SpirvGenerator as the first argument.

const std = @import("std");
const constants = @import("constants");
const constants_gen = @import("constants_gen");

pub const OpCode = constants.OpCode;
pub const ConstKey = constants_gen.ConstKey;

/// The generator type — imported lazily to avoid circular deps.
const SpirvGenerator = @import("generator").SpirvGenerator;

const instruction_emit = @import("instruction_emit");
const type_codegen = @import("type_codegen");

pub fn getConstantTrue(self: *SpirvGenerator) !u32 {
    const bool_type = try type_codegen.getBoolType(self);
    const key = ConstKey{ .type_id = bool_type, .value = 1 };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstantTrue, &.{ bool_type, id });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantFalse(self: *SpirvGenerator) !u32 {
    const bool_type = try type_codegen.getBoolType(self);
    const key = ConstKey{ .type_id = bool_type, .value = 0 };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstantFalse, &.{ bool_type, id });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantI32(self: *SpirvGenerator, value: i32) !u32 {
    const int_type = try type_codegen.getIntType(self, 32, true);
    const key = ConstKey{ .type_id = int_type, .value = @bitCast(@as(u64, @bitCast(@as(i64, value)))) };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @bitCast(value)) });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantU32(self: *SpirvGenerator, value: u32) !u32 {
    const int_type = try type_codegen.getIntType(self, 32, false);
    const key = ConstKey{ .type_id = int_type, .value = value };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ int_type, id, value });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantI64(self: *SpirvGenerator, value: i64) !u32 {
    const int_type = try type_codegen.getIntType(self, 64, true);
    const bits: u64 = @bitCast(value);
    const key = ConstKey{ .type_id = int_type, .value = bits };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantU64(self: *SpirvGenerator, value: u64) !u32 {
    const int_type = try type_codegen.getIntType(self, 64, false);
    const key = ConstKey{ .type_id = int_type, .value = value };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(value)), @as(u32, @truncate(value >> 32)) });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantF32(self: *SpirvGenerator, value: f32) !u32 {
    const float_type = try type_codegen.getFloatType(self, 32);
    const bits: u32 = @bitCast(value);
    const key = ConstKey{ .type_id = float_type, .value = bits };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ float_type, id, bits });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

pub fn getConstantF64(self: *SpirvGenerator, value: f64) !u32 {
    const float_type = try type_codegen.getFloatType(self, 64);
    const bits: u64 = @bitCast(value);
    const key = ConstKey{ .type_id = float_type, .value = bits };
    if (self.const_ids.get(key)) |id| return id;

    const id = self.allocId();
    try instruction_emit.emitOp(self, &self.const_section, .OpConstant, &.{ float_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
    try self.const_ids.put(self.allocator, key, id);
    return id;
}

test {
    std.testing.refAllDecls(@This());
}
