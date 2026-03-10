//! SPIR-V Constant Code Generation
//!
//! Constant generation logic extracted from generator.zig.
//! Provides methods for creating and caching SPIR-V constants.

const std = @import("std");
const constants = @import("constants.zig");
const constants_gen = @import("constants_gen.zig");

pub const OpCode = constants.OpCode;
pub const ConstKey = constants_gen.ConstKey;

/// Constant code generation mixin for SpirvGenerator.
/// Provides methods for creating and caching SPIR-V constants.
pub fn ConstCodeGenMixin(comptime Self: type) type {
    return struct {
        pub fn getConstantTrue(self: *Self) !u32 {
            const bool_type = try self.getBoolType();
            const key = ConstKey{ .type_id = bool_type, .value = 1 };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstantTrue, &.{ bool_type, id });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantFalse(self: *Self) !u32 {
            const bool_type = try self.getBoolType();
            const key = ConstKey{ .type_id = bool_type, .value = 0 };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstantFalse, &.{ bool_type, id });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantI32(self: *Self, value: i32) !u32 {
            const int_type = try self.getIntType(32, true);
            const key = ConstKey{ .type_id = int_type, .value = @bitCast(@as(u64, @bitCast(@as(i64, value)))) };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @bitCast(value)) });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantU32(self: *Self, value: u32) !u32 {
            const int_type = try self.getIntType(32, false);
            const key = ConstKey{ .type_id = int_type, .value = value };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, value });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantI64(self: *Self, value: i64) !u32 {
            const int_type = try self.getIntType(64, true);
            const bits: u64 = @bitCast(value);
            const key = ConstKey{ .type_id = int_type, .value = bits };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantU64(self: *Self, value: u64) !u32 {
            const int_type = try self.getIntType(64, false);
            const key = ConstKey{ .type_id = int_type, .value = value };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ int_type, id, @as(u32, @truncate(value)), @as(u32, @truncate(value >> 32)) });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantF32(self: *Self, value: f32) !u32 {
            const float_type = try self.getFloatType(32);
            const bits: u32 = @bitCast(value);
            const key = ConstKey{ .type_id = float_type, .value = bits };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ float_type, id, bits });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }

        pub fn getConstantF64(self: *Self, value: f64) !u32 {
            const float_type = try self.getFloatType(64);
            const bits: u64 = @bitCast(value);
            const key = ConstKey{ .type_id = float_type, .value = bits };
            if (self.const_ids.get(key)) |id| return id;

            const id = self.allocId();
            try self.emitOp(&self.const_section, .OpConstant, &.{ float_type, id, @as(u32, @truncate(bits)), @as(u32, @truncate(bits >> 32)) });
            try self.const_ids.put(self.allocator, key, id);
            return id;
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
