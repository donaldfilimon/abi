//! SPIR-V Instruction Emission
//!
//! Low-level methods for emitting SPIR-V instructions to output sections.

const std = @import("std");
const constants = @import("constants.zig");

pub const OpCode = constants.OpCode;
pub const Capability = constants.Capability;
pub const AddressingModel = constants.AddressingModel;
pub const MemoryModel = constants.MemoryModel;
pub const ExecutionModel = constants.ExecutionModel;
pub const ExecutionMode = constants.ExecutionMode;
pub const StorageClass = constants.StorageClass;
pub const Decoration = constants.Decoration;

/// Instruction emission mixin for SpirvGenerator.
/// Provides low-level methods for emitting SPIR-V instructions.
pub fn EmitMixin(comptime Self: type) type {
    return struct {
        pub fn emitOp(self: *Self, section: *std.ArrayListUnmanaged(u32), opcode: OpCode, operands: []const u32) !void {
            const word_count: u32 = @intCast(1 + operands.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(opcode));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands);
        }

        pub fn emitCapability(self: *Self, cap: Capability) !void {
            try self.emitOp(&self.words, .OpCapability, &.{@intFromEnum(cap)});
        }

        pub fn emitExtInstImport(self: *Self, result_id: u32, name_str: []const u8) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, result_id);
            try self.appendString(&operands, name_str);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpExtInstImport));
            try self.words.append(self.allocator, first_word);
            try self.words.appendSlice(self.allocator, operands.items);
        }

        pub fn emitMemoryModel(self: *Self, addressing: AddressingModel, memory: MemoryModel) !void {
            try self.emitOp(&self.words, .OpMemoryModel, &.{ @intFromEnum(addressing), @intFromEnum(memory) });
        }

        pub fn emitEntryPoint(self: *Self, model: ExecutionModel, func_id: u32, name_str: []const u8, interface: []const u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, @intFromEnum(model));
            try operands.append(self.allocator, func_id);
            try self.appendString(&operands, name_str);
            try operands.appendSlice(self.allocator, interface);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpEntryPoint));
            try self.entry_section.append(self.allocator, first_word);
            try self.entry_section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitExecutionMode(self: *Self, func_id: u32, mode: ExecutionMode, params: []const u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, func_id);
            try operands.append(self.allocator, @intFromEnum(mode));
            try operands.appendSlice(self.allocator, params);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpExecutionMode));
            try self.entry_section.append(self.allocator, first_word);
            try self.entry_section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitName(self: *Self, section: *std.ArrayListUnmanaged(u32), id: u32, name_str: []const u8) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, id);
            try self.appendString(&operands, name_str);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpName));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitMemberName(self: *Self, section: *std.ArrayListUnmanaged(u32), type_id: u32, member: u32, name_str: []const u8) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, type_id);
            try operands.append(self.allocator, member);
            try self.appendString(&operands, name_str);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpMemberName));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitDecorate(self: *Self, section: *std.ArrayListUnmanaged(u32), target: u32, decoration: Decoration, params: []const u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, target);
            try operands.append(self.allocator, @intFromEnum(decoration));
            try operands.appendSlice(self.allocator, params);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpDecorate));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitMemberDecorate(self: *Self, section: *std.ArrayListUnmanaged(u32), struct_type: u32, member: u32, decoration: Decoration, params: []const u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, struct_type);
            try operands.append(self.allocator, member);
            try operands.append(self.allocator, @intFromEnum(decoration));
            try operands.appendSlice(self.allocator, params);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpMemberDecorate));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitTypeStruct(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, member_types: []const u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, result_id);
            try operands.appendSlice(self.allocator, member_types);

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpTypeStruct));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitVariable(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, type_id: u32, storage_class: StorageClass, initializer: ?u32) !void {
            var operands = std.ArrayListUnmanaged(u32).empty;
            defer operands.deinit(self.allocator);
            try operands.append(self.allocator, type_id);
            try operands.append(self.allocator, result_id);
            try operands.append(self.allocator, @intFromEnum(storage_class));
            if (initializer) |init_id| {
                try operands.append(self.allocator, init_id);
            }

            const word_count: u32 = @intCast(1 + operands.items.len);
            const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpVariable));
            try section.append(self.allocator, first_word);
            try section.appendSlice(self.allocator, operands.items);
        }

        pub fn emitFunction(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, return_type: u32, control: anytype, func_type: u32) !void {
            _ = control;
            try self.emitOp(section, .OpFunction, &.{ return_type, result_id, 0, func_type });
        }

        pub fn emitFunctionEnd(self: *Self, section: *std.ArrayListUnmanaged(u32)) !void {
            try self.emitOp(section, .OpFunctionEnd, &.{});
        }

        pub fn emitLabel(self: *Self, section: *std.ArrayListUnmanaged(u32), id: u32) !void {
            try self.emitOp(section, .OpLabel, &.{id});
        }

        pub fn emitReturn(self: *Self, section: *std.ArrayListUnmanaged(u32)) !void {
            try self.emitOp(section, .OpReturn, &.{});
        }

        pub fn emitBranch(self: *Self, section: *std.ArrayListUnmanaged(u32), target: u32) !void {
            try self.emitOp(section, .OpBranch, &.{target});
        }

        pub fn emitBranchConditional(self: *Self, section: *std.ArrayListUnmanaged(u32), condition: u32, true_label: u32, false_label: u32) !void {
            try self.emitOp(section, .OpBranchConditional, &.{ condition, true_label, false_label });
        }

        pub fn emitSelectionMerge(self: *Self, section: *std.ArrayListUnmanaged(u32), merge_label: u32) !void {
            try self.emitOp(section, .OpSelectionMerge, &.{ merge_label, 0 }); // 0 = None
        }

        pub fn emitLoopMerge(self: *Self, section: *std.ArrayListUnmanaged(u32), merge_label: u32, continue_label: u32) !void {
            try self.emitOp(section, .OpLoopMerge, &.{ merge_label, continue_label, 0 }); // 0 = None
        }

        pub fn emitStore(self: *Self, section: *std.ArrayListUnmanaged(u32), ptr: u32, value: u32) !void {
            try self.emitOp(section, .OpStore, &.{ ptr, value });
        }

        pub fn appendString(self: *Self, operands: *std.ArrayListUnmanaged(u32), str: []const u8) !void {
            var word: u32 = 0;
            var byte_idx: usize = 0;

            for (str) |c| {
                word |= @as(u32, c) << @intCast(byte_idx * 8);
                byte_idx += 1;
                if (byte_idx == 4) {
                    try operands.append(self.allocator, word);
                    word = 0;
                    byte_idx = 0;
                }
            }
            // Append null terminator
            try operands.append(self.allocator, word); // Includes null terminator in remaining bytes
        }
    };
}
