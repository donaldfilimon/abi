//! SPIR-V Code Generation
//!
//! Methods for generating SPIR-V code from IR statements and expressions.

const std = @import("std");
const constants = @import("constants.zig");
const expr = @import("../../expr.zig");
const stmt = @import("../../stmt.zig");

pub const OpCode = constants.OpCode;
pub const Scope = constants.Scope;
pub const MemorySemantics = constants.MemorySemantics;
pub const StorageClass = constants.StorageClass;

/// Code generation mixin for SpirvGenerator.
/// Provides methods for generating SPIR-V code from IR.
pub fn CodeGenMixin(comptime Self: type) type {
    return struct {
        /// Generate code for a statement.
        pub fn generateStmt(self: *Self, s: *const stmt.Stmt) !void {
            switch (s.*) {
                .var_decl => |v| {
                    const ty = try self.typeFromIR(v.ty);
                    const ptr_type = try self.getPointerType(ty, .Function);
                    const var_id = self.allocId();

                    // Initialize variable
                    const init_id: ?u32 = if (v.init) |init_expr|
                        try self.generateExpr(init_expr)
                    else
                        null;

                    try self.emitVariable(&self.function_section, var_id, ptr_type, .Function, init_id);
                    try self.var_ids.put(self.allocator, v.name, var_id);
                },
                .assign => |a| {
                    const value_id = try self.generateExpr(a.value);
                    const target_id = try self.generateExprPtr(a.target);
                    try self.emitStore(&self.function_section, target_id, value_id);
                },
                .compound_assign => |ca| {
                    const target_ptr = try self.generateExprPtr(ca.target);
                    const current_val = try self.generateLoad(target_ptr);
                    const operand_val = try self.generateExpr(ca.value);
                    const result = try self.generateBinaryOp(ca.op, current_val, operand_val);
                    try self.emitStore(&self.function_section, target_ptr, result);
                },
                .if_ => |i| {
                    const cond_id = try self.generateExpr(i.condition);
                    const then_label = self.allocId();
                    const else_label = self.allocId();
                    const merge_label = self.allocId();

                    try self.emitSelectionMerge(&self.function_section, merge_label);
                    try self.emitBranchConditional(&self.function_section, cond_id, then_label, if (i.else_body != null) else_label else merge_label);

                    // Then block
                    try self.emitLabel(&self.function_section, then_label);
                    for (i.then_body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try self.emitBranch(&self.function_section, merge_label);

                    // Else block
                    if (i.else_body) |else_body| {
                        try self.emitLabel(&self.function_section, else_label);
                        for (else_body) |body_stmt| {
                            try self.generateStmt(body_stmt);
                        }
                        try self.emitBranch(&self.function_section, merge_label);
                    }

                    // Merge block
                    try self.emitLabel(&self.function_section, merge_label);
                },
                .for_ => |f| {
                    // Initialize
                    if (f.init) |init_stmt| {
                        try self.generateStmt(init_stmt);
                    }

                    const header_label = self.allocId();
                    const body_label = self.allocId();
                    const continue_label = self.allocId();
                    const merge_label = self.allocId();

                    try self.emitBranch(&self.function_section, header_label);
                    try self.emitLabel(&self.function_section, header_label);
                    try self.emitLoopMerge(&self.function_section, merge_label, continue_label);

                    if (f.condition) |cond| {
                        const cond_id = try self.generateExpr(cond);
                        try self.emitBranchConditional(&self.function_section, cond_id, body_label, merge_label);
                    } else {
                        try self.emitBranch(&self.function_section, body_label);
                    }

                    // Body
                    try self.emitLabel(&self.function_section, body_label);
                    for (f.body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try self.emitBranch(&self.function_section, continue_label);

                    // Continue block
                    try self.emitLabel(&self.function_section, continue_label);
                    if (f.update) |update| {
                        try self.generateStmt(update);
                    }
                    try self.emitBranch(&self.function_section, header_label);

                    // Merge
                    try self.emitLabel(&self.function_section, merge_label);
                },
                .while_ => |w| {
                    const header_label = self.allocId();
                    const body_label = self.allocId();
                    const merge_label = self.allocId();

                    try self.emitBranch(&self.function_section, header_label);
                    try self.emitLabel(&self.function_section, header_label);
                    try self.emitLoopMerge(&self.function_section, merge_label, header_label);

                    const cond_id = try self.generateExpr(w.condition);
                    try self.emitBranchConditional(&self.function_section, cond_id, body_label, merge_label);

                    try self.emitLabel(&self.function_section, body_label);
                    for (w.body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try self.emitBranch(&self.function_section, header_label);

                    try self.emitLabel(&self.function_section, merge_label);
                },
                .return_ => {
                    try self.emitReturn(&self.function_section);
                },
                .break_ => {
                    // Would need to track current loop merge label
                },
                .continue_ => {
                    // Would need to track current loop continue label
                },
                .expr_stmt => |e| {
                    _ = try self.generateExpr(e);
                },
                .block => |b| {
                    for (b.statements) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                },
                else => {},
            }
        }

        /// Generate an expression and return its result ID.
        pub fn generateExpr(self: *Self, e: *const expr.Expr) !u32 {
            return switch (e.*) {
                .literal => |lit| try self.generateLiteral(lit),
                .ref => |ref| {
                    if (ref.name) |name| {
                        if (self.var_ids.get(name)) |var_id| {
                            return try self.generateLoad(var_id);
                        }
                    }
                    return error.InvalidIR;
                },
                .unary => |un| {
                    const operand_id = try self.generateExpr(un.operand);
                    return try self.generateUnaryOp(un.op, operand_id);
                },
                .binary => |bin| {
                    const left_id = try self.generateExpr(bin.left);
                    const right_id = try self.generateExpr(bin.right);
                    return try self.generateBinaryOp(bin.op, left_id, right_id);
                },
                .call => |c| try self.generateCall(c),
                .index => |idx| {
                    const base_ptr = try self.generateExprPtr(idx.base);
                    const index_id = try self.generateExpr(idx.index);
                    const elem_ptr = try self.generateAccessChain(base_ptr, index_id);
                    return try self.generateLoad(elem_ptr);
                },
                .field => |f| {
                    const base_ptr = try self.generateExprPtr(f.base);
                    // Field index would need type info
                    const zero = try self.getConstantU32(0);
                    const field_ptr = try self.generateAccessChain(base_ptr, zero);
                    return try self.generateLoad(field_ptr);
                },
                .cast => |c| {
                    const operand_id = try self.generateExpr(c.operand);
                    const target_type = try self.typeFromIR(c.target_type);
                    return try self.generateCast(operand_id, target_type);
                },
                .select => |s| {
                    const cond_id = try self.generateExpr(s.condition);
                    const true_id = try self.generateExpr(s.true_value);
                    const false_id = try self.generateExpr(s.false_value);
                    const result_id = self.allocId();
                    const float_type = try self.getFloatType(32);
                    try self.emitOp(&self.function_section, .OpSelect, &.{ float_type, result_id, cond_id, true_id, false_id });
                    return result_id;
                },
                .vector_construct => |vc| {
                    var component_ids = std.ArrayListUnmanaged(u32){};
                    defer component_ids.deinit(self.allocator);

                    for (vc.components) |comp| {
                        try component_ids.append(self.allocator, try self.generateExpr(comp));
                    }

                    const elem_type = try self.scalarTypeFromIR(vc.element_type);
                    const vec_type = try self.getVectorType(elem_type, vc.size);
                    const result_id = self.allocId();

                    var operands = std.ArrayListUnmanaged(u32){};
                    defer operands.deinit(self.allocator);
                    try operands.append(self.allocator, vec_type);
                    try operands.append(self.allocator, result_id);
                    try operands.appendSlice(self.allocator, component_ids.items);

                    try self.emitOp(&self.function_section, .OpCompositeConstruct, operands.items);
                    return result_id;
                },
                .swizzle => |sw| {
                    const base_id = try self.generateExpr(sw.base);
                    if (sw.components.len == 1) {
                        // Extract single component
                        const result_id = self.allocId();
                        const float_type = try self.getFloatType(32);
                        try self.emitOp(&self.function_section, .OpCompositeExtract, &.{ float_type, result_id, base_id, sw.components[0] });
                        return result_id;
                    } else {
                        // Vector shuffle
                        const elem_type = try self.getFloatType(32);
                        const vec_type = try self.getVectorType(elem_type, @intCast(sw.components.len));
                        const result_id = self.allocId();

                        var operands = std.ArrayListUnmanaged(u32){};
                        defer operands.deinit(self.allocator);
                        try operands.append(self.allocator, vec_type);
                        try operands.append(self.allocator, result_id);
                        try operands.append(self.allocator, base_id);
                        try operands.append(self.allocator, base_id);
                        for (sw.components) |c| {
                            try operands.append(self.allocator, c);
                        }

                        try self.emitOp(&self.function_section, .OpVectorShuffle, operands.items);
                        return result_id;
                    }
                },
            };
        }

        /// Generate expression as pointer (for assignments).
        pub fn generateExprPtr(self: *Self, e: *const expr.Expr) !u32 {
            return switch (e.*) {
                .ref => |ref| {
                    if (ref.name) |name| {
                        if (self.var_ids.get(name)) |var_id| {
                            return var_id;
                        }
                    }
                    return error.InvalidIR;
                },
                .index => |idx| {
                    const base_ptr = try self.generateExprPtr(idx.base);
                    const index_id = try self.generateExpr(idx.index);
                    return try self.generateAccessChain(base_ptr, index_id);
                },
                .field => |f| {
                    const base_ptr = try self.generateExprPtr(f.base);
                    const zero = try self.getConstantU32(0);
                    return try self.generateAccessChain(base_ptr, zero);
                },
                else => error.InvalidIR,
            };
        }

        /// Generate a literal constant.
        pub fn generateLiteral(self: *Self, lit: expr.Literal) !u32 {
            return switch (lit) {
                .bool_ => |v| if (v) try self.getConstantTrue() else try self.getConstantFalse(),
                .i32_ => |v| try self.getConstantI32(v),
                .i64_ => |v| try self.getConstantI64(v),
                .u32_ => |v| try self.getConstantU32(v),
                .u64_ => |v| try self.getConstantU64(v),
                .f32_ => |v| try self.getConstantF32(v),
                .f64_ => |v| try self.getConstantF64(v),
            };
        }

        /// Generate a unary operation.
        pub fn generateUnaryOp(self: *Self, op: expr.UnaryOp, operand: u32) !u32 {
            const result_id = self.allocId();
            const float_type = try self.getFloatType(32);

            switch (op) {
                .neg => {
                    try self.emitOp(&self.function_section, .OpFNegate, &.{ float_type, result_id, operand });
                },
                .not => {
                    const bool_type = try self.getBoolType();
                    try self.emitOp(&self.function_section, .OpLogicalNot, &.{ bool_type, result_id, operand });
                },
                .bit_not => {
                    const int_type = try self.getIntType(32, true);
                    try self.emitOp(&self.function_section, .OpNot, &.{ int_type, result_id, operand });
                },
                .sqrt, .sin, .cos, .tan, .exp, .log, .abs, .floor, .ceil, .round => {
                    const ext_inst = switch (op) {
                        .sqrt => 31, // Sqrt
                        .sin => 13, // Sin
                        .cos => 14, // Cos
                        .tan => 15, // Tan
                        .exp => 27, // Exp
                        .log => 28, // Log
                        .abs => 4, // FAbs
                        .floor => 8, // Floor
                        .ceil => 9, // Ceil
                        .round => 1, // Round
                        else => 0,
                    };
                    try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, ext_inst, operand });
                },
                else => {
                    // Fallback
                    try self.emitOp(&self.function_section, .OpCopyObject, &.{ float_type, result_id, operand });
                },
            }
            return result_id;
        }

        /// Generate a binary operation.
        pub fn generateBinaryOp(self: *Self, op: expr.BinaryOp, left: u32, right: u32) !u32 {
            const result_id = self.allocId();
            const float_type = try self.getFloatType(32);
            const bool_type = try self.getBoolType();

            const opcode: OpCode = switch (op) {
                .add => .OpFAdd,
                .sub => .OpFSub,
                .mul => .OpFMul,
                .div => .OpFDiv,
                .mod => .OpFMod,
                .eq => .OpFOrdEqual,
                .ne => .OpFOrdNotEqual,
                .lt => .OpFOrdLessThan,
                .le => .OpFOrdLessThanEqual,
                .gt => .OpFOrdGreaterThan,
                .ge => .OpFOrdGreaterThanEqual,
                .and_ => .OpLogicalAnd,
                .or_ => .OpLogicalOr,
                .bit_and => .OpBitwiseAnd,
                .bit_or => .OpBitwiseOr,
                .bit_xor => .OpBitwiseXor,
                .shl => .OpShiftLeftLogical,
                .shr => .OpShiftRightLogical,
                else => .OpFAdd,
            };

            const result_type = if (op.isComparison()) bool_type else float_type;
            try self.emitOp(&self.function_section, opcode, &.{ result_type, result_id, left, right });
            return result_id;
        }

        /// Generate a function call.
        pub fn generateCall(self: *Self, c: expr.Expr.CallExpr) !u32 {
            switch (c.function) {
                .barrier => {
                    const exec_scope = try self.getConstantU32(@intFromEnum(Scope.Workgroup));
                    const mem_scope = try self.getConstantU32(@intFromEnum(Scope.Workgroup));
                    const semantics = try self.getConstantU32(@intFromEnum(MemorySemantics.WorkgroupMemory) | @intFromEnum(MemorySemantics.AcquireRelease));
                    try self.emitOp(&self.function_section, .OpControlBarrier, &.{ exec_scope, mem_scope, semantics });
                    return 0;
                },
                .memory_barrier => {
                    const scope = try self.getConstantU32(@intFromEnum(Scope.Device));
                    const semantics = try self.getConstantU32(@intFromEnum(MemorySemantics.AcquireRelease));
                    try self.emitOp(&self.function_section, .OpMemoryBarrier, &.{ scope, semantics });
                    return 0;
                },
                .atomic_add => {
                    if (c.args.len >= 2) {
                        const ptr = try self.generateExprPtr(c.args[0]);
                        const value = try self.generateExpr(c.args[1]);
                        const uint_type = try self.getIntType(32, false);
                        const scope = try self.getConstantU32(@intFromEnum(Scope.Device));
                        const semantics = try self.getConstantU32(0);
                        const result_id = self.allocId();
                        try self.emitOp(&self.function_section, .OpAtomicIAdd, &.{ uint_type, result_id, ptr, scope, semantics, value });
                        return result_id;
                    }
                    return 0;
                },
                .clamp => {
                    if (c.args.len >= 3) {
                        const x = try self.generateExpr(c.args[0]);
                        const min_val = try self.generateExpr(c.args[1]);
                        const max_val = try self.generateExpr(c.args[2]);
                        const float_type = try self.getFloatType(32);
                        const result_id = self.allocId();
                        // GLSL.std.450 FClamp = 43
                        try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 43, x, min_val, max_val });
                        return result_id;
                    }
                    return 0;
                },
                .mix => {
                    if (c.args.len >= 3) {
                        const x = try self.generateExpr(c.args[0]);
                        const y = try self.generateExpr(c.args[1]);
                        const a = try self.generateExpr(c.args[2]);
                        const float_type = try self.getFloatType(32);
                        const result_id = self.allocId();
                        // GLSL.std.450 FMix = 46
                        try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 46, x, y, a });
                        return result_id;
                    }
                    return 0;
                },
                .fma => {
                    if (c.args.len >= 3) {
                        const a = try self.generateExpr(c.args[0]);
                        const b = try self.generateExpr(c.args[1]);
                        const cc = try self.generateExpr(c.args[2]);
                        const float_type = try self.getFloatType(32);
                        const result_id = self.allocId();
                        // GLSL.std.450 Fma = 50
                        try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, 50, a, b, cc });
                        return result_id;
                    }
                    return 0;
                },
                else => return 0,
            }
        }

        /// Generate load from pointer.
        pub fn generateLoad(self: *Self, ptr: u32) !u32 {
            const result_id = self.allocId();
            const float_type = try self.getFloatType(32);
            try self.emitOp(&self.function_section, .OpLoad, &.{ float_type, result_id, ptr });
            return result_id;
        }

        /// Generate access chain.
        pub fn generateAccessChain(self: *Self, base: u32, index: u32) !u32 {
            const result_id = self.allocId();
            const float_type = try self.getFloatType(32);
            const ptr_type = try self.getPointerType(float_type, .StorageBuffer);
            try self.emitOp(&self.function_section, .OpAccessChain, &.{ ptr_type, result_id, base, index });
            return result_id;
        }

        /// Generate type cast.
        pub fn generateCast(self: *Self, value: u32, target_type: u32) !u32 {
            const result_id = self.allocId();
            try self.emitOp(&self.function_section, .OpBitcast, &.{ target_type, result_id, value });
            return result_id;
        }
    };
}
