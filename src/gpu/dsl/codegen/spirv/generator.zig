//! SPIR-V Generator Core
//!
//! Main SpirvGenerator struct and the generate() method that produces
//! SPIR-V binary bytecode from kernel IR.

const std = @import("std");
const constants = @import("constants.zig");
const types_gen = @import("types_gen.zig");
const constants_gen = @import("constants_gen.zig");
const emit_mod = @import("emit.zig");
const codegen_mod = @import("codegen.zig");

const dsl_types = @import("../../types.zig");
const dsl_expr = @import("../../expr.zig");
const dsl_stmt = @import("../../stmt.zig");
const kernel = @import("../../kernel.zig");
const backend = @import("../backend.zig");

pub const SPIRV_MAGIC = constants.SPIRV_MAGIC;
pub const SPIRV_VERSION = constants.SPIRV_VERSION;
pub const GENERATOR_ID = constants.GENERATOR_ID;

pub const OpCode = constants.OpCode;
pub const Capability = constants.Capability;
pub const ExecutionModel = constants.ExecutionModel;
pub const AddressingModel = constants.AddressingModel;
pub const MemoryModel = constants.MemoryModel;
pub const ExecutionMode = constants.ExecutionMode;
pub const StorageClass = constants.StorageClass;
pub const Decoration = constants.Decoration;
pub const BuiltIn = constants.BuiltIn;
pub const MemorySemantics = constants.MemorySemantics;
pub const Scope = constants.Scope;

pub const TypeKey = types_gen.TypeKey;
pub const ConstKey = constants_gen.ConstKey;

/// SPIR-V code generator.
pub const SpirvGenerator = struct {
    allocator: std.mem.Allocator,
    /// SPIR-V words (binary output).
    words: std.ArrayListUnmanaged(u32),
    /// Current ID bound.
    next_id: u32,
    /// Type ID cache.
    type_ids: std.AutoHashMapUnmanaged(TypeKey, u32),
    /// Constant ID cache.
    const_ids: std.AutoHashMapUnmanaged(ConstKey, u32),
    /// Variable ID map (name -> id).
    var_ids: std.StringHashMapUnmanaged(u32),
    /// Type declarations (stored for later emission).
    type_section: std.ArrayListUnmanaged(u32),
    /// Constants section.
    const_section: std.ArrayListUnmanaged(u32),
    /// Annotations section.
    annotation_section: std.ArrayListUnmanaged(u32),
    /// Debug names section.
    debug_section: std.ArrayListUnmanaged(u32),
    /// Entry points section.
    entry_section: std.ArrayListUnmanaged(u32),
    /// Function section.
    function_section: std.ArrayListUnmanaged(u32),
    /// GLSL.std.450 extended instruction set ID.
    glsl_ext_id: u32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .words = .{},
            .next_id = 1,
            .type_ids = .{},
            .const_ids = .{},
            .var_ids = .{},
            .type_section = .{},
            .const_section = .{},
            .annotation_section = .{},
            .debug_section = .{},
            .entry_section = .{},
            .function_section = .{},
            .glsl_ext_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.words.deinit(self.allocator);
        self.type_ids.deinit(self.allocator);
        self.const_ids.deinit(self.allocator);
        self.var_ids.deinit(self.allocator);
        self.type_section.deinit(self.allocator);
        self.const_section.deinit(self.allocator);
        self.annotation_section.deinit(self.allocator);
        self.debug_section.deinit(self.allocator);
        self.entry_section.deinit(self.allocator);
        self.function_section.deinit(self.allocator);
    }

    /// Allocate a new ID.
    pub fn allocId(self: *Self) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    /// Generate SPIR-V binary from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Reset state
        self.next_id = 1;
        self.words.clearRetainingCapacity();
        self.type_ids.clearRetainingCapacity();
        self.const_ids.clearRetainingCapacity();
        self.var_ids.clearRetainingCapacity();
        self.type_section.clearRetainingCapacity();
        self.const_section.clearRetainingCapacity();
        self.annotation_section.clearRetainingCapacity();
        self.debug_section.clearRetainingCapacity();
        self.entry_section.clearRetainingCapacity();
        self.function_section.clearRetainingCapacity();

        // Reserve header space (5 words)
        try self.words.appendSlice(self.allocator, &[_]u32{ 0, 0, 0, 0, 0 });

        // Import GLSL.std.450 extended instruction set
        self.glsl_ext_id = self.allocId();
        try self.emitExtInstImport(self.glsl_ext_id, "GLSL.std.450");

        // Emit capabilities
        try self.emitCapability(.Shader);
        if (ir.required_features.fp64) {
            try self.emitCapability(.Float64);
        }
        if (ir.required_features.int64) {
            try self.emitCapability(.Int64);
        }
        if (ir.required_features.subgroups) {
            try self.emitCapability(.GroupNonUniform);
        }

        // Memory model
        try self.emitMemoryModel(.Logical, .GLSL450);

        // Generate types needed for built-ins and buffers
        const void_type = try self.getVoidType();
        const func_type = try self.getFunctionType(void_type, &.{});

        // Generate built-in variables
        const uvec3_type = try self.getVectorType(try self.getIntType(32, false), 3);
        const uint_type = try self.getIntType(32, false);
        const uvec3_ptr_input = try self.getPointerType(uvec3_type, .Input);
        const uint_ptr_input = try self.getPointerType(uint_type, .Input);

        // Create built-in variables
        const global_inv_id = self.allocId();
        const local_inv_id = self.allocId();
        const workgroup_id_var = self.allocId();
        const local_inv_index = self.allocId();

        try self.emitVariable(&self.type_section, global_inv_id, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, local_inv_id, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, workgroup_id_var, uvec3_ptr_input, .Input, null);
        try self.emitVariable(&self.type_section, local_inv_index, uint_ptr_input, .Input, null);

        // Decorate built-ins
        try self.emitDecorate(&self.annotation_section, global_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.GlobalInvocationId)});
        try self.emitDecorate(&self.annotation_section, local_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationId)});
        try self.emitDecorate(&self.annotation_section, workgroup_id_var, .BuiltIn, &.{@intFromEnum(BuiltIn.WorkgroupId)});
        try self.emitDecorate(&self.annotation_section, local_inv_index, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationIndex)});

        // Register built-in variable names
        try self.var_ids.put(self.allocator, "globalInvocationId", global_inv_id);
        try self.var_ids.put(self.allocator, "localInvocationId", local_inv_id);
        try self.var_ids.put(self.allocator, "workgroupId", workgroup_id_var);
        try self.var_ids.put(self.allocator, "localInvocationIndex", local_inv_index);

        // Debug names
        try self.emitName(&self.debug_section, global_inv_id, "globalInvocationId");
        try self.emitName(&self.debug_section, local_inv_id, "localInvocationId");
        try self.emitName(&self.debug_section, workgroup_id_var, "workgroupId");
        try self.emitName(&self.debug_section, local_inv_index, "localInvocationIndex");

        // Generate buffer variables
        var interface_ids = std.ArrayListUnmanaged(u32){};
        defer interface_ids.deinit(self.allocator);

        // Add built-ins to interface
        try interface_ids.append(self.allocator, global_inv_id);
        try interface_ids.append(self.allocator, local_inv_id);
        try interface_ids.append(self.allocator, workgroup_id_var);
        try interface_ids.append(self.allocator, local_inv_index);

        // Process buffers
        for (ir.buffers) |buf| {
            const elem_type = try self.typeFromIR(buf.element_type);
            const runtime_arr = try self.getRuntimeArrayType(elem_type);

            // Create block struct
            const struct_id = self.allocId();
            try self.emitTypeStruct(&self.type_section, struct_id, &.{runtime_arr});

            // Decorate struct and member
            try self.emitDecorate(&self.annotation_section, struct_id, .Block, &.{});
            try self.emitMemberDecorate(&self.annotation_section, struct_id, 0, .Offset, &.{0});

            // Array stride
            const elem_size = self.getTypeSize(buf.element_type);
            try self.emitDecorate(&self.annotation_section, runtime_arr, .ArrayStride, &.{elem_size});

            // Create pointer and variable
            const ptr_type = try self.getPointerType(struct_id, .StorageBuffer);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .StorageBuffer, null);

            // Decorate binding
            try self.emitDecorate(&self.annotation_section, var_id, .DescriptorSet, &.{buf.group});
            try self.emitDecorate(&self.annotation_section, var_id, .Binding, &.{buf.binding});

            // Decorate access
            if (buf.access == .read_only) {
                try self.emitDecorate(&self.annotation_section, var_id, .NonWritable, &.{});
            }

            try self.var_ids.put(self.allocator, buf.name, var_id);
            try self.emitName(&self.debug_section, var_id, buf.name);
            try interface_ids.append(self.allocator, var_id);
        }

        // Process uniforms
        if (ir.uniforms.len > 0) {
            // Create uniform block struct
            var member_types = std.ArrayListUnmanaged(u32){};
            defer member_types.deinit(self.allocator);

            var member_offset: u32 = 0;
            const struct_id = self.allocId();

            for (ir.uniforms, 0..) |uni, i| {
                const member_type = try self.typeFromIR(uni.ty);
                try member_types.append(self.allocator, member_type);

                // Decorate member offset
                try self.emitMemberDecorate(&self.annotation_section, struct_id, @intCast(i), .Offset, &.{member_offset});
                try self.emitMemberName(&self.debug_section, struct_id, @intCast(i), uni.name);

                member_offset += self.getTypeSize(uni.ty);
            }

            try self.emitTypeStruct(&self.type_section, struct_id, member_types.items);
            try self.emitDecorate(&self.annotation_section, struct_id, .Block, &.{});

            const ptr_type = try self.getPointerType(struct_id, .Uniform);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .Uniform, null);
            try self.emitDecorate(&self.annotation_section, var_id, .DescriptorSet, &.{0});
            try self.emitDecorate(&self.annotation_section, var_id, .Binding, &.{0});

            try self.var_ids.put(self.allocator, "uniforms", var_id);
            try self.emitName(&self.debug_section, var_id, "uniforms");
            try interface_ids.append(self.allocator, var_id);
        }

        // Process shared memory
        for (ir.shared_memory) |shared| {
            const elem_type = try self.typeFromIR(shared.element_type);
            const arr_type = if (shared.size) |size|
                try self.getArrayType(elem_type, @intCast(size))
            else
                try self.getRuntimeArrayType(elem_type);

            const ptr_type = try self.getPointerType(arr_type, .Workgroup);
            const var_id = self.allocId();
            try self.emitVariable(&self.type_section, var_id, ptr_type, .Workgroup, null);

            try self.var_ids.put(self.allocator, shared.name, var_id);
            try self.emitName(&self.debug_section, var_id, shared.name);
        }

        // Entry point
        const main_func_id = self.allocId();
        try self.emitEntryPoint(.GLCompute, main_func_id, "main", interface_ids.items);
        try self.emitExecutionMode(main_func_id, .LocalSize, &.{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });

        // Generate main function
        try self.emitFunction(&self.function_section, main_func_id, void_type, .{ .none = {} }, func_type);

        const entry_label = self.allocId();
        try self.emitLabel(&self.function_section, entry_label);

        // Generate function body
        for (ir.body) |s| {
            try self.generateStmt(s);
        }

        // Return and end function
        try self.emitReturn(&self.function_section);
        try self.emitFunctionEnd(&self.function_section);

        // Assemble final module
        try self.assembleModule();

        // Convert to byte slice
        const word_bytes = std.mem.sliceAsBytes(self.words.items);
        const code = try self.allocator.dupe(u8, word_bytes);
        const entry_point_name = try self.allocator.dupe(u8, "main");

        return .{
            .code = code,
            .entry_point = entry_point_name,
            .backend = .vulkan,
            .language = .spirv,
            .spirv_binary = code,
        };
    }

    /// Assemble the final SPIR-V module.
    fn assembleModule(self: *Self) !void {
        // Write header
        self.words.items[0] = SPIRV_MAGIC;
        self.words.items[1] = SPIRV_VERSION;
        self.words.items[2] = GENERATOR_ID;
        self.words.items[3] = self.next_id; // Bound
        self.words.items[4] = 0; // Schema

        // Sections are already in words from capabilities/memory model
        // Now append other sections in order
        try self.words.appendSlice(self.allocator, self.entry_section.items);
        try self.words.appendSlice(self.allocator, self.debug_section.items);
        try self.words.appendSlice(self.allocator, self.annotation_section.items);
        try self.words.appendSlice(self.allocator, self.type_section.items);
        try self.words.appendSlice(self.allocator, self.const_section.items);
        try self.words.appendSlice(self.allocator, self.function_section.items);
    }

    // =========================================================================
    // Code Generation (from codegen.zig)
    // =========================================================================

    /// Generate code for a statement.
    pub fn generateStmt(self: *Self, s: *const dsl_stmt.Stmt) !void {
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
    pub fn generateExpr(self: *Self, e: *const dsl_expr.Expr) !u32 {
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
    pub fn generateExprPtr(self: *Self, e: *const dsl_expr.Expr) !u32 {
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
    pub fn generateLiteral(self: *Self, lit: dsl_expr.Literal) !u32 {
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
    pub fn generateUnaryOp(self: *Self, op: dsl_expr.UnaryOp, operand: u32) !u32 {
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
            .sqrt => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 31), operand }); // Sqrt
            },
            .sin => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 13), operand }); // Sin
            },
            .cos => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 14), operand }); // Cos
            },
            .tan => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 15), operand }); // Tan
            },
            .exp => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 27), operand }); // Exp
            },
            .log => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 28), operand }); // Log
            },
            .abs => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 4), operand }); // FAbs
            },
            .floor => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 8), operand }); // Floor
            },
            .ceil => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 9), operand }); // Ceil
            },
            .round => {
                try self.emitOp(&self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 1), operand }); // Round
            },
            else => {
                // Fallback
                try self.emitOp(&self.function_section, .OpCopyObject, &.{ float_type, result_id, operand });
            },
        }
        return result_id;
    }

    /// Generate a binary operation.
    pub fn generateBinaryOp(self: *Self, op: dsl_expr.BinaryOp, left: u32, right: u32) !u32 {
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
    pub fn generateCall(self: *Self, c: dsl_expr.Expr.CallExpr) !u32 {
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

    // =========================================================================
    // Type Generation (from types_gen.zig)
    // =========================================================================

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
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, id);
        try operands.append(self.allocator, return_type);
        try operands.appendSlice(self.allocator, param_types);
        try self.emitOp(&self.type_section, .OpTypeFunction, operands.items);
        try self.type_ids.put(self.allocator, key, id);
        return id;
    }

    // =========================================================================
    // Constant Generation (from constants_gen.zig)
    // =========================================================================

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

    // =========================================================================
    // Instruction Emission (from emit.zig)
    // =========================================================================

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
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, id);
        try self.appendString(&operands, name_str);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpName));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    pub fn emitMemberName(self: *Self, section: *std.ArrayListUnmanaged(u32), type_id: u32, member: u32, name_str: []const u8) !void {
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
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
        var operands = std.ArrayListUnmanaged(u32){};
        defer operands.deinit(self.allocator);
        try operands.append(self.allocator, result_id);
        try operands.appendSlice(self.allocator, member_types);

        const word_count: u32 = @intCast(1 + operands.items.len);
        const first_word = (word_count << 16) | @as(u32, @intFromEnum(OpCode.OpTypeStruct));
        try section.append(self.allocator, first_word);
        try section.appendSlice(self.allocator, operands.items);
    }

    pub fn emitVariable(self: *Self, section: *std.ArrayListUnmanaged(u32), result_id: u32, type_id: u32, storage_class: StorageClass, initializer: ?u32) !void {
        var operands = std.ArrayListUnmanaged(u32){};
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
