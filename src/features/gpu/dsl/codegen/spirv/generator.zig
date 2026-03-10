//! SPIR-V Generator Core
//!
//! Main SpirvGenerator struct and the generate() method that produces
//! SPIR-V binary bytecode from kernel IR.
//!
//! This file is the orchestrator that delegates to submodules:
//! - type_codegen.zig — type generation/encoding logic
//! - const_codegen.zig — constant generation logic
//! - instruction_emit.zig — instruction emission logic
//! - codegen.zig — statement/expression code generation (legacy mixin, not used here)

const std = @import("std");
const constants = @import("constants");
const types_gen = @import("types_gen");
const constants_gen = @import("constants_gen");
const type_codegen = @import("type_codegen");
const const_codegen = @import("const_codegen");
const instruction_emit = @import("instruction_emit");

const dsl_types = @import("../../types");
const dsl_expr = @import("../../expr");
const dsl_stmt = @import("../../stmt");
const kernel = @import("../../kernel");
const backend = @import("../backend");

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
    /// Current loop merge label (for break statements).
    loop_merge_label: ?u32 = null,
    /// Current loop continue label (for continue statements).
    loop_continue_label: ?u32 = null,
    /// Pointer-to-type map: maps a pointer result ID to the type ID it points to.
    ptr_type_map: std.AutoHashMapUnmanaged(u32, u32),
    /// Buffer element type map: maps buffer variable ID to its element type ID.
    buffer_elem_type_map: std.AutoHashMapUnmanaged(u32, u32),

    const Self = @This();

    // =========================================================================
    // Core lifecycle
    // =========================================================================

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
            .loop_merge_label = null,
            .loop_continue_label = null,
            .ptr_type_map = .{},
            .buffer_elem_type_map = .{},
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
        self.ptr_type_map.deinit(self.allocator);
        self.buffer_elem_type_map.deinit(self.allocator);
    }

    /// Allocate a new ID.
    pub fn allocId(self: *Self) u32 {
        const id = self.next_id;
        self.next_id += 1;
        return id;
    }

    /// Errors produced during SPIR-V code generation.
    pub const CodeGenError = error{ InvalidIR, OutOfMemory };

    // =========================================================================
    // Main generate() orchestrator
    // =========================================================================

    /// Generate SPIR-V binary from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) CodeGenError!backend.GeneratedSource {
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
        self.ptr_type_map.clearRetainingCapacity();
        self.buffer_elem_type_map.clearRetainingCapacity();
        self.loop_merge_label = null;
        self.loop_continue_label = null;

        // Reserve header space (5 words)
        try self.words.appendSlice(self.allocator, &[_]u32{ 0, 0, 0, 0, 0 });

        // Import GLSL.std.450 extended instruction set
        self.glsl_ext_id = self.allocId();
        try instruction_emit.emitExtInstImport(self, self.glsl_ext_id, "GLSL.std.450");

        // Emit capabilities
        try instruction_emit.emitCapability(self, .Shader);
        if (ir.required_features.fp64) {
            try instruction_emit.emitCapability(self, .Float64);
        }
        if (ir.required_features.int64) {
            try instruction_emit.emitCapability(self, .Int64);
        }
        if (ir.required_features.subgroups) {
            try instruction_emit.emitCapability(self, .GroupNonUniform);
        }

        // Memory model
        try instruction_emit.emitMemoryModel(self, .Logical, .GLSL450);

        // Generate types needed for built-ins and buffers
        const void_type = try type_codegen.getVoidType(self);
        const func_type = try type_codegen.getFunctionType(self, void_type, &.{});

        // Generate built-in variables
        const uvec3_type = try type_codegen.getVectorType(self, try type_codegen.getIntType(self, 32, false), 3);
        const uint_type = try type_codegen.getIntType(self, 32, false);
        const uvec3_ptr_input = try type_codegen.getPointerType(self, uvec3_type, .Input);
        const uint_ptr_input = try type_codegen.getPointerType(self, uint_type, .Input);

        // Create built-in variables
        const global_inv_id = self.allocId();
        const local_inv_id = self.allocId();
        const workgroup_id_var = self.allocId();
        const local_inv_index = self.allocId();

        try instruction_emit.emitVariable(self, &self.type_section, global_inv_id, uvec3_ptr_input, .Input, null);
        try instruction_emit.emitVariable(self, &self.type_section, local_inv_id, uvec3_ptr_input, .Input, null);
        try instruction_emit.emitVariable(self, &self.type_section, workgroup_id_var, uvec3_ptr_input, .Input, null);
        try instruction_emit.emitVariable(self, &self.type_section, local_inv_index, uint_ptr_input, .Input, null);

        // Decorate built-ins
        try instruction_emit.emitDecorate(self, &self.annotation_section, global_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.GlobalInvocationId)});
        try instruction_emit.emitDecorate(self, &self.annotation_section, local_inv_id, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationId)});
        try instruction_emit.emitDecorate(self, &self.annotation_section, workgroup_id_var, .BuiltIn, &.{@intFromEnum(BuiltIn.WorkgroupId)});
        try instruction_emit.emitDecorate(self, &self.annotation_section, local_inv_index, .BuiltIn, &.{@intFromEnum(BuiltIn.LocalInvocationIndex)});

        // Register built-in variable names
        try self.var_ids.put(self.allocator, "globalInvocationId", global_inv_id);
        try self.var_ids.put(self.allocator, "localInvocationId", local_inv_id);
        try self.var_ids.put(self.allocator, "workgroupId", workgroup_id_var);
        try self.var_ids.put(self.allocator, "localInvocationIndex", local_inv_index);

        // Debug names
        try instruction_emit.emitName(self, &self.debug_section, global_inv_id, "globalInvocationId");
        try instruction_emit.emitName(self, &self.debug_section, local_inv_id, "localInvocationId");
        try instruction_emit.emitName(self, &self.debug_section, workgroup_id_var, "workgroupId");
        try instruction_emit.emitName(self, &self.debug_section, local_inv_index, "localInvocationIndex");

        // Generate buffer variables
        var interface_ids = std.ArrayListUnmanaged(u32).empty;
        defer interface_ids.deinit(self.allocator);

        // Add built-ins to interface
        try interface_ids.append(self.allocator, global_inv_id);
        try interface_ids.append(self.allocator, local_inv_id);
        try interface_ids.append(self.allocator, workgroup_id_var);
        try interface_ids.append(self.allocator, local_inv_index);

        // Process buffers
        for (ir.buffers) |buf| {
            const elem_type = try type_codegen.typeFromIR(self, buf.element_type);
            const runtime_arr = try type_codegen.getRuntimeArrayType(self, elem_type);

            // Create block struct
            const struct_id = self.allocId();
            try instruction_emit.emitTypeStruct(self, &self.type_section, struct_id, &.{runtime_arr});

            // Decorate struct and member
            try instruction_emit.emitDecorate(self, &self.annotation_section, struct_id, .Block, &.{});
            try instruction_emit.emitMemberDecorate(self, &self.annotation_section, struct_id, 0, .Offset, &.{0});

            // Array stride
            const elem_size = type_codegen.getTypeSize(self, buf.element_type);
            try instruction_emit.emitDecorate(self, &self.annotation_section, runtime_arr, .ArrayStride, &.{elem_size});

            // Create pointer and variable
            const ptr_type = try type_codegen.getPointerType(self, struct_id, .StorageBuffer);
            const var_id = self.allocId();
            try instruction_emit.emitVariable(self, &self.type_section, var_id, ptr_type, .StorageBuffer, null);

            // Decorate binding
            try instruction_emit.emitDecorate(self, &self.annotation_section, var_id, .DescriptorSet, &.{buf.group});
            try instruction_emit.emitDecorate(self, &self.annotation_section, var_id, .Binding, &.{buf.binding});

            // Decorate access
            if (buf.access == .read_only) {
                try instruction_emit.emitDecorate(self, &self.annotation_section, var_id, .NonWritable, &.{});
            }

            try self.var_ids.put(self.allocator, buf.name, var_id);
            try instruction_emit.emitName(self, &self.debug_section, var_id, buf.name);
            try interface_ids.append(self.allocator, var_id);

            // Track buffer variable -> element type for type-aware loads/access chains
            try self.buffer_elem_type_map.put(self.allocator, var_id, elem_type);
        }

        // Process uniforms
        if (ir.uniforms.len > 0) {
            // Create uniform block struct
            var member_types = std.ArrayListUnmanaged(u32).empty;
            defer member_types.deinit(self.allocator);

            var member_offset: u32 = 0;
            const struct_id = self.allocId();

            for (ir.uniforms, 0..) |uni, i| {
                const member_type = try type_codegen.typeFromIR(self, uni.ty);
                try member_types.append(self.allocator, member_type);

                // Decorate member offset
                try instruction_emit.emitMemberDecorate(self, &self.annotation_section, struct_id, @intCast(i), .Offset, &.{member_offset});
                try instruction_emit.emitMemberName(self, &self.debug_section, struct_id, @intCast(i), uni.name);

                member_offset += type_codegen.getTypeSize(self, uni.ty);
            }

            try instruction_emit.emitTypeStruct(self, &self.type_section, struct_id, member_types.items);
            try instruction_emit.emitDecorate(self, &self.annotation_section, struct_id, .Block, &.{});

            const ptr_type = try type_codegen.getPointerType(self, struct_id, .Uniform);
            const var_id = self.allocId();
            try instruction_emit.emitVariable(self, &self.type_section, var_id, ptr_type, .Uniform, null);
            try instruction_emit.emitDecorate(self, &self.annotation_section, var_id, .DescriptorSet, &.{0});
            try instruction_emit.emitDecorate(self, &self.annotation_section, var_id, .Binding, &.{0});

            try self.var_ids.put(self.allocator, "uniforms", var_id);
            try instruction_emit.emitName(self, &self.debug_section, var_id, "uniforms");
            try interface_ids.append(self.allocator, var_id);
        }

        // Process shared memory
        for (ir.shared_memory) |shared| {
            const elem_type = try type_codegen.typeFromIR(self, shared.element_type);
            const arr_type = if (shared.size) |size|
                try type_codegen.getArrayType(self, elem_type, @intCast(size))
            else
                try type_codegen.getRuntimeArrayType(self, elem_type);

            const ptr_type = try type_codegen.getPointerType(self, arr_type, .Workgroup);
            const var_id = self.allocId();
            try instruction_emit.emitVariable(self, &self.type_section, var_id, ptr_type, .Workgroup, null);

            try self.var_ids.put(self.allocator, shared.name, var_id);
            try instruction_emit.emitName(self, &self.debug_section, var_id, shared.name);
        }

        // Entry point
        const main_func_id = self.allocId();
        try instruction_emit.emitEntryPoint(self, .GLCompute, main_func_id, "main", interface_ids.items);
        try instruction_emit.emitExecutionMode(self, main_func_id, .LocalSize, &.{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });

        // Generate main function
        try instruction_emit.emitFunction(self, &self.function_section, main_func_id, void_type, .{ .none = {} }, func_type);

        const entry_label = self.allocId();
        try instruction_emit.emitLabel(self, &self.function_section, entry_label);

        // Generate function body
        for (ir.body) |s| {
            try self.generateStmt(s);
        }

        // Return and end function
        try instruction_emit.emitReturn(self, &self.function_section);
        try instruction_emit.emitFunctionEnd(self, &self.function_section);

        // Assemble final module
        try self.assembleModule();

        // Convert to byte slice
        const word_bytes = std.mem.sliceAsBytes(self.words.items);
        const code = try self.allocator.dupe(u8, word_bytes);
        errdefer self.allocator.free(code);
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
    // Code Generation (statement/expression handling)
    // =========================================================================

    /// Generate code for a statement.
    pub fn generateStmt(self: *Self, s: *const dsl_stmt.Stmt) CodeGenError!void {
        switch (s.*) {
            .var_decl => |v| {
                const ty = try type_codegen.typeFromIR(self, v.ty);
                const ptr_type = try type_codegen.getPointerType(self, ty, .Function);
                const var_id = self.allocId();

                // Initialize variable
                const init_id: ?u32 = if (v.init) |init_expr|
                    try self.generateExpr(init_expr)
                else
                    null;

                try instruction_emit.emitVariable(self, &self.function_section, var_id, ptr_type, .Function, init_id);
                try self.var_ids.put(self.allocator, v.name, var_id);
            },
            .assign => |a| {
                const value_id = try self.generateExpr(a.value);
                const target_id = try self.generateExprPtr(a.target);
                try instruction_emit.emitStore(self, &self.function_section, target_id, value_id);
            },
            .compound_assign => |ca| {
                const target_ptr = try self.generateExprPtr(ca.target);
                const current_val = try self.generateLoad(target_ptr);
                const operand_val = try self.generateExpr(ca.value);
                const result = try self.generateBinaryOp(ca.op, current_val, operand_val);
                try instruction_emit.emitStore(self, &self.function_section, target_ptr, result);
            },
            .if_ => |i| {
                const cond_id = try self.generateExpr(i.condition);
                const then_label = self.allocId();
                const else_label = self.allocId();
                const merge_label = self.allocId();

                try instruction_emit.emitSelectionMerge(self, &self.function_section, merge_label);
                try instruction_emit.emitBranchConditional(self, &self.function_section, cond_id, then_label, if (i.else_body != null) else_label else merge_label);

                // Then block
                try instruction_emit.emitLabel(self, &self.function_section, then_label);
                for (i.then_body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try instruction_emit.emitBranch(self, &self.function_section, merge_label);

                // Else block
                if (i.else_body) |else_body| {
                    try instruction_emit.emitLabel(self, &self.function_section, else_label);
                    for (else_body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try instruction_emit.emitBranch(self, &self.function_section, merge_label);
                }

                // Merge block
                try instruction_emit.emitLabel(self, &self.function_section, merge_label);
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

                // Save outer loop context and set inner
                const saved_merge = self.loop_merge_label;
                const saved_continue = self.loop_continue_label;
                self.loop_merge_label = merge_label;
                self.loop_continue_label = continue_label;

                try instruction_emit.emitBranch(self, &self.function_section, header_label);
                try instruction_emit.emitLabel(self, &self.function_section, header_label);
                try instruction_emit.emitLoopMerge(self, &self.function_section, merge_label, continue_label);

                if (f.condition) |cond| {
                    const cond_id = try self.generateExpr(cond);
                    try instruction_emit.emitBranchConditional(self, &self.function_section, cond_id, body_label, merge_label);
                } else {
                    try instruction_emit.emitBranch(self, &self.function_section, body_label);
                }

                // Body
                try instruction_emit.emitLabel(self, &self.function_section, body_label);
                for (f.body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try instruction_emit.emitBranch(self, &self.function_section, continue_label);

                // Continue block
                try instruction_emit.emitLabel(self, &self.function_section, continue_label);
                if (f.update) |update| {
                    try self.generateStmt(update);
                }
                try instruction_emit.emitBranch(self, &self.function_section, header_label);

                // Merge
                try instruction_emit.emitLabel(self, &self.function_section, merge_label);

                // Restore outer loop context
                self.loop_merge_label = saved_merge;
                self.loop_continue_label = saved_continue;
            },
            .while_ => |w| {
                const header_label = self.allocId();
                const body_label = self.allocId();
                const continue_label = self.allocId();
                const merge_label = self.allocId();

                // Save outer loop context and set inner
                const saved_merge = self.loop_merge_label;
                const saved_continue = self.loop_continue_label;
                self.loop_merge_label = merge_label;
                self.loop_continue_label = continue_label;

                try instruction_emit.emitBranch(self, &self.function_section, header_label);
                try instruction_emit.emitLabel(self, &self.function_section, header_label);
                try instruction_emit.emitLoopMerge(self, &self.function_section, merge_label, continue_label);

                const cond_id = try self.generateExpr(w.condition);
                try instruction_emit.emitBranchConditional(self, &self.function_section, cond_id, body_label, merge_label);

                try instruction_emit.emitLabel(self, &self.function_section, body_label);
                for (w.body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try instruction_emit.emitBranch(self, &self.function_section, continue_label);

                // Continue target branches back to header
                try instruction_emit.emitLabel(self, &self.function_section, continue_label);
                try instruction_emit.emitBranch(self, &self.function_section, header_label);

                try instruction_emit.emitLabel(self, &self.function_section, merge_label);

                // Restore outer loop context
                self.loop_merge_label = saved_merge;
                self.loop_continue_label = saved_continue;
            },
            .do_while => |dw| {
                // do { body } while (condition);
                // Structure: header -> body -> continue (condition) -> header or merge
                const header_label = self.allocId();
                const body_label = self.allocId();
                const continue_label = self.allocId();
                const merge_label = self.allocId();

                // Save outer loop context and set inner
                const saved_merge = self.loop_merge_label;
                const saved_continue = self.loop_continue_label;
                self.loop_merge_label = merge_label;
                self.loop_continue_label = continue_label;

                // Branch to header
                try instruction_emit.emitBranch(self, &self.function_section, header_label);
                try instruction_emit.emitLabel(self, &self.function_section, header_label);
                try instruction_emit.emitLoopMerge(self, &self.function_section, merge_label, continue_label);
                // Unconditionally enter body (do-while always executes body first)
                try instruction_emit.emitBranch(self, &self.function_section, body_label);

                // Body block
                try instruction_emit.emitLabel(self, &self.function_section, body_label);
                for (dw.body) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
                try instruction_emit.emitBranch(self, &self.function_section, continue_label);

                // Continue block: evaluate condition
                try instruction_emit.emitLabel(self, &self.function_section, continue_label);
                const cond_id = try self.generateExpr(dw.condition);
                // If condition true, loop back to header; otherwise merge (exit)
                try instruction_emit.emitBranchConditional(self, &self.function_section, cond_id, header_label, merge_label);

                // Merge block
                try instruction_emit.emitLabel(self, &self.function_section, merge_label);

                // Restore outer loop context
                self.loop_merge_label = saved_merge;
                self.loop_continue_label = saved_continue;
            },
            .return_ => {
                try instruction_emit.emitReturn(self, &self.function_section);
            },
            .break_ => {
                if (self.loop_merge_label) |merge_label| {
                    try instruction_emit.emitBranch(self, &self.function_section, merge_label);
                }
                // If no loop context, break is a no-op (shouldn't happen in valid IR)
            },
            .continue_ => {
                if (self.loop_continue_label) |continue_label| {
                    try instruction_emit.emitBranch(self, &self.function_section, continue_label);
                }
                // If no loop context, continue is a no-op (shouldn't happen in valid IR)
            },
            .expr_stmt => |e| {
                _ = try self.generateExpr(e);
            },
            .block => |b| {
                for (b.statements) |body_stmt| {
                    try self.generateStmt(body_stmt);
                }
            },
            .discard => {
                // OpKill is for fragment shaders; in compute shaders discard is a no-op.
                // Emit OpReturn as a reasonable fallback to terminate the invocation.
                try instruction_emit.emitReturn(self, &self.function_section);
            },
            .switch_ => |sw| {
                // Generate selector value
                const selector_id = try self.generateExpr(sw.selector);
                const merge_label = self.allocId();

                // Allocate labels for each case and default
                var case_labels = std.ArrayListUnmanaged(u32).empty;
                defer case_labels.deinit(self.allocator);
                for (sw.cases) |_| {
                    try case_labels.append(self.allocator, self.allocId());
                }
                const default_label = if (sw.default != null) self.allocId() else merge_label;

                // Emit selection merge
                try instruction_emit.emitSelectionMerge(self, &self.function_section, merge_label);

                // Build OpSwitch operands: selector, default, then pairs of (literal, label)
                var switch_operands = std.ArrayListUnmanaged(u32).empty;
                defer switch_operands.deinit(self.allocator);
                try switch_operands.append(self.allocator, selector_id);
                try switch_operands.append(self.allocator, default_label);
                for (sw.cases, 0..) |case_item, idx| {
                    // Extract literal value as u32
                    const case_val: u32 = switch (case_item.value) {
                        .i32_ => |v| @bitCast(v),
                        .u32_ => |v| v,
                        .bool_ => |v| if (v) @as(u32, 1) else 0,
                        else => 0,
                    };
                    try switch_operands.append(self.allocator, case_val);
                    try switch_operands.append(self.allocator, case_labels.items[idx]);
                }
                try instruction_emit.emitOp(self, &self.function_section, .OpSwitch, switch_operands.items);

                // Emit case blocks
                for (sw.cases, 0..) |case_item, idx| {
                    try instruction_emit.emitLabel(self, &self.function_section, case_labels.items[idx]);
                    for (case_item.body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try instruction_emit.emitBranch(self, &self.function_section, merge_label);
                }

                // Emit default block if present
                if (sw.default) |default_body| {
                    try instruction_emit.emitLabel(self, &self.function_section, default_label);
                    for (default_body) |body_stmt| {
                        try self.generateStmt(body_stmt);
                    }
                    try instruction_emit.emitBranch(self, &self.function_section, merge_label);
                }

                // Merge block
                try instruction_emit.emitLabel(self, &self.function_section, merge_label);
            },
        }
    }

    /// Generate an expression and return its result ID.
    pub fn generateExpr(self: *Self, e: *const dsl_expr.Expr) CodeGenError!u32 {
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
                const zero = try const_codegen.getConstantU32(self, 0);
                const field_ptr = try self.generateAccessChain(base_ptr, zero);
                return try self.generateLoad(field_ptr);
            },
            .cast => |c| {
                const operand_id = try self.generateExpr(c.operand);
                const target_type = try type_codegen.typeFromIR(self, c.target_type);
                return try self.generateCast(operand_id, target_type);
            },
            .select => |s| {
                const cond_id = try self.generateExpr(s.condition);
                const true_id = try self.generateExpr(s.true_value);
                const false_id = try self.generateExpr(s.false_value);
                const result_id = self.allocId();
                const float_type = try type_codegen.getFloatType(self, 32);
                try instruction_emit.emitOp(self, &self.function_section, .OpSelect, &.{ float_type, result_id, cond_id, true_id, false_id });
                return result_id;
            },
            .vector_construct => |vc| {
                var component_ids = std.ArrayListUnmanaged(u32).empty;
                defer component_ids.deinit(self.allocator);

                for (vc.components) |comp| {
                    try component_ids.append(self.allocator, try self.generateExpr(comp));
                }

                const elem_type = try type_codegen.scalarTypeFromIR(self, vc.element_type);
                const vec_type = try type_codegen.getVectorType(self, elem_type, vc.size);
                const result_id = self.allocId();

                var operands = std.ArrayListUnmanaged(u32).empty;
                defer operands.deinit(self.allocator);
                try operands.append(self.allocator, vec_type);
                try operands.append(self.allocator, result_id);
                try operands.appendSlice(self.allocator, component_ids.items);

                try instruction_emit.emitOp(self, &self.function_section, .OpCompositeConstruct, operands.items);
                return result_id;
            },
            .swizzle => |sw| {
                const base_id = try self.generateExpr(sw.base);
                if (sw.components.len == 1) {
                    // Extract single component
                    const result_id = self.allocId();
                    const float_type = try type_codegen.getFloatType(self, 32);
                    try instruction_emit.emitOp(self, &self.function_section, .OpCompositeExtract, &.{ float_type, result_id, base_id, sw.components[0] });
                    return result_id;
                } else {
                    // Vector shuffle
                    const elem_type = try type_codegen.getFloatType(self, 32);
                    const vec_type = try type_codegen.getVectorType(self, elem_type, @intCast(sw.components.len));
                    const result_id = self.allocId();

                    var operands = std.ArrayListUnmanaged(u32).empty;
                    defer operands.deinit(self.allocator);
                    try operands.append(self.allocator, vec_type);
                    try operands.append(self.allocator, result_id);
                    try operands.append(self.allocator, base_id);
                    try operands.append(self.allocator, base_id);
                    for (sw.components) |c| {
                        try operands.append(self.allocator, c);
                    }

                    try instruction_emit.emitOp(self, &self.function_section, .OpVectorShuffle, operands.items);
                    return result_id;
                }
            },
        };
    }

    /// Generate expression as pointer (for assignments).
    pub fn generateExprPtr(self: *Self, e: *const dsl_expr.Expr) CodeGenError!u32 {
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
                const zero = try const_codegen.getConstantU32(self, 0);
                return try self.generateAccessChain(base_ptr, zero);
            },
            else => error.InvalidIR,
        };
    }

    /// Generate a literal constant.
    pub fn generateLiteral(self: *Self, lit: dsl_expr.Literal) !u32 {
        return switch (lit) {
            .bool_ => |v| if (v) try const_codegen.getConstantTrue(self) else try const_codegen.getConstantFalse(self),
            .i32_ => |v| try const_codegen.getConstantI32(self, v),
            .i64_ => |v| try const_codegen.getConstantI64(self, v),
            .u32_ => |v| try const_codegen.getConstantU32(self, v),
            .u64_ => |v| try const_codegen.getConstantU64(self, v),
            .f32_ => |v| try const_codegen.getConstantF32(self, v),
            .f64_ => |v| try const_codegen.getConstantF64(self, v),
        };
    }

    /// Generate a unary operation.
    /// Covers all GLSL.std.450 unary math functions and prefix operators.
    pub fn generateUnaryOp(self: *Self, op: dsl_expr.UnaryOp, operand: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try type_codegen.getFloatType(self, 32);

        switch (op) {
            .neg => {
                try instruction_emit.emitOp(self, &self.function_section, .OpFNegate, &.{ float_type, result_id, operand });
            },
            .not => {
                const bool_type = try type_codegen.getBoolType(self);
                try instruction_emit.emitOp(self, &self.function_section, .OpLogicalNot, &.{ bool_type, result_id, operand });
            },
            .bit_not => {
                const int_type = try type_codegen.getIntType(self, 32, true);
                try instruction_emit.emitOp(self, &self.function_section, .OpNot, &.{ int_type, result_id, operand });
            },
            // GLSL.std.450 math operations
            .sqrt => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 31), operand }); // Sqrt
            },
            .sin => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 13), operand }); // Sin
            },
            .cos => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 14), operand }); // Cos
            },
            .tan => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 15), operand }); // Tan
            },
            .asin => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 16), operand }); // Asin
            },
            .acos => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 17), operand }); // Acos
            },
            .atan => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 18), operand }); // Atan
            },
            .sinh => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 19), operand }); // Sinh
            },
            .cosh => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 20), operand }); // Cosh
            },
            .tanh => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 21), operand }); // Tanh
            },
            .exp => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 27), operand }); // Exp
            },
            .exp2 => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 29), operand }); // Exp2
            },
            .log => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 28), operand }); // Log
            },
            .log2 => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 30), operand }); // Log2
            },
            .log10 => {
                // GLSL.std.450 has no Log10; compute as Log2(x) / Log2(10)
                // Log2(10) ~ 3.321928... -> precompute as constant
                const log2_id = self.allocId();
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, log2_id, self.glsl_ext_id, @as(u32, 30), operand }); // Log2
                const log2_10 = try const_codegen.getConstantF32(self, 3.321928094887362);
                try instruction_emit.emitOp(self, &self.function_section, .OpFDiv, &.{ float_type, result_id, log2_id, log2_10 });
            },
            .abs => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 4), operand }); // FAbs
            },
            .floor => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 8), operand }); // Floor
            },
            .ceil => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 9), operand }); // Ceil
            },
            .round => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 1), operand }); // Round
            },
            .trunc => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 3), operand }); // Trunc
            },
            .fract => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 10), operand }); // Fract
            },
            .sign => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 6), operand }); // FSign
            },
            .normalize => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 69), operand }); // Normalize
            },
            .length => {
                try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, @as(u32, 66), operand }); // Length
            },
        }
        return result_id;
    }

    /// Classify a SPIR-V type ID into a type category for instruction selection.
    const TypeCategory = enum { float_, signed_int, unsigned_int, bool_, unknown };

    fn classifyType(self: *Self, type_id: u32) TypeCategory {
        // Check against known type IDs
        var it = self.type_ids.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* == type_id) {
                const key = entry.key_ptr.*;
                return switch (key.tag) {
                    .float => .float_,
                    .int => if (key.extra == 1) .signed_int else .unsigned_int,
                    .bool_ => .bool_,
                    else => .unknown,
                };
            }
        }
        return .unknown;
    }

    /// Generate a binary operation.
    /// Dispatches to the appropriate SPIR-V opcode based on operand type:
    /// - Float operands: OpF* instructions
    /// - Signed integer operands: OpS*/OpI* instructions
    /// - Unsigned integer operands: OpU*/OpI* instructions
    /// - Boolean operands: OpLogical* instructions
    pub fn generateBinaryOp(self: *Self, op: dsl_expr.BinaryOp, left: u32, right: u32) !u32 {
        const result_id = self.allocId();
        const bool_type = try type_codegen.getBoolType(self);

        // Determine operand type from left operand's type tracking
        const operand_type_id = self.ptr_type_map.get(left);
        const category: TypeCategory = if (operand_type_id) |tid|
            self.classifyType(tid)
        else
            .float_; // Default to float for backward compatibility

        // Select opcode based on type category
        const opcode: OpCode = switch (op) {
            .add => switch (category) {
                .signed_int, .unsigned_int => .OpIAdd,
                else => .OpFAdd,
            },
            .sub => switch (category) {
                .signed_int, .unsigned_int => .OpISub,
                else => .OpFSub,
            },
            .mul => switch (category) {
                .signed_int, .unsigned_int => .OpIMul,
                else => .OpFMul,
            },
            .div => switch (category) {
                .signed_int => .OpSDiv,
                .unsigned_int => .OpUDiv,
                else => .OpFDiv,
            },
            .mod => switch (category) {
                .signed_int => .OpSRem,
                .unsigned_int => .OpUMod,
                else => .OpFMod,
            },
            .eq => switch (category) {
                .signed_int, .unsigned_int => .OpIEqual,
                .bool_ => .OpLogicalEqual,
                else => .OpFOrdEqual,
            },
            .ne => switch (category) {
                .signed_int, .unsigned_int => .OpINotEqual,
                .bool_ => .OpLogicalNotEqual,
                else => .OpFOrdNotEqual,
            },
            .lt => switch (category) {
                .signed_int => .OpSLessThan,
                .unsigned_int => .OpULessThan,
                else => .OpFOrdLessThan,
            },
            .le => switch (category) {
                .signed_int => .OpSLessThanEqual,
                .unsigned_int => .OpULessThanEqual,
                else => .OpFOrdLessThanEqual,
            },
            .gt => switch (category) {
                .signed_int => .OpSGreaterThan,
                .unsigned_int => .OpUGreaterThan,
                else => .OpFOrdGreaterThan,
            },
            .ge => switch (category) {
                .signed_int => .OpSGreaterThanEqual,
                .unsigned_int => .OpUGreaterThanEqual,
                else => .OpFOrdGreaterThanEqual,
            },
            .and_ => .OpLogicalAnd,
            .or_ => .OpLogicalOr,
            .bit_and => .OpBitwiseAnd,
            .bit_or => .OpBitwiseOr,
            .bit_xor => .OpBitwiseXor,
            .shl => .OpShiftLeftLogical,
            .shr => switch (category) {
                .signed_int => .OpShiftRightArithmetic,
                else => .OpShiftRightLogical,
            },
            // Function-like binary ops dispatch to GLSL.std.450 ext inst
            .dot => return self.generateBinaryExtInst(40, left, right), // Dot (not in this switch)
            .cross => return self.generateBinaryExtInst(68, left, right), // Cross
            .min => switch (category) {
                .signed_int => return self.generateBinaryExtInst(39, left, right), // SMin
                .unsigned_int => return self.generateBinaryExtInst(38, left, right), // UMin
                else => return self.generateBinaryExtInst(37, left, right), // FMin
            },
            .max => switch (category) {
                .signed_int => return self.generateBinaryExtInst(42, left, right), // SMax
                .unsigned_int => return self.generateBinaryExtInst(41, left, right), // UMax
                else => return self.generateBinaryExtInst(40, left, right), // FMax
            },
            .pow => return self.generateBinaryExtInst(26, left, right), // Pow
            .atan2 => return self.generateBinaryExtInst(25, left, right), // Atan2
            .step => return self.generateBinaryExtInst(48, left, right), // Step
            .reflect => return self.generateBinaryExtInst(71, left, right), // Reflect
            .distance => return self.generateBinaryExtInst(67, left, right), // Distance
            .xor => .OpLogicalNotEqual, // Logical XOR is equivalent to !=
        };

        // Determine result type
        const result_type = if (op.isComparison()) bool_type else switch (category) {
            .signed_int => try type_codegen.getIntType(self, 32, true),
            .unsigned_int => try type_codegen.getIntType(self, 32, false),
            .bool_ => bool_type,
            else => try type_codegen.getFloatType(self, 32),
        };
        try instruction_emit.emitOp(self, &self.function_section, opcode, &.{ result_type, result_id, left, right });
        return result_id;
    }

    /// Generate a binary GLSL.std.450 extended instruction.
    fn generateBinaryExtInst(self: *Self, glsl_op: u32, left: u32, right: u32) !u32 {
        const result_id = self.allocId();
        const float_type = try type_codegen.getFloatType(self, 32);
        try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id, self.glsl_ext_id, glsl_op, left, right });
        return result_id;
    }

    /// Generate a function call.
    pub fn generateCall(self: *Self, c: dsl_expr.Expr.CallExpr) !u32 {
        switch (c.function) {
            .barrier => {
                const exec_scope = try const_codegen.getConstantU32(self, @intFromEnum(Scope.Workgroup));
                const mem_scope = try const_codegen.getConstantU32(self, @intFromEnum(Scope.Workgroup));
                const semantics = try const_codegen.getConstantU32(self, @intFromEnum(MemorySemantics.WorkgroupMemory) | @intFromEnum(MemorySemantics.AcquireRelease));
                try instruction_emit.emitOp(self, &self.function_section, .OpControlBarrier, &.{ exec_scope, mem_scope, semantics });
                return 0;
            },
            .memory_barrier => {
                const scope = try const_codegen.getConstantU32(self, @intFromEnum(Scope.Device));
                const semantics = try const_codegen.getConstantU32(self, @intFromEnum(MemorySemantics.AcquireRelease));
                try instruction_emit.emitOp(self, &self.function_section, .OpMemoryBarrier, &.{ scope, semantics });
                return 0;
            },
            .atomic_add => {
                if (c.args.len >= 2) {
                    const ptr = try self.generateExprPtr(c.args[0]);
                    const value = try self.generateExpr(c.args[1]);
                    const uint_type = try type_codegen.getIntType(self, 32, false);
                    const scope = try const_codegen.getConstantU32(self, @intFromEnum(Scope.Device));
                    const semantics_val = try const_codegen.getConstantU32(self, 0);
                    const result_id_inner = self.allocId();
                    try instruction_emit.emitOp(self, &self.function_section, .OpAtomicIAdd, &.{ uint_type, result_id_inner, ptr, scope, semantics_val, value });
                    return result_id_inner;
                }
                return 0;
            },
            .clamp => {
                if (c.args.len >= 3) {
                    const x = try self.generateExpr(c.args[0]);
                    const min_val = try self.generateExpr(c.args[1]);
                    const max_val = try self.generateExpr(c.args[2]);
                    const float_type = try type_codegen.getFloatType(self, 32);
                    const result_id_inner = self.allocId();
                    // GLSL.std.450 FClamp = 43
                    try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id_inner, self.glsl_ext_id, 43, x, min_val, max_val });
                    return result_id_inner;
                }
                return 0;
            },
            .mix => {
                if (c.args.len >= 3) {
                    const x = try self.generateExpr(c.args[0]);
                    const y = try self.generateExpr(c.args[1]);
                    const a = try self.generateExpr(c.args[2]);
                    const float_type = try type_codegen.getFloatType(self, 32);
                    const result_id_inner = self.allocId();
                    // GLSL.std.450 FMix = 46
                    try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id_inner, self.glsl_ext_id, 46, x, y, a });
                    return result_id_inner;
                }
                return 0;
            },
            .fma => {
                if (c.args.len >= 3) {
                    const a = try self.generateExpr(c.args[0]);
                    const b = try self.generateExpr(c.args[1]);
                    const cc = try self.generateExpr(c.args[2]);
                    const float_type = try type_codegen.getFloatType(self, 32);
                    const result_id_inner = self.allocId();
                    // GLSL.std.450 Fma = 50
                    try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id_inner, self.glsl_ext_id, 50, a, b, cc });
                    return result_id_inner;
                }
                return 0;
            },
            .smoothstep => {
                if (c.args.len >= 3) {
                    const edge0 = try self.generateExpr(c.args[0]);
                    const edge1 = try self.generateExpr(c.args[1]);
                    const x = try self.generateExpr(c.args[2]);
                    const float_type = try type_codegen.getFloatType(self, 32);
                    const result_id_inner = self.allocId();
                    // GLSL.std.450 SmoothStep = 49
                    try instruction_emit.emitOp(self, &self.function_section, .OpExtInst, &.{ float_type, result_id_inner, self.glsl_ext_id, 49, edge0, edge1, x });
                    return result_id_inner;
                }
                return 0;
            },
            else => return 0,
        }
    }

    /// Generate load from pointer.
    /// Resolves the pointed-to type from the pointer type map when available,
    /// falling back to f32 for backward compatibility.
    pub fn generateLoad(self: *Self, ptr: u32) !u32 {
        const result_id = self.allocId();
        const result_type = if (self.ptr_type_map.get(ptr)) |type_id|
            type_id
        else
            try type_codegen.getFloatType(self, 32);
        try instruction_emit.emitOp(self, &self.function_section, .OpLoad, &.{ result_type, result_id, ptr });
        return result_id;
    }

    /// Generate access chain.
    /// Resolves the element type from the buffer element type map when the base
    /// is a known buffer variable, falling back to f32 for backward compatibility.
    pub fn generateAccessChain(self: *Self, base: u32, index: u32) !u32 {
        const result_id = self.allocId();
        const elem_type = if (self.buffer_elem_type_map.get(base)) |type_id|
            type_id
        else
            try type_codegen.getFloatType(self, 32);
        const ptr_type = try type_codegen.getPointerType(self, elem_type, .StorageBuffer);
        try instruction_emit.emitOp(self, &self.function_section, .OpAccessChain, &.{ ptr_type, result_id, base, index });
        // Track that this access chain pointer points to elem_type
        try self.ptr_type_map.put(self.allocator, result_id, elem_type);
        return result_id;
    }

    /// Generate type cast.
    pub fn generateCast(self: *Self, value: u32, target_type: u32) !u32 {
        const result_id = self.allocId();
        try instruction_emit.emitOp(self, &self.function_section, .OpBitcast, &.{ target_type, result_id, value });
        return result_id;
    }
};

test {
    std.testing.refAllDecls(@This());
}
