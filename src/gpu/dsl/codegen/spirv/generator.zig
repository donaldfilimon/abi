//! SPIR-V Generator Core
//!
//! Main SpirvGenerator struct and the generate() method that produces
//! SPIR-V binary bytecode from kernel IR.

const std = @import("std");
const constants = @import("constants.zig");
const types_gen = @import("types_gen.zig");
const constants_gen = @import("constants_gen.zig");
const emit = @import("emit.zig");
const codegen = @import("codegen.zig");

const dsl_types = @import("../../types.zig");
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

    // Mix in type generation methods
    pub usingnamespace types_gen.TypeGenMixin(Self);
    // Mix in constant generation methods
    pub usingnamespace constants_gen.ConstGenMixin(Self);
    // Mix in instruction emission methods
    pub usingnamespace emit.EmitMixin(Self);
    // Mix in code generation methods
    pub usingnamespace codegen.CodeGenMixin(Self);

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
};
