//! GPU Kernel IR Definition
//!
//! Defines the complete kernel intermediate representation that combines
//! types, expressions, and statements into a compilable kernel.

const std = @import("std");
const types = @import("types.zig");
const expr = @import("expr.zig");
const stmt = @import("stmt.zig");

const ExprSet = std.AutoHashMapUnmanaged(*const expr.Expr, void);
const StmtSet = std.AutoHashMapUnmanaged(*const stmt.Stmt, void);

/// Key for tracking binding uniqueness.
const BindingKey = struct {
    group: u32,
    binding: u32,
};

/// Buffer binding descriptor.
/// Describes a storage buffer parameter to the kernel.
pub const BufferBinding = struct {
    /// Parameter name.
    name: []const u8,
    /// Binding index.
    binding: u32,
    /// Bind group (default 0).
    group: u32 = 0,
    /// Element type of the buffer.
    element_type: types.Type,
    /// Access mode (read, write, or read_write).
    access: types.AccessMode,
};

/// Uniform/constant binding.
/// Describes a uniform or push constant parameter.
pub const UniformBinding = struct {
    /// Parameter name.
    name: []const u8,
    /// Binding index.
    binding: u32,
    /// Bind group (default 0).
    group: u32 = 0,
    /// Type of the uniform.
    ty: types.Type,
};

/// Push constant binding (Vulkan-style).
/// Small, fast-access data passed directly in commands.
pub const PushConstant = struct {
    /// Parameter name.
    name: []const u8,
    /// Type of the push constant.
    ty: types.Type,
    /// Offset in bytes.
    offset: u32 = 0,
};

/// Workgroup (shared) memory declaration.
/// Shared memory is fast but limited, shared within a workgroup.
pub const SharedMemory = struct {
    /// Variable name.
    name: []const u8,
    /// Element type.
    element_type: types.Type,
    /// Size of the array. null means runtime-sized (dynamic shared memory).
    size: ?usize,
};

/// Helper function definition for use within kernels.
pub const HelperFunction = struct {
    /// Function name.
    name: []const u8,
    /// Parameters.
    params: []const Param,
    /// Return type (null for void).
    return_type: ?types.Type,
    /// Function body statements.
    body: []const *const stmt.Stmt,

    pub const Param = struct {
        /// Parameter name.
        name: []const u8,
        /// Parameter type.
        ty: types.Type,
    };
};

/// Complete kernel intermediate representation.
/// Contains all information needed to generate backend-specific code.
pub const KernelIR = struct {
    /// Kernel name (used for identification and debugging).
    name: []const u8,

    /// Entry point function name (usually "main").
    entry_point: []const u8,

    /// Workgroup/block size (x, y, z).
    workgroup_size: [3]u32,

    /// Storage buffer bindings.
    buffers: []const BufferBinding,

    /// Uniform bindings.
    uniforms: []const UniformBinding,

    /// Push constants (Vulkan-style).
    push_constants: []const PushConstant,

    /// Shared memory declarations.
    shared_memory: []const SharedMemory,

    /// Kernel body statements.
    body: []const *const stmt.Stmt,

    /// Helper functions (inlined or called from body).
    functions: []const HelperFunction,

    /// Required feature flags (e.g., subgroups, fp16).
    required_features: FeatureFlags = .{},

    /// Create an empty kernel IR.
    pub fn empty(kernel_name: []const u8) KernelIR {
        return .{
            .name = kernel_name,
            .entry_point = "main",
            .workgroup_size = .{ 256, 1, 1 },
            .buffers = &.{},
            .uniforms = &.{},
            .push_constants = &.{},
            .shared_memory = &.{},
            .body = &.{},
            .functions = &.{},
            .required_features = .{},
        };
    }

    /// Get the total number of bindings (buffers + uniforms).
    pub fn bindingCount(self: KernelIR) usize {
        return self.buffers.len + self.uniforms.len;
    }

    /// Get the total shared memory size in bytes.
    pub fn sharedMemoryBytes(self: KernelIR) ?usize {
        var total: usize = 0;
        for (self.shared_memory) |shared| {
            if (shared.size) |size| {
                const elem_size = switch (shared.element_type) {
                    .scalar => |s| @as(usize, s.byteSize()),
                    .vector => |v| v.byteSize(),
                    else => 4, // Default to 4 bytes
                };
                total += size * elem_size;
            } else {
                // Runtime-sized shared memory means we can't determine total
                return null;
            }
        }
        return total;
    }

    /// Get total workgroup size (x * y * z).
    pub fn totalWorkgroupSize(self: KernelIR) u32 {
        return self.workgroup_size[0] * self.workgroup_size[1] * self.workgroup_size[2];
    }

    /// Validate the kernel IR for common issues.
    pub fn validate(self: KernelIR) ValidationResult {
        var result = ValidationResult{};

        // Check workgroup size
        const total_size = self.totalWorkgroupSize();
        if (total_size == 0) {
            result.errors.workgroup_size_zero = true;
        }
        if (total_size > 1024) {
            result.warnings.workgroup_size_large = true;
        }

        // Check for duplicate binding indices
        var seen_bindings: std.AutoHashMapUnmanaged(BindingKey, void) = .empty;
        defer seen_bindings.deinit(std.heap.page_allocator);

        for (self.buffers) |buf| {
            const key = BindingKey{ .group = buf.group, .binding = buf.binding };
            if (seen_bindings.contains(key)) {
                result.errors.duplicate_bindings = true;
            } else {
                seen_bindings.put(std.heap.page_allocator, key, {}) catch {
                    result.errors.memory_error = true;
                };
            }
        }

        for (self.uniforms) |uni| {
            const key = BindingKey{ .group = uni.group, .binding = uni.binding };
            if (seen_bindings.contains(key)) {
                result.errors.duplicate_bindings = true;
            } else {
                seen_bindings.put(std.heap.page_allocator, key, {}) catch {
                    result.errors.memory_error = true;
                };
            }
        }

        // Check name validity
        if (self.name.len == 0) {
            result.errors.empty_name = true;
        }

        return result;
    }

    /// Deinitialize kernel IR allocations.
    /// This does not free name strings, which are treated as external.
    pub fn deinit(self: *const KernelIR, allocator: std.mem.Allocator) void {
        var expr_seen: ExprSet = .empty;
        defer expr_seen.deinit(allocator);
        var stmt_seen: StmtSet = .empty;
        defer stmt_seen.deinit(allocator);

        for (self.body) |statement| {
            deinitStmt(allocator, statement, &stmt_seen, &expr_seen);
        }

        for (self.functions) |func| {
            for (func.body) |statement| {
                deinitStmt(allocator, statement, &stmt_seen, &expr_seen);
            }
            if (func.body.len > 0) {
                allocator.free(func.body);
            }
            if (func.params.len > 0) {
                allocator.free(func.params);
            }
        }

        if (self.body.len > 0) {
            allocator.free(self.body);
        }
        if (self.functions.len > 0) {
            allocator.free(self.functions);
        }
        if (self.buffers.len > 0) {
            allocator.free(self.buffers);
        }
        if (self.uniforms.len > 0) {
            allocator.free(self.uniforms);
        }
        if (self.push_constants.len > 0) {
            allocator.free(self.push_constants);
        }
        if (self.shared_memory.len > 0) {
            allocator.free(self.shared_memory);
        }
    }
};

fn deinitStmt(
    allocator: std.mem.Allocator,
    node: *const stmt.Stmt,
    stmt_seen: *StmtSet,
    expr_seen: *ExprSet,
) void {
    if (stmt_seen.contains(node)) {
        return;
    }
    stmt_seen.put(allocator, node, {}) catch return;

    switch (node.*) {
        .var_decl => |decl| {
            if (decl.init) |init_expr| {
                deinitExpr(allocator, init_expr, expr_seen);
            }
        },
        .assign => |assign| {
            deinitExpr(allocator, assign.target, expr_seen);
            deinitExpr(allocator, assign.value, expr_seen);
        },
        .compound_assign => |assign| {
            deinitExpr(allocator, assign.target, expr_seen);
            deinitExpr(allocator, assign.value, expr_seen);
        },
        .if_ => |if_stmt| {
            deinitExpr(allocator, if_stmt.condition, expr_seen);
            for (if_stmt.then_body) |statement| {
                deinitStmt(allocator, statement, stmt_seen, expr_seen);
            }
            if (if_stmt.then_body.len > 0) {
                allocator.free(if_stmt.then_body);
            }
            if (if_stmt.else_body) |else_body| {
                for (else_body) |statement| {
                    deinitStmt(allocator, statement, stmt_seen, expr_seen);
                }
                if (else_body.len > 0) {
                    allocator.free(else_body);
                }
            }
        },
        .for_ => |for_stmt| {
            if (for_stmt.init) |init_stmt| {
                deinitStmt(allocator, init_stmt, stmt_seen, expr_seen);
            }
            if (for_stmt.condition) |condition| {
                deinitExpr(allocator, condition, expr_seen);
            }
            if (for_stmt.update) |update_stmt| {
                deinitStmt(allocator, update_stmt, stmt_seen, expr_seen);
            }
            for (for_stmt.body) |statement| {
                deinitStmt(allocator, statement, stmt_seen, expr_seen);
            }
            if (for_stmt.body.len > 0) {
                allocator.free(for_stmt.body);
            }
        },
        .while_ => |while_stmt| {
            deinitExpr(allocator, while_stmt.condition, expr_seen);
            for (while_stmt.body) |statement| {
                deinitStmt(allocator, statement, stmt_seen, expr_seen);
            }
            if (while_stmt.body.len > 0) {
                allocator.free(while_stmt.body);
            }
        },
        .do_while => |do_while| {
            for (do_while.body) |statement| {
                deinitStmt(allocator, statement, stmt_seen, expr_seen);
            }
            if (do_while.body.len > 0) {
                allocator.free(do_while.body);
            }
            deinitExpr(allocator, do_while.condition, expr_seen);
        },
        .return_ => |ret| {
            if (ret.value) |value| {
                deinitExpr(allocator, value, expr_seen);
            }
        },
        .break_, .continue_, .discard => {},
        .expr_stmt => |expression| {
            deinitExpr(allocator, expression, expr_seen);
        },
        .block => |block| {
            for (block.statements) |statement| {
                deinitStmt(allocator, statement, stmt_seen, expr_seen);
            }
            if (block.statements.len > 0) {
                allocator.free(block.statements);
            }
        },
        .switch_ => |switch_stmt| {
            deinitExpr(allocator, switch_stmt.selector, expr_seen);
            for (switch_stmt.cases) |case_item| {
                for (case_item.body) |statement| {
                    deinitStmt(allocator, statement, stmt_seen, expr_seen);
                }
                if (case_item.body.len > 0) {
                    allocator.free(case_item.body);
                }
            }
            if (switch_stmt.default) |default_body| {
                for (default_body) |statement| {
                    deinitStmt(allocator, statement, stmt_seen, expr_seen);
                }
                if (default_body.len > 0) {
                    allocator.free(default_body);
                }
            }
            if (switch_stmt.cases.len > 0) {
                allocator.free(switch_stmt.cases);
            }
        },
    }

    allocator.destroy(@constCast(node));
}

fn deinitExpr(
    allocator: std.mem.Allocator,
    node: *const expr.Expr,
    expr_seen: *ExprSet,
) void {
    if (expr_seen.contains(node)) {
        return;
    }
    expr_seen.put(allocator, node, {}) catch return;

    switch (node.*) {
        .literal, .ref => {},
        .unary => |unary| {
            deinitExpr(allocator, unary.operand, expr_seen);
        },
        .binary => |binary| {
            deinitExpr(allocator, binary.left, expr_seen);
            deinitExpr(allocator, binary.right, expr_seen);
        },
        .call => |call| {
            for (call.args) |arg| {
                deinitExpr(allocator, arg, expr_seen);
            }
            if (call.args.len > 0) {
                allocator.free(call.args);
            }
        },
        .vector_construct => |vector_construct| {
            for (vector_construct.components) |component| {
                deinitExpr(allocator, component, expr_seen);
            }
            if (vector_construct.components.len > 0) {
                allocator.free(vector_construct.components);
            }
        },
        .index => |index_expr| {
            deinitExpr(allocator, index_expr.base, expr_seen);
            deinitExpr(allocator, index_expr.index, expr_seen);
        },
        .field => |field_expr| {
            deinitExpr(allocator, field_expr.base, expr_seen);
        },
        .cast => |cast_expr| {
            deinitExpr(allocator, cast_expr.operand, expr_seen);
        },
        .select => |select_expr| {
            deinitExpr(allocator, select_expr.condition, expr_seen);
            deinitExpr(allocator, select_expr.true_value, expr_seen);
            deinitExpr(allocator, select_expr.false_value, expr_seen);
        },
        .swizzle => |swizzle| {
            deinitExpr(allocator, swizzle.base, expr_seen);
            if (swizzle.components.len > 0) {
                allocator.free(swizzle.components);
            }
        },
    }

    allocator.destroy(@constCast(node));
}

/// Feature flags for kernel requirements.
pub const FeatureFlags = packed struct {
    /// Requires subgroup operations.
    subgroups: bool = false,
    /// Requires 16-bit float support.
    fp16: bool = false,
    /// Requires 64-bit float support.
    fp64: bool = false,
    /// Requires 8-bit integer support.
    int8: bool = false,
    /// Requires 16-bit integer support.
    int16: bool = false,
    /// Requires 64-bit integer support.
    int64: bool = false,
    /// Requires atomic operations.
    atomics: bool = false,
    /// Requires dynamic shared memory.
    dynamic_shared_memory: bool = false,
    /// Requires mesh shader support (Metal 3+ / Vulkan).
    mesh_shaders: bool = false,
    /// Requires ray tracing support (Metal 3+ / Vulkan RT).
    ray_tracing: bool = false,
    /// Reserved bits for future use.
    _reserved: u6 = 0,
};

/// Result of kernel IR validation.
pub const ValidationResult = struct {
    errors: Errors = .{},
    warnings: Warnings = .{},

    pub const Errors = struct {
        workgroup_size_zero: bool = false,
        duplicate_bindings: bool = false,
        empty_name: bool = false,
        memory_error: bool = false,
    };

    pub const Warnings = struct {
        workgroup_size_large: bool = false,
    };

    pub fn hasErrors(self: ValidationResult) bool {
        return self.errors.workgroup_size_zero or
            self.errors.duplicate_bindings or
            self.errors.empty_name or
            self.errors.memory_error;
    }

    pub fn hasWarnings(self: ValidationResult) bool {
        return self.warnings.workgroup_size_large;
    }

    pub fn isValid(self: ValidationResult) bool {
        return !self.hasErrors();
    }
};

/// Portable kernel source representation.
/// This is the user-facing type for specifying kernels.
pub const PortableKernelSource = struct {
    /// Kernel name.
    name: []const u8,
    /// Pre-built kernel IR (if available).
    ir: ?*const KernelIR = null,
    /// Portable source code (if using text DSL).
    source: ?[]const u8 = null,
    /// Entry point function name.
    entry_point: []const u8 = "main",
    /// Workgroup size.
    workgroup_size: [3]u32 = .{ 256, 1, 1 },
};

// ============================================================================
// Builtin Kernel Templates
// ============================================================================

/// Built-in kernel operation types.
pub const BuiltinKernel = enum {
    // Element-wise operations
    vector_add,
    vector_sub,
    vector_mul,
    vector_div,
    vector_scale,

    // Matrix operations
    matrix_multiply,
    matrix_transpose,

    // Reduction operations
    reduce_sum,
    reduce_max,
    reduce_min,
    reduce_product,

    // Basic activations
    softmax,
    relu,
    sigmoid,
    tanh,

    // Linear algebra
    dot_product,
    normalize,
    saxpy, // a*x + y
    copy,
    fill,

    // Neural network activations (new)
    gelu, // GELU(x) = x * Φ(x) ≈ 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))
    gelu_fast, // Fast approximation: x * sigmoid(1.702 * x)
    swiglu, // SwiGLU(x, gate) = x * silu(gate) for LLaMA/Mixtral
    silu, // SiLU(x) = x * sigmoid(x), also called Swish

    // Normalization layers (new)
    layer_norm, // LayerNorm: (x - μ) / √(σ² + ε) * γ + β
    rms_norm, // RMSNorm: x / √(mean(x²) + ε) * γ
    batch_norm, // BatchNorm for vision models

    // Fused operations (new)
    fused_add_norm, // residual + layer_norm
    fused_linear_gelu, // xW + b, then GELU

    // Batch operations (new)
    batch_matmul, // Batched matrix multiply
    batch_cosine_similarity, // Batch cosine similarity for embeddings

    // Vision operations (new)
    conv2d, // 2D convolution using im2col + GEMM
    max_pool2d, // Max pooling with indices for backward pass
    avg_pool2d, // Average pooling
    batch_norm2d, // Batch normalization for 2D (vision)
    im2col, // Image to column transform (conv helper)
    col2im, // Column to image transform (backward helper)

    /// Returns the minimum number of buffer bindings required.
    pub fn minBufferCount(self: BuiltinKernel) u8 {
        return switch (self) {
            .copy, .fill => 1,
            .vector_scale, .reduce_sum, .reduce_max, .reduce_min, .reduce_product => 2,
            .vector_add, .vector_sub, .vector_mul, .vector_div => 3,
            .softmax, .relu, .sigmoid, .tanh, .normalize => 2,
            .dot_product => 3,
            .saxpy => 3,
            .matrix_multiply, .matrix_transpose => 3,
            // NN kernels
            .gelu, .gelu_fast, .silu => 2,
            .swiglu => 3, // input, gate, output
            .layer_norm, .rms_norm => 4, // input, gamma, beta, output
            .batch_norm => 5, // input, gamma, beta, mean, var, output
            .fused_add_norm => 5, // input, residual, gamma, beta, output
            .fused_linear_gelu => 4, // input, weight, bias, output
            .batch_matmul => 3, // a, b, c
            .batch_cosine_similarity => 3, // query, vectors, results
            // Vision kernels
            .conv2d => 4, // input, weights, bias, output
            .max_pool2d => 3, // input, output, indices
            .avg_pool2d => 2, // input, output
            .batch_norm2d => 6, // input, gamma, beta, running_mean, running_var, output
            .im2col => 2, // input, col_output
            .col2im => 2, // col_input, output
        };
    }

    /// Returns true if this kernel requires reduction operations.
    pub fn isReduction(self: BuiltinKernel) bool {
        return switch (self) {
            .reduce_sum, .reduce_max, .reduce_min, .reduce_product, .dot_product => true,
            .layer_norm, .rms_norm, .batch_norm => true, // Need mean/variance reduction
            .max_pool2d, .avg_pool2d, .batch_norm2d => true, // Vision pooling/norm needs reduction
            else => false,
        };
    }

    /// Returns true if this is a neural network kernel.
    pub fn isNeuralNetwork(self: BuiltinKernel) bool {
        return switch (self) {
            .gelu, .gelu_fast, .swiglu, .silu => true,
            .layer_norm, .rms_norm, .batch_norm => true,
            .fused_add_norm, .fused_linear_gelu => true,
            .batch_matmul, .batch_cosine_similarity => true,
            .softmax, .relu, .sigmoid, .tanh => true,
            // Vision kernels
            .conv2d, .max_pool2d, .avg_pool2d, .batch_norm2d => true,
            .im2col, .col2im => true,
            else => false,
        };
    }

    /// Returns true if this is a vision/CNN kernel.
    pub fn isVision(self: BuiltinKernel) bool {
        return switch (self) {
            .conv2d, .max_pool2d, .avg_pool2d, .batch_norm2d => true,
            .im2col, .col2im => true,
            else => false,
        };
    }

    /// Returns the name of this built-in kernel.
    pub fn name(self: BuiltinKernel) []const u8 {
        return switch (self) {
            .vector_add => "vector_add",
            .vector_sub => "vector_sub",
            .vector_mul => "vector_mul",
            .vector_div => "vector_div",
            .vector_scale => "vector_scale",
            .matrix_multiply => "matrix_multiply",
            .matrix_transpose => "matrix_transpose",
            .reduce_sum => "reduce_sum",
            .reduce_max => "reduce_max",
            .reduce_min => "reduce_min",
            .reduce_product => "reduce_product",
            .softmax => "softmax",
            .relu => "relu",
            .sigmoid => "sigmoid",
            .tanh => "tanh",
            .dot_product => "dot_product",
            .normalize => "normalize",
            .saxpy => "saxpy",
            .copy => "copy",
            .fill => "fill",
            // NN kernels
            .gelu => "gelu",
            .gelu_fast => "gelu_fast",
            .swiglu => "swiglu",
            .silu => "silu",
            .layer_norm => "layer_norm",
            .rms_norm => "rms_norm",
            .batch_norm => "batch_norm",
            .fused_add_norm => "fused_add_norm",
            .fused_linear_gelu => "fused_linear_gelu",
            .batch_matmul => "batch_matmul",
            .batch_cosine_similarity => "batch_cosine_similarity",
            // Vision kernels
            .conv2d => "conv2d",
            .max_pool2d => "max_pool2d",
            .avg_pool2d => "avg_pool2d",
            .batch_norm2d => "batch_norm2d",
            .im2col => "im2col",
            .col2im => "col2im",
        };
    }

    /// Reverse lookup: find the BuiltinKernel enum value from a kernel name string.
    /// Returns null if the name doesn't match any built-in kernel.
    pub fn fromName(kernel_name: []const u8) ?BuiltinKernel {
        const fields = @typeInfo(BuiltinKernel).@"enum".fields;
        inline for (fields) |field| {
            const variant: BuiltinKernel = @enumFromInt(field.value);
            if (std.mem.eql(u8, kernel_name, variant.name())) {
                return variant;
            }
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "KernelIR.empty" {
    const ir = KernelIR.empty("test_kernel");
    try std.testing.expectEqualStrings("test_kernel", ir.name);
    try std.testing.expectEqualStrings("main", ir.entry_point);
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(usize, 0), ir.buffers.len);
}

test "KernelIR.totalWorkgroupSize" {
    var ir = KernelIR.empty("test");
    ir.workgroup_size = .{ 16, 16, 4 };
    try std.testing.expectEqual(@as(u32, 1024), ir.totalWorkgroupSize());
}

test "KernelIR.validate" {
    const valid_ir = KernelIR.empty("test_kernel");
    const result = valid_ir.validate();
    try std.testing.expect(result.isValid());
    try std.testing.expect(!result.hasWarnings());

    var invalid_ir = KernelIR.empty("");
    invalid_ir.workgroup_size = .{ 0, 1, 1 };
    const invalid_result = invalid_ir.validate();
    try std.testing.expect(!invalid_result.isValid());
    try std.testing.expect(invalid_result.errors.empty_name);
    try std.testing.expect(invalid_result.errors.workgroup_size_zero);
}

test "BuiltinKernel.minBufferCount" {
    try std.testing.expectEqual(@as(u8, 3), BuiltinKernel.vector_add.minBufferCount());
    try std.testing.expectEqual(@as(u8, 2), BuiltinKernel.reduce_sum.minBufferCount());
    try std.testing.expectEqual(@as(u8, 1), BuiltinKernel.copy.minBufferCount());
}

test "BuiltinKernel.isReduction" {
    try std.testing.expect(BuiltinKernel.reduce_sum.isReduction());
    try std.testing.expect(BuiltinKernel.dot_product.isReduction());
    try std.testing.expect(!BuiltinKernel.vector_add.isReduction());
    try std.testing.expect(!BuiltinKernel.matrix_multiply.isReduction());
}
