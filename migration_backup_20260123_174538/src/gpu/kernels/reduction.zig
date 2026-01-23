//! Reduction GPU Kernel Definitions
//!
//! Pre-defined kernel IR for parallel reduction operations.
//! Uses tree-based parallel reduction in shared memory.
//!
//! ## Operations
//! - reduce_sum: Sum all elements
//! - reduce_max: Find maximum element
//! - reduce_min: Find minimum element
//! - reduce_product: Product of all elements

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

/// Build reduce_sum kernel (parallel reduction)
///
/// Uses tree-based parallel reduction in shared memory:
/// 1. Each thread loads one element into shared memory (0 if out of bounds)
/// 2. Barrier synchronization
/// 3. Tree reduction: for each power of 2 stride, accumulate pairs
/// 4. Thread 0 uses atomic add to accumulate to output
pub fn buildReduceSumKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "reduce_sum");
    errdefer builder.deinit();

    const BLOCK_SIZE: u32 = 256;
    _ = builder.setWorkgroupSize(BLOCK_SIZE, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .read_write);
    const n = try builder.addUniform("n", Type.u32Type());

    // Add shared memory for workgroup reduction
    const shared = try builder.addSharedMemory("shared_data", Type.f32Type(), BLOCK_SIZE);

    const gid = builder.globalInvocationId();
    const lid = builder.localInvocationId();
    const idx = try gid.x();
    const local_idx = try lid.x();

    // Load input[idx] if in bounds, else 0
    const in_bounds = try builder.lt(idx, try n.toExpr());
    const input_val = try input.at(idx);
    const zero = try builder.f32Lit(0.0);
    const load_val = try builder.select(in_bounds, input_val, zero);

    // Store to shared memory
    const shared_ptr = try shared.at(local_idx);
    const store_to_shared = try builder.assignStmt(shared_ptr, load_val);
    try builder.statements.append(allocator, store_to_shared);

    // Barrier to ensure all loads are complete
    try builder.barrier();

    // Tree reduction with unrolled loop (common GPU pattern)
    // Each iteration halves the active threads
    const strides = [_]u32{ 128, 64, 32, 16, 8, 4, 2, 1 };
    for (strides) |stride| {
        const stride_lit = try builder.u32Lit(stride);
        const active = try builder.lt(local_idx, stride_lit);

        // shared[local_idx] += shared[local_idx + stride]
        const partner_idx = try builder.add(local_idx, stride_lit);
        const partner_val = try shared.at(partner_idx);
        const current_val = try shared.at(local_idx);
        const sum = try builder.add(current_val, partner_val);
        const update_shared = try builder.assignStmt(shared_ptr, sum);

        try builder.ifStmt(active, &[_]*const dsl.Stmt{update_shared}, null);
        try builder.barrier();
    }

    // Thread 0 writes result using atomic add to support multiple workgroups
    const is_thread_zero = try builder.eq(local_idx, try builder.u32Lit(0));
    const output_idx = try builder.u32Lit(0);
    const output_ptr = try output.at(output_idx);
    const final_val = try shared.at(try builder.u32Lit(0));

    // Use atomic add for multi-workgroup reduction
    const atomic_add_expr = try builder.call(.atomic_add, &.{ output_ptr, final_val });
    const atomic_stmt = try dsl.exprStmt(allocator, atomic_add_expr);
    try builder.ifStmt(is_thread_zero, &[_]*const dsl.Stmt{atomic_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build reduce_max kernel (parallel reduction)
///
/// Uses tree-based parallel reduction with max operation.
pub fn buildReduceMaxKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "reduce_max");
    errdefer builder.deinit();

    const BLOCK_SIZE: u32 = 256;
    _ = builder.setWorkgroupSize(BLOCK_SIZE, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .read_write);
    const n = try builder.addUniform("n", Type.u32Type());

    const shared = try builder.addSharedMemory("shared_data", Type.f32Type(), BLOCK_SIZE);

    const gid = builder.globalInvocationId();
    const lid = builder.localInvocationId();
    const idx = try gid.x();
    const local_idx = try lid.x();

    // Load input[idx] if in bounds, else -infinity
    const in_bounds = try builder.lt(idx, try n.toExpr());
    const input_val = try input.at(idx);
    const neg_inf = try builder.f32Lit(-3.4028235e+38); // -FLT_MAX
    const load_val = try builder.select(in_bounds, input_val, neg_inf);

    // Store to shared memory
    const shared_ptr = try shared.at(local_idx);
    const store_to_shared = try builder.assignStmt(shared_ptr, load_val);
    try builder.statements.append(allocator, store_to_shared);

    try builder.barrier();

    // Tree reduction with max
    const strides = [_]u32{ 128, 64, 32, 16, 8, 4, 2, 1 };
    for (strides) |stride| {
        const stride_lit = try builder.u32Lit(stride);
        const active = try builder.lt(local_idx, stride_lit);

        const partner_idx = try builder.add(local_idx, stride_lit);
        const partner_val = try shared.at(partner_idx);
        const current_val = try shared.at(local_idx);

        // max(current, partner)
        const is_greater = try builder.gt(current_val, partner_val);
        const max_val = try builder.select(is_greater, current_val, partner_val);
        const update_shared = try builder.assignStmt(shared_ptr, max_val);

        try builder.ifStmt(active, &[_]*const dsl.Stmt{update_shared}, null);
        try builder.barrier();
    }

    // Thread 0 writes result using atomic max
    const is_thread_zero = try builder.eq(local_idx, try builder.u32Lit(0));
    const output_idx = try builder.u32Lit(0);
    const output_ptr = try output.at(output_idx);
    const final_val = try shared.at(try builder.u32Lit(0));

    const atomic_max_expr = try builder.call(.atomic_max, &.{ output_ptr, final_val });
    const atomic_stmt = try dsl.exprStmt(allocator, atomic_max_expr);
    try builder.ifStmt(is_thread_zero, &[_]*const dsl.Stmt{atomic_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build reduce_min kernel (parallel reduction)
///
/// Uses tree-based parallel reduction with min operation.
pub fn buildReduceMinKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "reduce_min");
    errdefer builder.deinit();

    const BLOCK_SIZE: u32 = 256;
    _ = builder.setWorkgroupSize(BLOCK_SIZE, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .read_write);
    const n = try builder.addUniform("n", Type.u32Type());

    const shared = try builder.addSharedMemory("shared_data", Type.f32Type(), BLOCK_SIZE);

    const gid = builder.globalInvocationId();
    const lid = builder.localInvocationId();
    const idx = try gid.x();
    const local_idx = try lid.x();

    // Load input[idx] if in bounds, else +infinity
    const in_bounds = try builder.lt(idx, try n.toExpr());
    const input_val = try input.at(idx);
    const pos_inf = try builder.f32Lit(3.4028235e+38); // FLT_MAX
    const load_val = try builder.select(in_bounds, input_val, pos_inf);

    // Store to shared memory
    const shared_ptr = try shared.at(local_idx);
    const store_to_shared = try builder.assignStmt(shared_ptr, load_val);
    try builder.statements.append(allocator, store_to_shared);

    try builder.barrier();

    // Tree reduction with min
    const strides = [_]u32{ 128, 64, 32, 16, 8, 4, 2, 1 };
    for (strides) |stride| {
        const stride_lit = try builder.u32Lit(stride);
        const active = try builder.lt(local_idx, stride_lit);

        const partner_idx = try builder.add(local_idx, stride_lit);
        const partner_val = try shared.at(partner_idx);
        const current_val = try shared.at(local_idx);

        // min(current, partner)
        const is_less = try builder.lt(current_val, partner_val);
        const min_val = try builder.select(is_less, current_val, partner_val);
        const update_shared = try builder.assignStmt(shared_ptr, min_val);

        try builder.ifStmt(active, &[_]*const dsl.Stmt{update_shared}, null);
        try builder.barrier();
    }

    // Thread 0 writes result using atomic min
    const is_thread_zero = try builder.eq(local_idx, try builder.u32Lit(0));
    const output_idx = try builder.u32Lit(0);
    const output_ptr = try output.at(output_idx);
    const final_val = try shared.at(try builder.u32Lit(0));

    const atomic_min_expr = try builder.call(.atomic_min, &.{ output_ptr, final_val });
    const atomic_stmt = try dsl.exprStmt(allocator, atomic_min_expr);
    try builder.ifStmt(is_thread_zero, &[_]*const dsl.Stmt{atomic_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build reduce_product kernel (parallel reduction)
///
/// Uses tree-based parallel reduction with multiplication.
/// Note: No atomic_mul exists, so this kernel is designed for single-workgroup use.
/// For multi-workgroup reduction, use multiple kernel launches.
pub fn buildReduceProductKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "reduce_product");
    errdefer builder.deinit();

    const BLOCK_SIZE: u32 = 256;
    _ = builder.setWorkgroupSize(BLOCK_SIZE, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const shared = try builder.addSharedMemory("shared_data", Type.f32Type(), BLOCK_SIZE);

    const gid = builder.globalInvocationId();
    const lid = builder.localInvocationId();
    const idx = try gid.x();
    const local_idx = try lid.x();

    // Load input[idx] if in bounds, else 1 (identity for multiplication)
    const in_bounds = try builder.lt(idx, try n.toExpr());
    const input_val = try input.at(idx);
    const one = try builder.f32Lit(1.0);
    const load_val = try builder.select(in_bounds, input_val, one);

    // Store to shared memory
    const shared_ptr = try shared.at(local_idx);
    const store_to_shared = try builder.assignStmt(shared_ptr, load_val);
    try builder.statements.append(allocator, store_to_shared);

    try builder.barrier();

    // Tree reduction with multiplication
    const strides = [_]u32{ 128, 64, 32, 16, 8, 4, 2, 1 };
    for (strides) |stride| {
        const stride_lit = try builder.u32Lit(stride);
        const active = try builder.lt(local_idx, stride_lit);

        const partner_idx = try builder.add(local_idx, stride_lit);
        const partner_val = try shared.at(partner_idx);
        const current_val = try shared.at(local_idx);
        const product = try builder.mul(current_val, partner_val);
        const update_shared = try builder.assignStmt(shared_ptr, product);

        try builder.ifStmt(active, &[_]*const dsl.Stmt{update_shared}, null);
        try builder.barrier();
    }

    // Thread 0 writes final result
    const is_thread_zero = try builder.eq(local_idx, try builder.u32Lit(0));
    const output_idx = try builder.u32Lit(0);
    const output_ptr = try output.at(output_idx);
    const final_val = try shared.at(try builder.u32Lit(0));
    const store_result = try builder.assignStmt(output_ptr, final_val);

    try builder.ifStmt(is_thread_zero, &[_]*const dsl.Stmt{store_result}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}
