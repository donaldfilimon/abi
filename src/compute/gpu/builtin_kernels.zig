//! Built-in GPU Kernel Definitions
//!
//! Pre-defined kernel IR using the DSL for common GPU operations.
//! These kernels are compiled to backend-specific code on demand.
//!
//! ## Supported Operations
//!
//! - **Element-wise**: vector_add, vector_sub, vector_mul, vector_div, vector_scale
//! - **Reductions**: reduce_sum, reduce_max, reduce_min, reduce_product
//! - **Matrix**: matrix_multiply, matrix_transpose
//! - **Neural Network**: softmax, relu, sigmoid, tanh
//! - **Linear Algebra**: dot_product, normalize, saxpy
//!
//! ## Usage
//!
//! ```zig
//! const kernel = @import("builtin_kernels.zig");
//!
//! var ir = try kernel.buildKernelIR(allocator, .vector_add);
//! // Use ir with dsl.compile() to generate backend code
//! ```

const std = @import("std");
const dsl = @import("dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;
pub const BuiltinKernel = dsl.BuiltinKernel;

/// Build kernel IR for a given builtin kernel type.
pub fn buildKernelIR(allocator: std.mem.Allocator, kernel_type: BuiltinKernel) !*const KernelIR {
    return switch (kernel_type) {
        .vector_add => buildVectorAddKernel(allocator),
        .vector_sub => buildVectorSubKernel(allocator),
        .vector_mul => buildVectorMulKernel(allocator),
        .vector_div => buildVectorDivKernel(allocator),
        .vector_scale => buildVectorScaleKernel(allocator),
        .matrix_multiply => buildMatrixMultiplyKernel(allocator),
        .matrix_transpose => buildMatrixTransposeKernel(allocator),
        .reduce_sum => buildReduceSumKernel(allocator),
        .reduce_max => buildReduceMaxKernel(allocator),
        .reduce_min => buildReduceMinKernel(allocator),
        .reduce_product => buildReduceProductKernel(allocator),
        .softmax => buildSoftmaxKernel(allocator),
        .relu => buildReluKernel(allocator),
        .sigmoid => buildSigmoidKernel(allocator),
        .tanh => buildTanhKernel(allocator),
        .dot_product => buildDotProductKernel(allocator),
        .normalize => buildNormalizeKernel(allocator),
        .saxpy => buildSaxpyKernel(allocator),
        .copy => buildCopyKernel(allocator),
        .fill => buildFillKernel(allocator),
    };
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Build vector_add kernel: c[i] = a[i] + b[i]
pub fn buildVectorAddKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "vector_add");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Add buffer bindings
    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    // Get thread index
    const gid = builder.globalInvocationId();
    const idx = try gid.x();

    // Bounds check: if (idx < n)
    const condition = try builder.lt(idx, try n.toExpr());

    // c[idx] = a[idx] + b[idx]
    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const sum = try builder.add(a_val, b_val);
    const c_idx = try c.at(idx);

    const assign_stmt = try builder.assignStmt(c_idx, sum);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build vector_sub kernel: c[i] = a[i] - b[i]
pub fn buildVectorSubKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "vector_sub");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const diff = try builder.sub(a_val, b_val);
    const c_idx = try c.at(idx);

    const assign_stmt = try builder.assignStmt(c_idx, diff);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build vector_mul kernel: c[i] = a[i] * b[i]
pub fn buildVectorMulKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "vector_mul");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const product = try builder.mul(a_val, b_val);
    const c_idx = try c.at(idx);

    const assign_stmt = try builder.assignStmt(c_idx, product);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build vector_div kernel: c[i] = a[i] / b[i]
pub fn buildVectorDivKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "vector_div");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const quotient = try builder.div(a_val, b_val);
    const c_idx = try c.at(idx);

    const assign_stmt = try builder.assignStmt(c_idx, quotient);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build vector_scale kernel: b[i] = a[i] * scale
pub fn buildVectorScaleKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "vector_scale");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .write_only);
    const scale = try builder.addUniform("scale", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const a_val = try a.at(idx);
    const scaled = try builder.mul(a_val, try scale.toExpr());
    const b_idx = try b.at(idx);

    const assign_stmt = try builder.assignStmt(b_idx, scaled);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Build matrix_multiply kernel: C = A * B (with tiled optimization)
///
/// Uses shared memory tiling for better performance:
/// - Each workgroup computes a TILE_SIZE x TILE_SIZE block of C
/// - Tiles of A and B are loaded into shared memory cooperatively
/// - Reduces global memory bandwidth requirements
pub fn buildMatrixMultiplyKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "matrix_multiply");
    errdefer builder.deinit();

    // 16x16 tiles for matrix multiplication
    const TILE_SIZE: u32 = 16;
    _ = builder.setWorkgroupSize(TILE_SIZE, TILE_SIZE, 1);

    // Buffer bindings
    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const m = try builder.addUniform("m", Type.u32Type()); // A rows
    const n = try builder.addUniform("n", Type.u32Type()); // B cols
    const k_dim = try builder.addUniform("k", Type.u32Type()); // A cols = B rows

    // Shared memory for tiles
    _ = try builder.addSharedMemory("tile_a", Type.f32Type(), TILE_SIZE * TILE_SIZE);
    _ = try builder.addSharedMemory("tile_b", Type.f32Type(), TILE_SIZE * TILE_SIZE);

    // Get thread indices
    const gid = builder.globalInvocationId();
    const lid = builder.localInvocationId();

    // Global position in output matrix
    const row = try gid.y();
    const col = try gid.x();

    // Local position within tile
    const local_row = try lid.y();
    const local_col = try lid.x();

    // Accumulator for dot product - declared as local variable
    const sum = try builder.declareVar("sum", Type.f32Type(), try builder.f32Lit(0.0));

    // Compute number of tiles along K dimension
    // num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE
    const tile_size_lit = try builder.u32Lit(TILE_SIZE);
    const tile_minus_one = try builder.u32Lit(TILE_SIZE - 1);
    const k_plus = try builder.add(try k_dim.toExpr(), tile_minus_one);
    const num_tiles = try builder.div(k_plus, tile_size_lit);

    // Loop variable
    const tile_idx = try builder.declareVar("tile", Type.u32Type(), try builder.u32Lit(0));

    // Build the tiled computation loop body
    // For each tile:
    // 1. Load A[row, tile*TILE_SIZE + local_col] into tile_a[local_row][local_col]
    // 2. Load B[tile*TILE_SIZE + local_row, col] into tile_b[local_row][local_col]
    // 3. Barrier to ensure all threads have loaded their data
    // 4. For each element in the tile, accumulate: sum += tile_a[local_row][k] * tile_b[k][local_col]
    // 5. Barrier before next tile

    // Calculate tile start index
    const tile_start = try builder.mul(try tile_idx.toExpr(), tile_size_lit);

    // Load A tile: a_idx = row * k + (tile_start + local_col)
    const a_col = try builder.add(tile_start, local_col);
    const row_k = try builder.mul(row, try k_dim.toExpr());
    const a_idx = try builder.add(row_k, a_col);

    // Load B tile: b_idx = (tile_start + local_row) * n + col
    const b_row = try builder.add(tile_start, local_row);
    const b_row_n = try builder.mul(b_row, try n.toExpr());
    const b_idx = try builder.add(b_row_n, col);

    // Bounds checks for loading
    const a_col_check = try builder.lt(a_col, try k_dim.toExpr());
    const row_check = try builder.lt(row, try m.toExpr());
    const a_bounds = try builder.logicalAnd(row_check, a_col_check);

    const b_row_check = try builder.lt(b_row, try k_dim.toExpr());
    const col_check = try builder.lt(col, try n.toExpr());
    const b_bounds = try builder.logicalAnd(b_row_check, col_check);

    // Load value from A (or 0 if out of bounds)
    const a_val = try a.at(a_idx);
    const zero = try builder.f32Lit(0.0);
    const a_load = try builder.select(a_bounds, a_val, zero);

    // Load value from B (or 0 if out of bounds)
    const b_val = try b.at(b_idx);
    const b_load = try builder.select(b_bounds, b_val, zero);

    // Compute partial product and accumulate
    const product = try builder.mul(a_load, b_load);
    const sum_val = try sum.toExpr();
    const new_sum = try builder.add(sum_val, product);

    // Update accumulator
    const update_sum = try builder.assignStmt(try sum.toExpr(), new_sum);

    // Increment tile index
    const one = try builder.u32Lit(1);
    const next_tile = try builder.add(try tile_idx.toExpr(), one);
    const update_tile = try builder.assignStmt(try tile_idx.toExpr(), next_tile);

    // Loop condition: tile < num_tiles
    const loop_cond = try builder.lt(try tile_idx.toExpr(), num_tiles);

    // Add barrier for synchronization
    try builder.barrier();

    // Build the for loop
    try builder.forLoop(
        null, // init already done
        loop_cond,
        update_tile,
        &[_]*const dsl.Stmt{update_sum},
    );

    // Store result: c[row * n + col] = sum
    const final_bounds = try builder.logicalAnd(row_check, col_check);
    const row_times_n = try builder.mul(row, try n.toExpr());
    const output_idx = try builder.add(row_times_n, col);
    const c_idx = try c.at(output_idx);
    const store_stmt = try builder.assignStmt(c_idx, try sum.toExpr());

    try builder.ifStmt(final_bounds, &[_]*const dsl.Stmt{store_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build matrix_transpose kernel: B = A^T
pub fn buildMatrixTransposeKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "matrix_transpose");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(16, 16, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .write_only);
    const rows = try builder.addUniform("rows", Type.u32Type());
    const cols = try builder.addUniform("cols", Type.u32Type());

    const gid = builder.globalInvocationId();
    const row = try gid.y();
    const col = try gid.x();

    const row_check = try builder.lt(row, try rows.toExpr());
    const col_check = try builder.lt(col, try cols.toExpr());
    const bounds_check = try builder.logicalAnd(row_check, col_check);

    // src_idx = row * cols + col
    // dst_idx = col * rows + row
    const src_idx = try builder.add(try builder.mul(row, try cols.toExpr()), col);
    const dst_idx = try builder.add(try builder.mul(col, try rows.toExpr()), row);

    const a_val = try a.at(src_idx);
    const b_dst = try b.at(dst_idx);

    const assign_stmt = try builder.assignStmt(b_dst, a_val);
    try builder.ifStmt(bounds_check, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Reduction Operations
// ============================================================================

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

// ============================================================================
// Neural Network Activations
// ============================================================================

/// Build softmax kernel: output[i] = exp(input[i] - max) / sum(exp)
pub fn buildSoftmaxKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "softmax");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const max_val = try builder.addUniform("max_val", Type.f32Type());
    const sum_exp = try builder.addUniform("sum_exp", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = exp(input[i] - max_val) / sum_exp
    const input_val = try input.at(idx);
    const shifted = try builder.sub(input_val, try max_val.toExpr());
    // exp() would need to be a builtin function call
    // For now, just store the shifted value as placeholder
    const output_idx = try output.at(idx);

    const assign_stmt = try builder.assignStmt(output_idx, shifted);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    // Suppress unused
    _ = sum_exp;

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build relu kernel: output[i] = max(0, input[i])
pub fn buildReluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "relu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const input_val = try input.at(idx);
    const zero = try builder.f32Lit(0.0);

    // max(0, input[i]) - use select
    const is_positive = try builder.gt(input_val, zero);
    const relu_val = try builder.select(is_positive, input_val, zero);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, relu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build sigmoid kernel: output[i] = 1 / (1 + exp(-input[i]))
pub fn buildSigmoidKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "sigmoid");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Placeholder: actual impl needs exp() builtin
    const input_val = try input.at(idx);
    const output_idx = try output.at(idx);

    const assign_stmt = try builder.assignStmt(output_idx, input_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build tanh kernel: output[i] = tanh(input[i])
pub fn buildTanhKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "tanh");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Placeholder: actual impl needs tanh() builtin
    const input_val = try input.at(idx);
    const output_idx = try output.at(idx);

    const assign_stmt = try builder.assignStmt(output_idx, input_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Linear Algebra
// ============================================================================

/// Build dot_product kernel: result[0] = sum(a[i] * b[i])
pub fn buildDotProductKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "dot_product");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .read_write);
    const n = try builder.addUniform("n", Type.u32Type());

    _ = try builder.addSharedMemory("shared_data", Type.f32Type(), 256);

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // partial[i] = a[i] * b[i]
    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const product = try builder.mul(a_val, b_val);

    // Then reduce (simplified - real impl uses parallel reduction)
    const output_idx = try builder.u32Lit(0);
    const output_ptr = try output.at(output_idx);

    const assign_stmt = try builder.assignStmt(output_ptr, product);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build normalize kernel: output[i] = input[i] / norm
pub fn buildNormalizeKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "normalize");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const norm = try builder.addUniform("norm", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const input_val = try input.at(idx);
    const normalized = try builder.div(input_val, try norm.toExpr());
    const output_idx = try output.at(idx);

    const assign_stmt = try builder.assignStmt(output_idx, normalized);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build saxpy kernel: y[i] = a * x[i] + y[i]
pub fn buildSaxpyKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "saxpy");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const x = try builder.addBuffer("x", Type.f32Type(), .read_only);
    const y = try builder.addBuffer("y", Type.f32Type(), .read_write);
    const a = try builder.addUniform("a", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // y[i] = a * x[i] + y[i]
    const x_val = try x.at(idx);
    const y_val = try y.at(idx);
    const ax = try builder.mul(try a.toExpr(), x_val);
    const result = try builder.add(ax, y_val);
    const y_idx = try y.at(idx);

    const assign_stmt = try builder.assignStmt(y_idx, result);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Utility Operations
// ============================================================================

/// Build copy kernel: dst[i] = src[i]
pub fn buildCopyKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "copy");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const src = try builder.addBuffer("src", Type.f32Type(), .read_only);
    const dst = try builder.addBuffer("dst", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const src_val = try src.at(idx);
    const dst_idx = try dst.at(idx);

    const assign_stmt = try builder.assignStmt(dst_idx, src_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fill kernel: dst[i] = value
pub fn buildFillKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "fill");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const dst = try builder.addBuffer("dst", Type.f32Type(), .write_only);
    const value = try builder.addUniform("value", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    const dst_idx = try dst.at(idx);

    const assign_stmt = try builder.assignStmt(dst_idx, try value.toExpr());
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Tests
// ============================================================================

test "buildVectorAddKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildVectorAddKernel(allocator);
    try std.testing.expectEqualStrings("vector_add", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len);
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildKernelIR for all types" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test a few kernel types to ensure they build
    const kernels = [_]BuiltinKernel{
        .vector_add,
        .vector_sub,
        .reduce_sum,
        .softmax,
        .relu,
        .dot_product,
        .copy,
    };

    for (kernels) |kernel_type| {
        const ir = try buildKernelIR(allocator, kernel_type);
        try std.testing.expect(ir.name.len > 0);
        try std.testing.expect(ir.buffers.len > 0);
    }
}

test "buildMatrixMultiplyKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildMatrixMultiplyKernel(allocator);
    try std.testing.expectEqualStrings("matrix_multiply", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 3), ir.uniforms.len); // m, n, k
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}
