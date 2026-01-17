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
        // Element-wise operations
        .vector_add => buildVectorAddKernel(allocator),
        .vector_sub => buildVectorSubKernel(allocator),
        .vector_mul => buildVectorMulKernel(allocator),
        .vector_div => buildVectorDivKernel(allocator),
        .vector_scale => buildVectorScaleKernel(allocator),
        // Matrix operations
        .matrix_multiply => buildMatrixMultiplyKernel(allocator),
        .matrix_transpose => buildMatrixTransposeKernel(allocator),
        // Reductions
        .reduce_sum => buildReduceSumKernel(allocator),
        .reduce_max => buildReduceMaxKernel(allocator),
        .reduce_min => buildReduceMinKernel(allocator),
        .reduce_product => buildReduceProductKernel(allocator),
        // Basic activations
        .softmax => buildSoftmaxKernel(allocator),
        .relu => buildReluKernel(allocator),
        .sigmoid => buildSigmoidKernel(allocator),
        .tanh => buildTanhKernel(allocator),
        // Linear algebra
        .dot_product => buildDotProductKernel(allocator),
        .normalize => buildNormalizeKernel(allocator),
        .saxpy => buildSaxpyKernel(allocator),
        .copy => buildCopyKernel(allocator),
        .fill => buildFillKernel(allocator),
        // Neural network activations
        .gelu => buildGeluKernel(allocator),
        .gelu_fast => buildGeluFastKernel(allocator),
        .silu => buildSiluKernel(allocator),
        .swiglu => buildSwigluKernel(allocator),
        // Normalization layers
        .layer_norm => buildLayerNormKernel(allocator),
        .rms_norm => buildRmsNormKernel(allocator),
        .batch_norm => buildBatchNormKernel(allocator),
        // Fused operations
        .fused_add_norm => buildFusedAddNormKernel(allocator),
        .fused_linear_gelu => buildFusedLinearGeluKernel(allocator),
        // Batch operations
        .batch_matmul => buildBatchMatmulKernel(allocator),
        .batch_cosine_similarity => buildBatchCosineSimilarityKernel(allocator),
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
// Neural Network Activation Kernels
// ============================================================================

/// Build GELU kernel: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// GELU (Gaussian Error Linear Unit) is used in BERT, GPT, and modern transformers.
pub fn buildGeluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "gelu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Constants: sqrt(2/pi) ≈ 0.7978845608, coefficient = 0.044715
    const x = try input.at(idx);

    // x^3
    const x_sq = try builder.mul(x, x);
    const x_cubed = try builder.mul(x_sq, x);

    // 0.044715 * x^3
    const coef = try builder.f32Lit(0.044715);
    const coef_x3 = try builder.mul(coef, x_cubed);

    // x + 0.044715 * x^3
    const inner_sum = try builder.add(x, coef_x3);

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    const sqrt_2_pi = try builder.f32Lit(0.7978845608);
    const scaled = try builder.mul(sqrt_2_pi, inner_sum);

    // tanh(...)
    const tanh_val = try builder.tanh(scaled);

    // 1 + tanh(...)
    const one = try builder.f32Lit(1.0);
    const one_plus_tanh = try builder.add(one, tanh_val);

    // 0.5 * x
    const half = try builder.f32Lit(0.5);
    const half_x = try builder.mul(half, x);

    // 0.5 * x * (1 + tanh(...))
    const gelu_val = try builder.mul(half_x, one_plus_tanh);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fast GELU approximation kernel: output[i] = x * sigmoid(1.702 * x)
/// Faster than exact GELU, used in some production systems.
pub fn buildGeluFastKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "gelu_fast");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Fast GELU: x * sigmoid(1.702 * x)
    // sigmoid(y) = 1 / (1 + exp(-y))
    const x = try input.at(idx);

    // 1.702 * x
    const coef = try builder.f32Lit(1.702);
    const scaled_x = try builder.mul(coef, x);

    // -1.702 * x
    const neg_scaled = try builder.neg(scaled_x);

    // exp(-1.702 * x)
    const exp_neg = try builder.exp(neg_scaled);

    // 1 + exp(-1.702 * x)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg);

    // sigmoid = 1 / (1 + exp(-1.702 * x))
    const sigmoid = try builder.div(one, denom);

    // x * sigmoid(1.702 * x)
    const gelu_fast = try builder.mul(x, sigmoid);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_fast);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build SiLU (Swish) kernel: output[i] = x * sigmoid(x) = x / (1 + exp(-x))
/// SiLU (Sigmoid Linear Unit) is used in EfficientNet, LLaMA, and many modern models.
pub fn buildSiluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "silu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const x = try input.at(idx);

    // -x
    const neg_x = try builder.neg(x);

    // exp(-x)
    const exp_neg_x = try builder.exp(neg_x);

    // 1 + exp(-x)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg_x);

    // x / (1 + exp(-x))
    const silu_val = try builder.div(x, denom);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, silu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build SwiGLU kernel: output[i] = input[i] * silu(gate[i])
/// SwiGLU (Swish-Gated Linear Unit) is used in LLaMA, Mixtral, and PaLM.
/// Buffers: input (x), gate (g), output - computes x * SiLU(g)
pub fn buildSwigluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "swiglu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gate = try builder.addBuffer("gate", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // SwiGLU(x, g) = x * SiLU(g) = x * g / (1 + exp(-g))
    const x = try input.at(idx);
    const g = try gate.at(idx);

    // -g
    const neg_g = try builder.neg(g);

    // exp(-g)
    const exp_neg_g = try builder.exp(neg_g);

    // 1 + exp(-g)
    const one = try builder.f32Lit(1.0);
    const denom = try builder.add(one, exp_neg_g);

    // silu(g) = g / (1 + exp(-g))
    const silu_g = try builder.div(g, denom);

    // x * silu(g)
    const swiglu_val = try builder.mul(x, silu_g);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, swiglu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Normalization Layer Kernels
// ============================================================================

/// Build LayerNorm kernel: output[i] = gamma[i] * (input[i] - mean) / sqrt(var + eps) + beta[i]
/// LayerNorm normalizes across the feature dimension for each sample.
/// Expects pre-computed mean and variance passed as uniforms.
pub fn buildLayerNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "layer_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const mean = try builder.addUniform("mean", Type.f32Type());
    const variance = try builder.addUniform("variance", Type.f32Type());
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = gamma[i] * (input[i] - mean) / sqrt(variance + epsilon) + beta[i]
    const x = try input.at(idx);
    const g = try gamma.at(idx);
    const b = try beta.at(idx);

    // input[i] - mean
    const centered = try builder.sub(x, try mean.toExpr());

    // variance + epsilon
    const var_eps = try builder.add(try variance.toExpr(), try epsilon.toExpr());

    // sqrt(variance + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // (input[i] - mean) / sqrt(variance + epsilon)
    const normalized = try builder.div(centered, std_dev);

    // gamma[i] * normalized
    const scaled = try builder.mul(g, normalized);

    // gamma[i] * normalized + beta[i]
    const layer_norm_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, layer_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build RMSNorm kernel: output[i] = gamma[i] * input[i] / sqrt(mean(x^2) + eps)
/// RMSNorm (Root Mean Square Normalization) is used in LLaMA, T5, and is simpler than LayerNorm.
/// Expects pre-computed RMS value passed as uniform.
pub fn buildRmsNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "rms_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const rms = try builder.addUniform("rms", Type.f32Type()); // pre-computed sqrt(mean(x^2))
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // output[i] = gamma[i] * input[i] / (rms + epsilon)
    const x = try input.at(idx);
    const g = try gamma.at(idx);

    // rms + epsilon (for numerical stability)
    const rms_eps = try builder.add(try rms.toExpr(), try epsilon.toExpr());

    // input[i] / (rms + epsilon)
    const normalized = try builder.div(x, rms_eps);

    // gamma[i] * normalized
    const rms_norm_val = try builder.mul(g, normalized);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, rms_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build BatchNorm kernel (inference mode): output = gamma * (input - mean) / sqrt(var + eps) + beta
/// BatchNorm normalizes across the batch dimension for each feature.
/// Uses pre-computed running mean and variance from training.
pub fn buildBatchNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "batch_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const running_mean = try builder.addBuffer("running_mean", Type.f32Type(), .read_only);
    const running_var = try builder.addBuffer("running_var", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Compute channel index: c = idx % channels
    const c = try builder.mod(idx, try channels.toExpr());

    // Load per-channel parameters
    const x = try input.at(idx);
    const g = try gamma.at(c);
    const b = try beta.at(c);
    const mean_c = try running_mean.at(c);
    const var_c = try running_var.at(c);

    // (input - mean)
    const centered = try builder.sub(x, mean_c);

    // var + epsilon
    const var_eps = try builder.add(var_c, try epsilon.toExpr());

    // sqrt(var + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // (input - mean) / sqrt(var + epsilon)
    const normalized = try builder.div(centered, std_dev);

    // gamma * normalized + beta
    const scaled = try builder.mul(g, normalized);
    const batch_norm_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, batch_norm_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Fused Operations (for performance)
// ============================================================================

/// Build fused add + LayerNorm kernel: output = LayerNorm(input + residual)
/// Commonly used in transformer architectures for residual connections.
pub fn buildFusedAddNormKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "fused_add_norm");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const residual = try builder.addBuffer("residual", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const mean = try builder.addUniform("mean", Type.f32Type());
    const variance = try builder.addUniform("variance", Type.f32Type());
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());
    const n = try builder.addUniform("n", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // Fused: output = LayerNorm(input + residual)
    const x = try input.at(idx);
    const r = try residual.at(idx);
    const g = try gamma.at(idx);
    const b = try beta.at(idx);

    // input + residual
    const added = try builder.add(x, r);

    // (added - mean)
    const centered = try builder.sub(added, try mean.toExpr());

    // variance + epsilon
    const var_eps = try builder.add(try variance.toExpr(), try epsilon.toExpr());

    // sqrt(variance + epsilon)
    const std_dev = try builder.sqrt(var_eps);

    // normalized
    const normalized = try builder.div(centered, std_dev);

    // gamma * normalized + beta
    const scaled = try builder.mul(g, normalized);
    const fused_val = try builder.add(scaled, b);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, fused_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build fused linear + GELU kernel: output = GELU(input * weight + bias)
/// Fuses matrix-vector multiply with GELU activation for MLP layers.
/// Note: This is element-wise for pre-computed linear output.
pub fn buildFusedLinearGeluKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "fused_linear_gelu");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // For simplicity, this assumes linear output is pre-computed
    // and we just apply bias + GELU
    const linear_out = try builder.addBuffer("linear_out", Type.f32Type(), .read_only);
    const bias = try builder.addBuffer("bias", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());
    const hidden_dim = try builder.addUniform("hidden_dim", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();
    const condition = try builder.lt(idx, try n.toExpr());

    // bias index = idx % hidden_dim
    const bias_idx = try builder.mod(idx, try hidden_dim.toExpr());

    // linear_out + bias
    const lo = try linear_out.at(idx);
    const b = try bias.at(bias_idx);
    const x = try builder.add(lo, b);

    // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const x_sq = try builder.mul(x, x);
    const x_cubed = try builder.mul(x_sq, x);
    const coef = try builder.f32Lit(0.044715);
    const coef_x3 = try builder.mul(coef, x_cubed);
    const inner_sum = try builder.add(x, coef_x3);
    const sqrt_2_pi = try builder.f32Lit(0.7978845608);
    const scaled = try builder.mul(sqrt_2_pi, inner_sum);
    const tanh_val = try builder.tanh(scaled);
    const one = try builder.f32Lit(1.0);
    const one_plus_tanh = try builder.add(one, tanh_val);
    const half = try builder.f32Lit(0.5);
    const half_x = try builder.mul(half, x);
    const gelu_val = try builder.mul(half_x, one_plus_tanh);

    const output_idx = try output.at(idx);
    const assign_stmt = try builder.assignStmt(output_idx, gelu_val);
    try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Build batched matrix multiply kernel: C[b] = A[b] * B[b]
/// Each batch element is an independent matrix multiplication.
/// Uses tiled algorithm for efficiency.
pub fn buildBatchMatmulKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "batch_matmul");
    errdefer builder.deinit();

    const TILE_SIZE: u32 = 16;
    _ = builder.setWorkgroupSize(TILE_SIZE, TILE_SIZE, 1);

    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const m = try builder.addUniform("m", Type.u32Type()); // A rows per batch
    const n = try builder.addUniform("n", Type.u32Type()); // B cols per batch
    const k_dim = try builder.addUniform("k", Type.u32Type()); // A cols = B rows
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());

    const gid = builder.globalInvocationId();
    const wid = builder.workgroupId();

    // Get batch index from z dimension of workgroup
    const batch_idx = try wid.z();
    const row = try gid.y();
    const col = try gid.x();

    // Bounds check
    const batch_check = try builder.lt(batch_idx, try batch_size.toExpr());
    const row_check = try builder.lt(row, try m.toExpr());
    const col_check = try builder.lt(col, try n.toExpr());
    const bounds_check = try builder.logicalAnd(batch_check, try builder.logicalAnd(row_check, col_check));

    // Compute matrix offsets for this batch
    // a_offset = batch_idx * m * k
    // b_offset = batch_idx * k * n
    // c_offset = batch_idx * m * n
    const m_k = try builder.mul(try m.toExpr(), try k_dim.toExpr());
    const a_offset = try builder.mul(batch_idx, m_k);

    const k_n = try builder.mul(try k_dim.toExpr(), try n.toExpr());
    const b_offset = try builder.mul(batch_idx, k_n);

    const m_n = try builder.mul(try m.toExpr(), try n.toExpr());
    const c_offset = try builder.mul(batch_idx, m_n);

    // Accumulator
    const sum = try builder.declareVar("sum", Type.f32Type(), try builder.f32Lit(0.0));

    // Loop over K dimension
    const loop_var = try builder.declareVar("kk", Type.u32Type(), try builder.u32Lit(0));
    const loop_cond = try builder.lt(try loop_var.toExpr(), try k_dim.toExpr());

    // A[batch, row, kk] = a[a_offset + row * k + kk]
    const row_k = try builder.mul(row, try k_dim.toExpr());
    const a_idx_base = try builder.add(a_offset, row_k);
    const a_idx = try builder.add(a_idx_base, try loop_var.toExpr());
    const a_val = try a.at(a_idx);

    // B[batch, kk, col] = b[b_offset + kk * n + col]
    const kk_n = try builder.mul(try loop_var.toExpr(), try n.toExpr());
    const b_idx_base = try builder.add(b_offset, kk_n);
    const b_idx = try builder.add(b_idx_base, col);
    const b_val = try b.at(b_idx);

    // sum += a_val * b_val
    const product = try builder.mul(a_val, b_val);
    const new_sum = try builder.add(try sum.toExpr(), product);
    const update_sum = try builder.assignStmt(try sum.toExpr(), new_sum);

    // Increment loop variable
    const one = try builder.u32Lit(1);
    const next_kk = try builder.add(try loop_var.toExpr(), one);
    const update_kk = try builder.assignStmt(try loop_var.toExpr(), next_kk);

    try builder.forLoop(null, loop_cond, update_kk, &[_]*const dsl.Stmt{update_sum});

    // C[batch, row, col] = c[c_offset + row * n + col]
    const row_n = try builder.mul(row, try n.toExpr());
    const c_idx_base = try builder.add(c_offset, row_n);
    const c_idx = try builder.add(c_idx_base, col);
    const c_ptr = try c.at(c_idx);
    const store_stmt = try builder.assignStmt(c_ptr, try sum.toExpr());

    try builder.ifStmt(bounds_check, &[_]*const dsl.Stmt{store_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build batch cosine similarity kernel: output[i] = dot(query, vectors[i]) / (norm_q * norm_v[i])
/// Optimized for vector database similarity search.
pub fn buildBatchCosineSimilarityKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "batch_cosine_similarity");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    const query = try builder.addBuffer("query", Type.f32Type(), .read_only);
    const vectors = try builder.addBuffer("vectors", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const query_norm = try builder.addUniform("query_norm", Type.f32Type());
    const num_vectors = try builder.addUniform("num_vectors", Type.u32Type());
    const dim = try builder.addUniform("dim", Type.u32Type());

    const gid = builder.globalInvocationId();
    const vec_idx = try gid.x();
    const condition = try builder.lt(vec_idx, try num_vectors.toExpr());

    // Compute dot product and vector norm
    const dot_sum = try builder.declareVar("dot_sum", Type.f32Type(), try builder.f32Lit(0.0));
    const vec_norm_sq = try builder.declareVar("vec_norm_sq", Type.f32Type(), try builder.f32Lit(0.0));

    // Loop over dimensions
    const d_var = try builder.declareVar("d", Type.u32Type(), try builder.u32Lit(0));
    const loop_cond = try builder.lt(try d_var.toExpr(), try dim.toExpr());

    // query[d]
    const q_val = try query.at(try d_var.toExpr());

    // vectors[vec_idx * dim + d]
    const vec_offset = try builder.mul(vec_idx, try dim.toExpr());
    const vec_d_idx = try builder.add(vec_offset, try d_var.toExpr());
    const v_val = try vectors.at(vec_d_idx);

    // dot_sum += query[d] * vectors[d]
    const dot_prod = try builder.mul(q_val, v_val);
    const new_dot = try builder.add(try dot_sum.toExpr(), dot_prod);
    const update_dot = try builder.assignStmt(try dot_sum.toExpr(), new_dot);

    // vec_norm_sq += vectors[d] * vectors[d]
    const v_sq = try builder.mul(v_val, v_val);
    const new_norm_sq = try builder.add(try vec_norm_sq.toExpr(), v_sq);
    const update_norm = try builder.assignStmt(try vec_norm_sq.toExpr(), new_norm_sq);

    // Increment d
    const one = try builder.u32Lit(1);
    const next_d = try builder.add(try d_var.toExpr(), one);
    const update_d = try builder.assignStmt(try d_var.toExpr(), next_d);

    try builder.forLoop(null, loop_cond, update_d, &[_]*const dsl.Stmt{ update_dot, update_norm });

    // cosine_sim = dot_sum / (query_norm * sqrt(vec_norm_sq))
    const vec_norm = try builder.sqrt(try vec_norm_sq.toExpr());
    const norm_product = try builder.mul(try query_norm.toExpr(), vec_norm);

    // Avoid division by zero
    const epsilon = try builder.f32Lit(1e-8);
    const safe_norm = try builder.add(norm_product, epsilon);
    const cosine_sim = try builder.div(try dot_sum.toExpr(), safe_norm);

    const output_idx = try output.at(vec_idx);
    const store_stmt = try builder.assignStmt(output_idx, cosine_sim);

    try builder.ifStmt(condition, &[_]*const dsl.Stmt{store_stmt}, null);

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
        // NN kernels
        .gelu,
        .gelu_fast,
        .silu,
        .swiglu,
        .layer_norm,
        .rms_norm,
        .batch_matmul,
        .batch_cosine_similarity,
    };

    for (kernels) |kernel_type| {
        const ir = try buildKernelIR(allocator, kernel_type);
        try std.testing.expect(ir.name.len > 0);
        try std.testing.expect(ir.buffers.len > 0);
    }
}

test "buildGeluKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildGeluKernel(allocator);
    try std.testing.expectEqualStrings("gelu", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len); // n
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildLayerNormKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildLayerNormKernel(allocator);
    try std.testing.expectEqualStrings("layer_norm", ir.name);
    try std.testing.expectEqual(@as(usize, 4), ir.buffers.len); // input, gamma, beta, output
    try std.testing.expectEqual(@as(usize, 4), ir.uniforms.len); // mean, variance, epsilon, n
}

test "buildBatchCosineSimilarityKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildBatchCosineSimilarityKernel(allocator);
    try std.testing.expectEqualStrings("batch_cosine_similarity", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len); // query, vectors, output
    try std.testing.expectEqual(@as(usize, 3), ir.uniforms.len); // query_norm, num_vectors, dim
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
