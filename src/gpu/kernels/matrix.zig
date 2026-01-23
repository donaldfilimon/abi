//! Matrix GPU Kernel Definitions
//!
//! Pre-defined kernel IR for matrix operations.
//!
//! ## Operations
//! - matrix_multiply: C = A * B (tiled for performance)
//! - matrix_transpose: B = A^T

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

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
// Tests
// ============================================================================

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
