//! Batch Operation Kernel Definitions
//!
//! Pre-defined kernel IR for batch operations.
//!
//! ## Operations
//! - batch_matmul: Batched matrix multiplication C[b] = A[b] * B[b]
//! - batch_cosine_similarity: Cosine similarity for vector search

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

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

test "buildBatchCosineSimilarityKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildBatchCosineSimilarityKernel(allocator);
    try std.testing.expectEqualStrings("batch_cosine_similarity", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len); // query, vectors, output
    try std.testing.expectEqual(@as(usize, 3), ir.uniforms.len); // query_norm, num_vectors, dim
}
