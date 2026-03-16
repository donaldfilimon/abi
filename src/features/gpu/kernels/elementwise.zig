//! Element-wise GPU Kernel Definitions
//!
//! Pre-defined kernel IR for element-wise operations on vectors.
//! These kernels operate on individual elements independently.
//!
//! ## Operations
//! - vector_add: c[i] = a[i] + b[i]
//! - vector_sub: c[i] = a[i] - b[i]
//! - vector_mul: c[i] = a[i] * b[i]
//! - vector_div: c[i] = a[i] / b[i]
//! - vector_scale: b[i] = a[i] * scale

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

/// Build vector_add kernel: c[i] = a[i] + b[i]
pub fn buildVectorAddKernel(allocator: std.mem.Allocator) !*const KernelIR {
    return buildVectorAddKernelSIMD(allocator, 1); // Default to scalar version
}

/// Build SIMD vector_add kernel with vectorization factor
pub fn buildVectorAddKernelSIMD(allocator: std.mem.Allocator, vector_width: u32) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, if (vector_width == 1) "vector_add" else "vector_add_simd");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Add buffer bindings
    const a = try builder.addBuffer("a", Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", Type.u32Type());

    // Get thread index
    const gid = builder.globalInvocationId();
    const base_idx = try gid.x();

    // Generate vectorized bounds-checked assignments
    var i: u32 = 0;
    while (i < vector_width) : (i += 1) {
        const idx = try builder.add(base_idx, try builder.u32Lit(@intCast(i)));

        // Bounds check: if (idx < n)
        const condition = try builder.lt(idx, try n.toExpr());

        // c[idx] = a[idx] + b[idx]
        const a_val = try a.at(idx);
        const b_val = try b.at(idx);
        const sum = try builder.add(a_val, b_val);
        const c_idx = try c.at(idx);

        const assign_stmt = try builder.assignStmt(c_idx, sum);
        // ifStmt adds directly to the builder's statement list
        try builder.ifStmt(condition, &[_]*const dsl.Stmt{assign_stmt}, null);
    }

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

test {
    std.testing.refAllDecls(@This());
}
