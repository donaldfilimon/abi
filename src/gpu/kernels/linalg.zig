//! Linear Algebra Kernel Definitions
//!
//! Pre-defined kernel IR for linear algebra operations.
//!
//! ## Operations
//! - dot_product: Sum of element-wise products
//! - normalize: Divide by norm
//! - saxpy: y = a*x + y
//! - copy: dst = src
//! - fill: dst = value

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

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

test "buildDotProductKernel - IR structure validation" {
    const allocator = std.testing.allocator;
    const ir = try buildDotProductKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    // Verify kernel name
    try std.testing.expectEqualStrings("dot_product", ir.name);

    // Verify workgroup size
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 1), ir.workgroup_size[1]);
    try std.testing.expectEqual(@as(u32, 1), ir.workgroup_size[2]);

    // Verify buffer bindings (a, b, output)
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqualStrings("a", ir.buffers[0].name);
    try std.testing.expectEqual(AccessMode.read_only, ir.buffers[0].access);
    try std.testing.expectEqualStrings("b", ir.buffers[1].name);
    try std.testing.expectEqual(AccessMode.read_only, ir.buffers[1].access);
    try std.testing.expectEqualStrings("output", ir.buffers[2].name);
    try std.testing.expectEqual(AccessMode.read_write, ir.buffers[2].access);

    // Verify uniform binding (n)
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len);
    try std.testing.expectEqualStrings("n", ir.uniforms[0].name);

    // Verify shared memory declaration
    try std.testing.expectEqual(@as(usize, 1), ir.shared_memory.len);
    try std.testing.expectEqualStrings("shared_data", ir.shared_memory[0].name);
    try std.testing.expectEqual(@as(?usize, 256), ir.shared_memory[0].size);

    // Verify kernel body has statements
    try std.testing.expect(ir.body.len > 0);

    // Verify kernel passes validation
    const validation = ir.validate();
    try std.testing.expect(validation.isValid());
}

test "buildNormalizeKernel - IR structure validation" {
    const allocator = std.testing.allocator;
    const ir = try buildNormalizeKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    // Verify kernel name
    try std.testing.expectEqualStrings("normalize", ir.name);

    // Verify workgroup size
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);

    // Verify buffer bindings (input, output)
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len);
    try std.testing.expectEqualStrings("input", ir.buffers[0].name);
    try std.testing.expectEqual(AccessMode.read_only, ir.buffers[0].access);
    try std.testing.expectEqualStrings("output", ir.buffers[1].name);
    try std.testing.expectEqual(AccessMode.write_only, ir.buffers[1].access);

    // Verify uniform bindings (norm, n)
    try std.testing.expectEqual(@as(usize, 2), ir.uniforms.len);
    try std.testing.expectEqualStrings("norm", ir.uniforms[0].name);
    try std.testing.expectEqualStrings("n", ir.uniforms[1].name);

    // Verify no shared memory needed for normalize
    try std.testing.expectEqual(@as(usize, 0), ir.shared_memory.len);

    // Verify kernel body has statements
    try std.testing.expect(ir.body.len > 0);

    // Verify kernel passes validation
    const validation = ir.validate();
    try std.testing.expect(validation.isValid());
}

test "buildSaxpyKernel - IR structure validation" {
    const allocator = std.testing.allocator;
    const ir = try buildSaxpyKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    // Verify kernel name
    try std.testing.expectEqualStrings("saxpy", ir.name);

    // Verify workgroup size
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);

    // Verify buffer bindings (x, y)
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len);
    try std.testing.expectEqualStrings("x", ir.buffers[0].name);
    try std.testing.expectEqual(AccessMode.read_only, ir.buffers[0].access);
    try std.testing.expectEqualStrings("y", ir.buffers[1].name);
    try std.testing.expectEqual(AccessMode.read_write, ir.buffers[1].access);

    // Verify uniform bindings (a, n)
    try std.testing.expectEqual(@as(usize, 2), ir.uniforms.len);
    try std.testing.expectEqualStrings("a", ir.uniforms[0].name);
    try std.testing.expectEqualStrings("n", ir.uniforms[1].name);

    // Verify no shared memory needed for SAXPY
    try std.testing.expectEqual(@as(usize, 0), ir.shared_memory.len);

    // Verify kernel body has statements
    try std.testing.expect(ir.body.len > 0);

    // Verify kernel passes validation
    const validation = ir.validate();
    try std.testing.expect(validation.isValid());
}

test "buildCopyKernel - IR structure validation" {
    const allocator = std.testing.allocator;
    const ir = try buildCopyKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    // Verify kernel name
    try std.testing.expectEqualStrings("copy", ir.name);

    // Verify buffer bindings (src, dst)
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len);
    try std.testing.expectEqualStrings("src", ir.buffers[0].name);
    try std.testing.expectEqual(AccessMode.read_only, ir.buffers[0].access);
    try std.testing.expectEqualStrings("dst", ir.buffers[1].name);
    try std.testing.expectEqual(AccessMode.write_only, ir.buffers[1].access);

    // Verify uniform binding (n)
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len);
    try std.testing.expectEqualStrings("n", ir.uniforms[0].name);

    // Verify kernel passes validation
    const validation = ir.validate();
    try std.testing.expect(validation.isValid());
}

test "buildFillKernel - IR structure validation" {
    const allocator = std.testing.allocator;
    const ir = try buildFillKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    // Verify kernel name
    try std.testing.expectEqualStrings("fill", ir.name);

    // Verify buffer bindings (dst only)
    try std.testing.expectEqual(@as(usize, 1), ir.buffers.len);
    try std.testing.expectEqualStrings("dst", ir.buffers[0].name);
    try std.testing.expectEqual(AccessMode.write_only, ir.buffers[0].access);

    // Verify uniform bindings (value, n)
    try std.testing.expectEqual(@as(usize, 2), ir.uniforms.len);
    try std.testing.expectEqualStrings("value", ir.uniforms[0].name);
    try std.testing.expectEqualStrings("n", ir.uniforms[1].name);

    // Verify kernel passes validation
    const validation = ir.validate();
    try std.testing.expect(validation.isValid());
}

test "linalg kernels - binding index uniqueness" {
    const allocator = std.testing.allocator;

    // Test each kernel for binding conflicts
    const kernels = .{
        try buildDotProductKernel(allocator),
        try buildNormalizeKernel(allocator),
        try buildSaxpyKernel(allocator),
        try buildCopyKernel(allocator),
        try buildFillKernel(allocator),
    };

    inline for (kernels) |ir| {
        defer {
            ir.deinit(allocator);
            allocator.destroy(@constCast(ir));
        }

        // Validation should pass (no duplicate bindings)
        const validation = ir.validate();
        try std.testing.expect(!validation.errors.duplicate_bindings);
    }
}

test "linalg kernels - element types are f32" {
    const allocator = std.testing.allocator;

    // All linalg kernels should use f32 for buffers
    const ir = try buildDotProductKernel(allocator);
    defer {
        ir.deinit(allocator);
        allocator.destroy(@constCast(ir));
    }

    for (ir.buffers) |buffer| {
        try std.testing.expectEqual(Type{ .scalar = .f32 }, buffer.element_type);
    }
}
