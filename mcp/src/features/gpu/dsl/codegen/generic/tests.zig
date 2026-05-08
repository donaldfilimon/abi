//! Tests for the generic code generator instances.

const std = @import("std");
const kernel = @import("../../kernel.zig");
const instances = @import("instances.zig");

test "GlslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = instances.GlslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 450") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "void main()") != null);
}

test "WgslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = instances.WgslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "@compute") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "@workgroup_size") != null);
}

test "MslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = instances.MslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#include <metal_stdlib>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "kernel void") != null);
}

test "CudaGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = instances.CudaGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#include <cuda_runtime.h>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "__global__") != null);
}
