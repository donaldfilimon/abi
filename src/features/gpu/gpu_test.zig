const std = @import("std");
const gpu = @import("mod.zig");
const backends = @import("backends/backends.zig");

test "Backend enum properties" {
    const backend = backends.Backend.vulkan;
    try std.testing.expectEqualStrings("Vulkan", backend.toString());
    try std.testing.expect(backend.getPriority() == 10);
}

test "BackendManager initialization and detection" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = try backends.BackendManager.init(allocator);
    defer manager.deinit();

    try std.testing.expect(manager.available_backends.items.len > 0);

    // Check that Vulkan and OpenGL are detected (as per current stub logic)
    var has_vulkan = false;
    var has_opengl = false;
    var has_cpu = false;

    for (manager.available_backends.items) |b| {
        if (b == .vulkan) has_vulkan = true;
        if (b == .opengl) has_opengl = true;
        if (b == .cpu_fallback) has_cpu = true;
    }

    try std.testing.expect(has_vulkan);
    try std.testing.expect(has_opengl);
    try std.testing.expect(has_cpu);
}

test "BackendManager selection" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = try backends.BackendManager.init(allocator);
    defer manager.deinit();

    const best = manager.selectBestBackend();
    try std.testing.expect(best != null);
    // Vulkan has highest priority (10)
    try std.testing.expectEqual(backends.Backend.vulkan, best.?);

    try manager.selectBackend(.cpu_fallback);
    try std.testing.expectEqual(backends.Backend.cpu_fallback, manager.current_backend.?);
}

test "BackendManager capabilities" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = try backends.BackendManager.init(allocator);
    defer manager.deinit();

    const caps = try manager.getCapabilities(.vulkan);
    try std.testing.expectEqualStrings("Vulkan GPU", caps.name);
    try std.testing.expect(caps.compute_shaders);
    try std.testing.expect(caps.max_buffer_size > 0);
}

test "ShaderCompiler compilation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var compiler = try backends.ShaderCompiler.init(allocator, .vulkan);
    defer compiler.deinit();

    const source = "void main() {}";
    const result = try compiler.compileShader(source, .compute);
    defer allocator.free(result);

    try std.testing.expect(std.mem.startsWith(u8, result, "SPIRV_BC"));
    try std.testing.expect(std.mem.endsWith(u8, result, source));
}

test "BackendConfig defaults" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = try backends.BackendManager.init(allocator);
    defer manager.deinit();

    const config = manager.backend_configs.get(.vulkan).?;
    try std.testing.expect(config.vulkan.enable_debug_utils == true);
}
