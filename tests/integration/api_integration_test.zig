const std = @import("std");
const abi = @import("abi");
const testing = std.testing;

test "API Integration: web server endpoints" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test that web features are available when enabled
    const build_options = @import("build_options");
    if (build_options.enable_web) {
        // Web server should be available
        try testing.expect(framework.web.? != undefined);
    }
}

test "API Integration: authentication flow" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test authentication manager
    const build_options = @import("build_options");
    if (build_options.enable_web) {
        // Auth manager should be available
        try testing.expect(framework.web.?.auth_manager != undefined);

        // Test token validation
        const valid_token = try framework.web.?.auth_manager.generateToken(allocator, "testuser");
        defer allocator.free(valid_token);

        const is_valid = try framework.web.?.auth_manager.validateToken(valid_token);
        try testing.expect(is_valid);
    }
}

test "API Integration: vector database operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test database availability
    const build_options = @import("build_options");
    if (build_options.enable_database) {
        // Database should be available
        try testing.expect(framework.database.? != undefined);

        // Test basic vector operations
        const test_vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const id = try framework.database.?.addEmbedding(&test_vector);
        try testing.expect(id >= 0);

        // Test search
        const results = try framework.database.?.search(&test_vector, 1, allocator);
        defer abi.features.database.database.Db.freeResults(results, allocator);
        try testing.expect(results.len > 0);
        try testing.expect(results[0].id == id);
    }
}

test "API Integration: GPU acceleration availability" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test GPU availability based on build options
    const build_options = @import("build_options");
    if (build_options.enable_gpu) {
        // GPU features should be available
        try testing.expect(framework.gpu.? != undefined);

        // Test vector search GPU integration
        const accelerator = &framework.gpu.?.accelerator;
        var vector_search = abi.gpu.vector_search_gpu.VectorSearchGPU.init(allocator, accelerator, 4);
        defer vector_search.deinit();

        // Test basic operations
        const test_vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const id = try vector_search.insert(&test_vector);
        try testing.expect(id >= 0);

        // Test search (may fall back to CPU)
        const results = try vector_search.search(&test_vector, 1);
        defer allocator.free(results);
        try testing.expect(results.len > 0);
    }
}

test "API Integration: monitoring and metrics" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test monitoring availability
    const build_options = @import("build_options");
    if (build_options.enable_monitoring) {
        // Monitoring should be available
        try testing.expect(framework.monitoring.? != undefined);

        // Test health check
        const health = try framework.monitoring.?.health.getSystemHealth(allocator);
        defer allocator.free(health.checks);

        try testing.expect(health.status == .healthy or health.status == .degraded);
    }
}
