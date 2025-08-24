const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // WDBX CLI executable
    const wdbx_exe = b.addExecutable(.{
        .name = "wdbx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Create modules
    const database_mod = b.createModule(.{
        .root_source_file = b.path("src/database.zig"),
        .target = target,
        .optimize = optimize,
    });

    const http_server_mod = b.createModule(.{
        .root_source_file = b.path("src/wdbx_http_server.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Enhanced WDBX module with all 15 improvements
    const wdbx_enhanced_mod = b.createModule(.{
        .root_source_file = b.path("src/wdbx_enhanced.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add SIMD module dependency
    const simd_mod = b.createModule(.{
        .root_source_file = b.path("src/simd/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    wdbx_enhanced_mod.addImport("simd", simd_mod);

    // Add modules to executable
    wdbx_exe.root_module.addImport("database", database_mod);
    wdbx_exe.root_module.addImport("wdbx_http_server", http_server_mod);
    wdbx_exe.root_module.addImport("wdbx_enhanced", wdbx_enhanced_mod);

    b.installArtifact(wdbx_exe);

    // Run tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Database tests
    const db_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/database.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_db_tests = b.addRunArtifact(db_tests);
    const db_test_step = b.step("test-db", "Run database tests");
    db_test_step.dependOn(&run_db_tests.step);

    // HTTP server tests
    const http_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx_http_server.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_http_tests = b.addRunArtifact(http_tests);
    const http_test_step = b.step("test-http", "Run HTTP server tests");
    http_test_step.dependOn(&run_http_tests.step);

    // Enhanced WDBX demo executable
    const wdbx_enhanced_demo = b.addExecutable(.{
        .name = "wdbx_enhanced_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/wdbx_enhanced_demo.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    wdbx_enhanced_demo.root_module.addImport("wdbx_enhanced", wdbx_enhanced_mod);
    b.installArtifact(wdbx_enhanced_demo);

    const run_enhanced_demo = b.addRunArtifact(wdbx_enhanced_demo);
    const enhanced_demo_step = b.step("demo-enhanced", "Run WDBX Enhanced demo showcasing all 15 improvements");
    enhanced_demo_step.dependOn(&run_enhanced_demo.step);

    // Enhanced WDBX tests
    const enhanced_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx_enhanced.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    enhanced_tests.root_module.addImport("simd", simd_mod);

    const run_enhanced_tests = b.addRunArtifact(enhanced_tests);
    const enhanced_test_step = b.step("test-enhanced", "Run WDBX Enhanced tests");
    enhanced_test_step.dependOn(&run_enhanced_tests.step);

    // Add enhanced tests to main test step
    test_step.dependOn(&run_enhanced_tests.step);

    // Production WDBX executable
    const wdbx_production = b.addExecutable(.{
        .name = "wdbx_production",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wdbx_production.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    wdbx_production.root_module.addImport("simd", simd_mod);
    b.installArtifact(wdbx_production);

    const run_production = b.addRunArtifact(wdbx_production);
    const production_step = b.step("production", "Run WDBX Production server");
    production_step.dependOn(&run_production.step);

    // Stress test tool
    const stress_test = b.addExecutable(.{
        .name = "wdbx_stress_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/stress_test.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    b.installArtifact(stress_test);

    const run_stress = b.addRunArtifact(stress_test);
    const stress_step = b.step("stress-test", "Run stress testing against WDBX");
    stress_step.dependOn(&run_stress.step);
}
