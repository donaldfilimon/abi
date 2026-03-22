//! ABI build root — Zig 0.16, self-contained.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Feature flags ───────────────────────────────────────────────────
    const feat_gpu = b.option(bool, "feat-gpu", "GPU compute backends") orelse true;
    const feat_ai = b.option(bool, "feat-ai", "AI services") orelse true;
    const feat_database = b.option(bool, "feat-database", "Vector database") orelse true;
    const feat_network = b.option(bool, "feat-network", "Networking / Raft") orelse true;
    const feat_profiling = b.option(bool, "feat-profiling", "Observability / profiling") orelse true;
    const feat_web = b.option(bool, "feat-web", "Web framework") orelse true;
    const feat_pages = b.option(bool, "feat-pages", "Dashboard pages") orelse true;
    const feat_analytics = b.option(bool, "feat-analytics", "Analytics") orelse true;
    const feat_cloud = b.option(bool, "feat-cloud", "Cloud integration") orelse true;
    const feat_auth = b.option(bool, "feat-auth", "Authentication") orelse true;
    const feat_messaging = b.option(bool, "feat-messaging", "Messaging / pub-sub") orelse true;
    const feat_cache = b.option(bool, "feat-cache", "Caching") orelse true;
    const feat_storage = b.option(bool, "feat-storage", "Storage backends") orelse true;
    const feat_search = b.option(bool, "feat-search", "Full-text search") orelse true;
    const feat_mobile = b.option(bool, "feat-mobile", "Mobile (iOS/Android)") orelse false;
    const feat_gateway = b.option(bool, "feat-gateway", "API gateway") orelse true;
    const feat_benchmarks = b.option(bool, "feat-benchmarks", "Benchmark suites") orelse true;
    const feat_compute = b.option(bool, "feat-compute", "Distributed compute") orelse true;
    const feat_documents = b.option(bool, "feat-documents", "Document processing") orelse true;
    const feat_desktop = b.option(bool, "feat-desktop", "Desktop integration") orelse true;

    // AI sub-feature flags
    const feat_llm = b.option(bool, "feat-llm", "LLM inference") orelse feat_ai;
    const feat_training = b.option(bool, "feat-training", "Model training") orelse feat_ai;
    const feat_vision = b.option(bool, "feat-vision", "Vision models") orelse feat_ai;
    const feat_explore = b.option(bool, "feat-explore", "AI exploration") orelse feat_ai;
    const feat_reasoning = b.option(bool, "feat-reasoning", "Reasoning engine") orelse feat_ai;

    // Protocol flags
    const feat_lsp = b.option(bool, "feat-lsp", "Language Server Protocol") orelse true;
    const feat_mcp = b.option(bool, "feat-mcp", "Model Context Protocol") orelse true;

    // GPU backend flags
    const gpu_backend_str = b.option([]const u8, "gpu-backend", "GPU backends: metal,cuda,vulkan,webgpu,opengl,opengles,webgl2,stdgpu,fpga,tpu (comma-separated)");
    const gpu_metal = feat_gpu and hasBackend(gpu_backend_str, "metal");
    const gpu_cuda = feat_gpu and hasBackend(gpu_backend_str, "cuda");
    const gpu_vulkan = feat_gpu and hasBackend(gpu_backend_str, "vulkan");
    const gpu_webgpu = feat_gpu and hasBackend(gpu_backend_str, "webgpu");
    const gpu_opengl = feat_gpu and hasBackend(gpu_backend_str, "opengl");
    const gpu_opengles = feat_gpu and hasBackend(gpu_backend_str, "opengles");
    const gpu_webgl2 = feat_gpu and hasBackend(gpu_backend_str, "webgl2");
    const gpu_stdgpu = feat_gpu and (gpu_backend_str == null or hasBackend(gpu_backend_str, "stdgpu"));
    const gpu_fpga = feat_gpu and hasBackend(gpu_backend_str, "fpga");
    const gpu_tpu = feat_gpu and hasBackend(gpu_backend_str, "tpu");

    // ── Build options module ────────────────────────────────────────────
    const build_opts = b.addOptions();
    build_opts.addOption(bool, "feat_gpu", feat_gpu);
    build_opts.addOption(bool, "feat_ai", feat_ai);
    build_opts.addOption(bool, "feat_database", feat_database);
    build_opts.addOption(bool, "feat_network", feat_network);
    build_opts.addOption(bool, "feat_profiling", feat_profiling);
    build_opts.addOption(bool, "feat_web", feat_web);
    build_opts.addOption(bool, "feat_pages", feat_pages);
    build_opts.addOption(bool, "feat_analytics", feat_analytics);
    build_opts.addOption(bool, "feat_cloud", feat_cloud);
    build_opts.addOption(bool, "feat_auth", feat_auth);
    build_opts.addOption(bool, "feat_messaging", feat_messaging);
    build_opts.addOption(bool, "feat_cache", feat_cache);
    build_opts.addOption(bool, "feat_storage", feat_storage);
    build_opts.addOption(bool, "feat_search", feat_search);
    build_opts.addOption(bool, "feat_mobile", feat_mobile);
    build_opts.addOption(bool, "feat_gateway", feat_gateway);
    build_opts.addOption(bool, "feat_benchmarks", feat_benchmarks);
    build_opts.addOption(bool, "feat_compute", feat_compute);
    build_opts.addOption(bool, "feat_documents", feat_documents);
    build_opts.addOption(bool, "feat_desktop", feat_desktop);
    build_opts.addOption(bool, "feat_llm", feat_llm);
    build_opts.addOption(bool, "feat_training", feat_training);
    build_opts.addOption(bool, "feat_vision", feat_vision);
    build_opts.addOption(bool, "feat_explore", feat_explore);
    build_opts.addOption(bool, "feat_reasoning", feat_reasoning);
    build_opts.addOption(bool, "feat_lsp", feat_lsp);
    build_opts.addOption(bool, "feat_mcp", feat_mcp);
    build_opts.addOption(bool, "gpu_metal", gpu_metal);
    build_opts.addOption(bool, "gpu_cuda", gpu_cuda);
    build_opts.addOption(bool, "gpu_vulkan", gpu_vulkan);
    build_opts.addOption(bool, "gpu_webgpu", gpu_webgpu);
    build_opts.addOption(bool, "gpu_opengl", gpu_opengl);
    build_opts.addOption(bool, "gpu_opengles", gpu_opengles);
    build_opts.addOption(bool, "gpu_webgl2", gpu_webgl2);
    build_opts.addOption(bool, "gpu_stdgpu", gpu_stdgpu);
    build_opts.addOption(bool, "gpu_fpga", gpu_fpga);
    build_opts.addOption(bool, "gpu_tpu", gpu_tpu);
    build_opts.addOption([]const u8, "package_version", "0.1.0");

    const build_options_module = build_opts.createModule();

    // ── ABI library module ──────────────────────────────────────────────
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);

    // ── Static library ──────────────────────────────────────────────────
    const static_lib = b.addLibrary(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    static_lib.root_module.addImport("build_options", build_options_module);
    b.installArtifact(static_lib);
    b.step("lib", "Build static library").dependOn(&b.addInstallArtifact(static_lib, .{}).step);

    // ── Platform linking ────────────────────────────────────────────────
    if (target.result.os.tag == .macos) {
        for ([_][]const u8{ "System", "c" }) |lib| {
            static_lib.root_module.linkSystemLibrary(lib, .{});
        }
        if (gpu_metal) {
            for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |fw| {
                static_lib.root_module.linkFramework(fw, .{});
            }
        }
    }

    // ── Tests ───────────────────────────────────────────────────────────
    const test_step = b.step("test", "Run tests");

    // Library unit tests (refAllDecls in root.zig)
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_tests.root_module.addImport("build_options", build_options_module);
    if (target.result.os.tag == .macos) {
        lib_tests.root_module.linkSystemLibrary("System", .{});
        lib_tests.root_module.linkSystemLibrary("c", .{});
        if (gpu_metal) {
            for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |fw| {
                lib_tests.root_module.linkFramework(fw, .{});
            }
        }
    }
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);

    // Integration tests (test/ directory)
    const integration_test_mod = b.createModule(.{
        .root_source_file = b.path("test/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    integration_test_mod.addImport("abi", abi_module);
    integration_test_mod.addImport("build_options", build_options_module);
    const integration_tests = b.addTest(.{ .root_module = integration_test_mod });
    if (target.result.os.tag == .macos) {
        integration_tests.root_module.linkSystemLibrary("System", .{});
        integration_tests.root_module.linkSystemLibrary("c", .{});
    }
    test_step.dependOn(&b.addRunArtifact(integration_tests).step);

    // ── Stub parity check ───────────────────────────────────────────────
    const parity_mod = b.createModule(.{
        .root_source_file = b.path("src/feature_parity_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    parity_mod.addImport("build_options", build_options_module);
    const parity_tests = b.addTest(.{ .root_module = parity_mod });
    const check_parity_step = b.step("check-parity", "Verify mod/stub declaration parity");
    check_parity_step.dependOn(&b.addRunArtifact(parity_tests).step);

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "src", "test" };
    b.step("lint", "Check formatting").dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = true }).step);
    b.step("fix", "Fix formatting").dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = false }).step);

    // ── Cross-compilation check ────────────────────────────────────────
    const cross_check_step = b.step("cross-check", "Verify cross-compilation for key targets");
    const cross_targets = [_]struct { arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, name: []const u8 }{
        .{ .arch = .aarch64, .os = .linux, .name = "aarch64-linux" },
        .{ .arch = .x86_64, .os = .linux, .name = "x86_64-linux" },
        .{ .arch = .wasm32, .os = .wasi, .name = "wasm32-wasi" },
        .{ .arch = .x86_64, .os = .macos, .name = "x86_64-macos" },
    };
    inline for (cross_targets) |ct| {
        const cross_target = b.resolveTargetQuery(.{ .cpu_arch = ct.arch, .os_tag = ct.os });
        const cross_opts = b.addOptions();
        // Disable platform-specific features for cross targets
        cross_opts.addOption(bool, "feat_gpu", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_ai", true);
        cross_opts.addOption(bool, "feat_database", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_network", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_profiling", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_web", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_pages", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_analytics", true);
        cross_opts.addOption(bool, "feat_cloud", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_auth", true);
        cross_opts.addOption(bool, "feat_messaging", true);
        cross_opts.addOption(bool, "feat_cache", true);
        cross_opts.addOption(bool, "feat_storage", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_search", true);
        cross_opts.addOption(bool, "feat_mobile", false);
        cross_opts.addOption(bool, "feat_gateway", true);
        cross_opts.addOption(bool, "feat_benchmarks", true);
        cross_opts.addOption(bool, "feat_compute", true);
        cross_opts.addOption(bool, "feat_documents", true);
        cross_opts.addOption(bool, "feat_desktop", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_llm", true);
        cross_opts.addOption(bool, "feat_training", true);
        cross_opts.addOption(bool, "feat_vision", true);
        cross_opts.addOption(bool, "feat_explore", true);
        cross_opts.addOption(bool, "feat_reasoning", true);
        cross_opts.addOption(bool, "feat_lsp", ct.os != .wasi);
        cross_opts.addOption(bool, "feat_mcp", ct.os != .wasi);
        cross_opts.addOption(bool, "gpu_metal", false);
        cross_opts.addOption(bool, "gpu_cuda", false);
        cross_opts.addOption(bool, "gpu_vulkan", false);
        cross_opts.addOption(bool, "gpu_webgpu", false);
        cross_opts.addOption(bool, "gpu_opengl", false);
        cross_opts.addOption(bool, "gpu_opengles", false);
        cross_opts.addOption(bool, "gpu_webgl2", false);
        cross_opts.addOption(bool, "gpu_stdgpu", ct.os != .wasi);
        cross_opts.addOption(bool, "gpu_fpga", false);
        cross_opts.addOption(bool, "gpu_tpu", false);
        cross_opts.addOption([]const u8, "package_version", "0.1.0");

        const cross_mod = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = cross_target,
            .optimize = optimize,
        });
        cross_mod.addImport("build_options", cross_opts.createModule());
        const cross_lib = b.addLibrary(.{
            .name = "cross-" ++ ct.name,
            .root_module = cross_mod,
            .linkage = .static,
        });
        cross_check_step.dependOn(&cross_lib.step);
    }

    // ── Aggregate check ─────────────────────────────────────────────────
    const check_step = b.step("check", "Run lint + test + parity");
    check_step.dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = true }).step);
    check_step.dependOn(&b.addRunArtifact(lib_tests).step);
    check_step.dependOn(&b.addRunArtifact(parity_tests).step);
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn hasBackend(backend_str: ?[]const u8, name: []const u8) bool {
    const str = backend_str orelse return false;
    var it = std.mem.splitScalar(u8, str, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (std.mem.eql(u8, trimmed, name)) return true;
    }
    return false;
}
