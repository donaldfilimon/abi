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
    const feat_tui = b.option(bool, "feat-tui", "Terminal user interface") orelse false;

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

    // ── GPU backend validation ──────────────────────────────────────────
    // Valid combos: metal=macOS only, cuda=not WASM, vulkan=Linux/Windows/Android,
    // webgpu/webgl2=WASM only, stdgpu=all, fpga/tpu=specialized hardware
    if (gpu_metal and gpu_cuda)
        std.log.warn("Both Metal and CUDA enabled — unusual; intended for cross-compilation only", .{});

    // ── Feature dependency warnings ─────────────────────────────────────
    if (!feat_ai) {
        if (feat_llm) std.log.warn("feat_llm requires feat_ai — llm will be stubbed", .{});
        if (feat_training) std.log.warn("feat_training requires feat_ai — training will be stubbed", .{});
        if (feat_vision) std.log.warn("feat_vision requires feat_ai — vision will be stubbed", .{});
        if (feat_explore) std.log.warn("feat_explore requires feat_ai — explore will be stubbed", .{});
        if (feat_reasoning) std.log.warn("feat_reasoning requires feat_ai — reasoning will be stubbed", .{});
    }
    if (feat_mcp and !feat_database)
        std.log.info("feat_mcp benefits from feat_database for DB tools", .{});

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
    build_opts.addOption(bool, "feat_tui", feat_tui);
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
        // Accelerate is needed whenever feat_gpu is on: gpu/mod.zig unconditionally
        // imports execution_coordinator → backends/metal → metal/accelerate.zig which
        // declares extern "Accelerate" symbols (CBLAS, vDSP, vForce).
        if (feat_gpu) {
            static_lib.root_module.linkFramework("Accelerate", .{});
        }
        // IOKit is needed for platform/smc (IOServiceOpen, IOConnectCallStructMethod, etc.)
        static_lib.root_module.linkFramework("IOKit", .{});
        // Objective-C runtime for desktop/macos_menu (objc_msgSend, objc_getClass, sel_registerName)
        static_lib.root_module.linkSystemLibrary("objc", .{});
        if (gpu_metal) {
            for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |fw| {
                static_lib.root_module.linkFramework(fw, .{});
            }
        }
    }

    // ── MCP server binary ────────────────────────────────────────────────
    const mcp_exe = b.addExecutable(.{
        .name = "abi-mcp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    mcp_exe.root_module.addImport("build_options", build_options_module);
    if (target.result.os.tag == .macos) {
        mcp_exe.root_module.linkSystemLibrary("System", .{});
        mcp_exe.root_module.linkSystemLibrary("c", .{});
        mcp_exe.root_module.linkSystemLibrary("objc", .{});
        mcp_exe.root_module.linkFramework("IOKit", .{});
        if (feat_gpu) mcp_exe.root_module.linkFramework("Accelerate", .{});
    }
    b.step("mcp", "Build MCP stdio server").dependOn(&b.addInstallArtifact(mcp_exe, .{}).step);

    // ── CLI binary ──────────────────────────────────────────────────────
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli_exe.root_module.addImport("build_options", build_options_module);
    if (target.result.os.tag == .macos) {
        cli_exe.root_module.linkSystemLibrary("System", .{});
        cli_exe.root_module.linkSystemLibrary("c", .{});
        cli_exe.root_module.linkSystemLibrary("objc", .{});
        cli_exe.root_module.linkFramework("IOKit", .{});
        if (feat_gpu) cli_exe.root_module.linkFramework("Accelerate", .{});
    }
    b.step("cli", "Build ABI command-line interface").dependOn(&b.addInstallArtifact(cli_exe, .{}).step);

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
        lib_tests.root_module.linkSystemLibrary("objc", .{});
        lib_tests.root_module.linkFramework("IOKit", .{});
        if (feat_gpu) {
            lib_tests.root_module.linkFramework("Accelerate", .{});
        }
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
        integration_tests.root_module.linkSystemLibrary("objc", .{});
        integration_tests.root_module.linkFramework("IOKit", .{});
        if (feat_gpu) {
            integration_tests.root_module.linkFramework("Accelerate", .{});
        }
        if (gpu_metal) {
            for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |fw| {
                integration_tests.root_module.linkFramework(fw, .{});
            }
        }
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
    //
    // Platform feature availability matrix:
    //
    //  Feature       | macOS | Linux | Windows | WASM/WASI | Freestanding
    //  --------------|-------|-------|---------|-----------|-------------
    //  feat_gpu      |  yes  |  yes  |   yes   |    no     |     no
    //  feat_ai       |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_database |  yes  |  yes  |   yes   |    no     |     no
    //  feat_network  |  yes  |  yes  |   yes   |    no     |     no
    //  feat_profiling|  yes  |  yes  |   yes   |    no     |     no
    //  feat_web      |  yes  |  yes  |   yes   |    no     |     no
    //  feat_pages    |  yes  |  yes  |   yes   |    no     |     no
    //  feat_analytics|  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_cloud    |  yes  |  yes  |   yes   |    no     |     no
    //  feat_auth     |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_messaging|  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_cache    |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_storage  |  yes  |  yes  |   yes   |    no     |     no
    //  feat_search   |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_mobile   |  no*  |  no   |    no   |    no     |     no
    //  feat_gateway  |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_benchmarks| yes  |  yes  |   yes   |   yes     |    yes
    //  feat_compute  |  yes  |  yes  |   yes   |    no     |     no
    //  feat_documents|  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_desktop  |  yes  |  no** |    no   |    no     |     no
    //  feat_lsp      |  yes  |  yes  |   yes   |    no     |     no
    //  feat_mcp      |  yes  |  yes  |   yes   |    no     |     no
    //
    //  GPU backends:
    //  gpu_metal     | macOS only (requires Metal framework)
    //  gpu_cuda      | Linux/Windows only (requires NVIDIA driver)
    //  gpu_vulkan    | Linux/Windows/Android (requires Vulkan loader)
    //  gpu_webgpu    | WASM only (requires browser WebGPU)
    //  gpu_webgl2    | WASM only (requires browser WebGL2)
    //  gpu_stdgpu    | All non-WASM targets (software fallback)
    //  gpu_opengl    | Linux/Windows/macOS (deprecated on macOS)
    //  gpu_opengles  | Mobile/embedded targets
    //  gpu_fpga/tpu  | Specialized hardware only
    //
    //  * feat_mobile defaults to false; enable explicitly for iOS cross-builds.
    //  ** feat_desktop uses macOS-specific NSStatusItem; stubbed on other OSes.
    //
    const cross_check_step = b.step("cross-check", "Verify cross-compilation for key targets");
    const cross_targets = [_]struct { arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, name: []const u8 }{
        .{ .arch = .aarch64, .os = .linux, .name = "aarch64-linux" },
        .{ .arch = .x86_64, .os = .linux, .name = "x86_64-linux" },
        .{ .arch = .wasm32, .os = .wasi, .name = "wasm32-wasi" },
        .{ .arch = .x86_64, .os = .macos, .name = "x86_64-macos" },
    };
    inline for (cross_targets) |ct| {
        const is_wasm = ct.os == .wasi;
        const is_linux = ct.os == .linux;

        const cross_target = b.resolveTargetQuery(.{ .cpu_arch = ct.arch, .os_tag = ct.os });
        const cross_opts = b.addOptions();

        // Core features — disable those requiring OS syscalls on WASM
        cross_opts.addOption(bool, "feat_gpu", !is_wasm);
        cross_opts.addOption(bool, "feat_ai", true);
        cross_opts.addOption(bool, "feat_database", !is_wasm);
        cross_opts.addOption(bool, "feat_network", !is_wasm);
        cross_opts.addOption(bool, "feat_profiling", !is_wasm);
        cross_opts.addOption(bool, "feat_web", !is_wasm);
        cross_opts.addOption(bool, "feat_pages", !is_wasm);
        cross_opts.addOption(bool, "feat_analytics", true);
        cross_opts.addOption(bool, "feat_cloud", !is_wasm);
        cross_opts.addOption(bool, "feat_auth", true);
        cross_opts.addOption(bool, "feat_messaging", true);
        cross_opts.addOption(bool, "feat_cache", true);
        cross_opts.addOption(bool, "feat_storage", !is_wasm);
        cross_opts.addOption(bool, "feat_search", true);
        cross_opts.addOption(bool, "feat_mobile", false);
        cross_opts.addOption(bool, "feat_gateway", true);
        cross_opts.addOption(bool, "feat_benchmarks", true);
        // WASM: disable compute (requires OS threading/process spawning)
        cross_opts.addOption(bool, "feat_compute", !is_wasm);
        cross_opts.addOption(bool, "feat_documents", true);
        // Desktop: only macOS (uses NSStatusItem / ObjC runtime)
        cross_opts.addOption(bool, "feat_desktop", !is_wasm and !is_linux);
        // TUI: needs POSIX termios — not WASM
        cross_opts.addOption(bool, "feat_tui", !is_wasm);
        cross_opts.addOption(bool, "feat_llm", true);
        cross_opts.addOption(bool, "feat_training", true);
        cross_opts.addOption(bool, "feat_vision", true);
        cross_opts.addOption(bool, "feat_explore", true);
        cross_opts.addOption(bool, "feat_reasoning", true);
        cross_opts.addOption(bool, "feat_lsp", !is_wasm);
        cross_opts.addOption(bool, "feat_mcp", !is_wasm);

        // GPU backends — per-platform availability
        cross_opts.addOption(bool, "gpu_metal", false); // Only for macOS native builds with -Dgpu-backend=metal
        cross_opts.addOption(bool, "gpu_cuda", false); // Only with NVIDIA driver present
        cross_opts.addOption(bool, "gpu_vulkan", false); // Only with Vulkan loader present
        cross_opts.addOption(bool, "gpu_webgpu", false); // Only for WASM browser targets
        cross_opts.addOption(bool, "gpu_opengl", false);
        cross_opts.addOption(bool, "gpu_opengles", false);
        cross_opts.addOption(bool, "gpu_webgl2", false);
        cross_opts.addOption(bool, "gpu_stdgpu", !is_wasm); // Software fallback on all native targets
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

    // ── Doctor step ────────────────────────────────────────────────────
    const doctor_step = b.step("doctor", "Report build configuration and diagnostics");
    const doc1 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\ABI Build Configuration Report
            \\==============================
            \\Features:
            \\  feat_ai={} feat_gpu={} feat_database={} feat_network={}
            \\  feat_profiling={} feat_web={} feat_pages={} feat_analytics={}
            \\  feat_cloud={} feat_auth={} feat_messaging={} feat_cache={}
            \\  feat_storage={} feat_search={} feat_mobile={} feat_gateway={}
            \\  feat_benchmarks={} feat_compute={} feat_documents={} feat_desktop={}
        , .{
            feat_ai,         feat_gpu,     feat_database,  feat_network,
            feat_profiling,  feat_web,     feat_pages,     feat_analytics,
            feat_cloud,      feat_auth,    feat_messaging, feat_cache,
            feat_storage,    feat_search,  feat_mobile,    feat_gateway,
            feat_benchmarks, feat_compute, feat_documents, feat_desktop,
        }),
    });
    const doc2 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\AI Sub-features:
            \\  feat_llm={} feat_training={} feat_vision={} feat_explore={} feat_reasoning={}
            \\Protocols:
            \\  feat_lsp={} feat_mcp={}
            \\GPU Backends:
            \\  metal={} cuda={} vulkan={} webgpu={} opengl={}
            \\  opengles={} webgl2={} stdgpu={} fpga={} tpu={}
        , .{
            feat_llm,   feat_training, feat_vision,  feat_explore, feat_reasoning,
            feat_lsp,   feat_mcp,      gpu_metal,    gpu_cuda,     gpu_vulkan,
            gpu_webgpu, gpu_opengl,    gpu_opengles, gpu_webgl2,   gpu_stdgpu,
            gpu_fpga,   gpu_tpu,
        }),
    });
    doc2.step.dependOn(&doc1.step);
    doctor_step.dependOn(&doc2.step);
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
