//! ABI build root — Zig 0.17-dev, self-contained.
const std = @import("std");
const builtin = @import("builtin");
const build_flags = @import("build/flags.zig");
const build_cross = @import("build/cross.zig");
const build_linking = @import("build/linking.zig");
const build_validation = @import("build/validation.zig");
const FeatureFlags = build_flags.FeatureFlags;
const hasBackend = build_flags.hasBackend;
const addAllBuildOptions = build_flags.addAllBuildOptions;
const linkIfDarwin = build_linking.linkIfDarwin;

fn addScriptStep(b: *std.Build, name: []const u8, description: []const u8, script: []const u8) *std.Build.Step {
    const step = b.step(name, description);
    const run = b.addSystemCommand(&.{ "bash", script });
    step.dependOn(&run.step);
    return step;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    //  Feature flags (automated via reflection)
    var flags: FeatureFlags = undefined;
    inline for (@typeInfo(FeatureFlags).@"struct".fields) |field| {
        if (std.mem.startsWith(u8, field.name, "feat_")) {
            const flag_name = "feat-" ++ field.name[5..];
            const desc = "Enable " ++ field.name[5..];

            // Define defaults based on project requirements
            var default_val = true;
            if (std.mem.eql(u8, field.name, "feat_mobile") or std.mem.eql(u8, field.name, "feat_tui") or std.mem.eql(u8, field.name, "feat_external_ai")) {
                default_val = false;
            }
            @field(flags, field.name) = b.option(bool, flag_name, desc) orelse default_val;
        }
    }

    // GPU backend flags
    const gpu_backend_str = b.option([]const u8, "gpu-backend", "GPU backends: metal,cuda,vulkan,webgpu,opengl,opengles,webgl2,stdgpu,fpga,tpu (comma-separated)");
    flags.gpu_metal = flags.feat_gpu and hasBackend(gpu_backend_str, "metal");
    flags.gpu_cuda = flags.feat_gpu and hasBackend(gpu_backend_str, "cuda");
    flags.gpu_vulkan = flags.feat_gpu and hasBackend(gpu_backend_str, "vulkan");
    flags.gpu_webgpu = flags.feat_gpu and hasBackend(gpu_backend_str, "webgpu");
    flags.gpu_opengl = flags.feat_gpu and hasBackend(gpu_backend_str, "opengl");
    flags.gpu_opengles = flags.feat_gpu and hasBackend(gpu_backend_str, "opengles");
    flags.gpu_webgl2 = flags.feat_gpu and hasBackend(gpu_backend_str, "webgl2");
    flags.gpu_stdgpu = flags.feat_gpu and (gpu_backend_str == null or hasBackend(gpu_backend_str, "stdgpu"));
    flags.gpu_fpga = flags.feat_gpu and hasBackend(gpu_backend_str, "fpga");
    flags.gpu_tpu = flags.feat_gpu and hasBackend(gpu_backend_str, "tpu");

    if (gpu_backend_str) |str| {
        const valid_backends: []const []const u8 = &.{
            "metal",    "cuda",   "vulkan", "webgpu", "opengl",
            "opengles", "webgl2", "stdgpu", "fpga",   "tpu",
        };
        var backend_it = std.mem.splitScalar(u8, str, ',');
        while (backend_it.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " ");
            if (trimmed.len == 0) continue;
            var found = false;
            for (valid_backends) |vb| {
                if (std.mem.eql(u8, trimmed, vb)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.log.err("Unknown GPU backend '{s}'. Valid: metal, cuda, vulkan, webgpu, opengl, opengles, webgl2, stdgpu, fpga, tpu", .{trimmed});
                std.process.exit(1);
            }
        }
    }
    if (flags.gpu_metal and flags.gpu_cuda)
        std.log.warn("Both Metal and CUDA enabled — unusual; intended for cross-compilation only", .{});

    //  Feature dependency warnings
    if (!flags.feat_ai) {
        if (flags.feat_llm) std.log.warn("feat_llm requires feat_ai — llm will be stubbed", .{});
        if (flags.feat_training) std.log.warn("feat_training requires feat_ai — training will be stubbed", .{});
        if (flags.feat_vision) std.log.warn("feat_vision requires feat_ai — vision will be stubbed", .{});
        if (flags.feat_explore) std.log.warn("feat_explore requires feat_ai — explore will be stubbed", .{});
        if (flags.feat_reasoning) std.log.warn("feat_reasoning requires feat_ai — reasoning will be stubbed", .{});
    }
    if (flags.feat_ai and !flags.feat_connectors)
        std.log.warn("feat_ai requires feat_connectors — AI connector imports will fail", .{});
    if (flags.feat_mcp and !flags.feat_database)
        std.log.info("feat_mcp benefits from feat_database for DB tools", .{});

    const build_opts = b.addOptions();
    const pkg_version = comptime blk: {
        const zon = @import("build.zig.zon");
        break :blk zon.version;
    };
    addAllBuildOptions(build_opts, flags, pkg_version, builtin.zig_version_string);
    const build_options_module = build_opts.createModule();

    const common_module = b.createModule(.{
        .root_source_file = b.path("src/common/env_gate.zig"),
        .target = target,
        .optimize = optimize,
    });

    const cross_steps = build_cross.addSteps(.{
        .b = b,
        .target = target,
        .optimize = optimize,
        .build_options_module = build_options_module,
        .package_version = pkg_version,
    });

    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);
    abi_module.addImport("common", common_module);

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
    static_lib.root_module.addImport("common", common_module);
    linkIfDarwin(static_lib, .static_lib, flags.feat_gpu, flags.gpu_metal);
    b.installArtifact(static_lib);
    const install_static_lib = b.addInstallArtifact(static_lib, .{});
    b.step("lib", "Build static ABI library").dependOn(&install_static_lib.step);

    const mcp_exe = b.addExecutable(.{
        .name = "abi-mcp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    mcp_exe.root_module.addImport("build_options", build_options_module);
    mcp_exe.root_module.addImport("common", common_module);
    linkIfDarwin(mcp_exe, .executable, flags.feat_gpu, flags.gpu_metal);
    const install_mcp = b.addInstallArtifact(mcp_exe, .{});
    b.step("mcp", "Build MCP stdio/SSE server").dependOn(&install_mcp.step);

    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli_exe.root_module.addImport("build_options", build_options_module);
    cli_exe.root_module.addImport("common", common_module);
    linkIfDarwin(cli_exe, .executable, flags.feat_gpu, flags.gpu_metal);
    const install_cli = b.addInstallArtifact(cli_exe, .{});
    b.step("cli", "Build ABI command-line interface").dependOn(&install_cli.step);

    const tools_step = b.step("tools", "Install abi and abi-mcp into the selected prefix");
    tools_step.dependOn(&install_cli.step);
    tools_step.dependOn(&install_mcp.step);

    const validation_steps = build_validation.addSteps(.{
        .b = b,
        .target = target,
        .optimize = optimize,
        .flags = flags,
        .build_options_module = build_options_module,
        .abi_module = abi_module,
        .common_module = common_module,
        .package_version = pkg_version,
    });

    const fmt_paths = &.{ "build.zig", "build", "src", "test" };
    const lint_fmt = b.addFmt(.{ .paths = fmt_paths, .check = true });
    b.step("lint", "Check formatting").dependOn(&lint_fmt.step);
    b.step("fix", "Fix formatting").dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = false }).step);

    const dev_step = b.step("dev", "Build fast developer targets (typecheck + CLI + MCP)");
    dev_step.dependOn(cross_steps.typecheck_step);
    dev_step.dependOn(&install_cli.step);
    dev_step.dependOn(&install_mcp.step);

    const quick_step = b.step("quick", "Run fast local validation (fmt + typecheck + parity)");
    quick_step.dependOn(&lint_fmt.step);
    quick_step.dependOn(cross_steps.typecheck_step);
    quick_step.dependOn(validation_steps.check_parity_step);

    const ci_step = b.step("ci", "Run CI validation (lint + tests + parity + MCP + cross-check)");
    ci_step.dependOn(&lint_fmt.step);
    ci_step.dependOn(validation_steps.test_step);
    ci_step.dependOn(validation_steps.check_parity_step);
    ci_step.dependOn(validation_steps.mcp_tests_step);
    ci_step.dependOn(cross_steps.cross_check_step);

    _ = addScriptStep(b, "mcp-health", "Check configured MCP HA health endpoints", "scripts/check-mcp-health.sh");
    _ = addScriptStep(b, "interop", "Check MCP health and optional ACP endpoint reachability", "scripts/check-interop.sh");
    _ = addScriptStep(b, "acp-endpoints", "List and check ACP endpoints from ACP_ENDPOINTS", "scripts/list-acp-endpoints.sh");

    const doctor_step = b.step("doctor", "Report build feature configuration");
    const doc1 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\ABI Build Configuration Report
            \\==============================
            \\Features:
            \\  feat_ai={} feat_gpu={} feat_database={} feat_network={}
            \\  feat_observability={} feat_web={} feat_pages={} feat_analytics={}
            \\  feat_cloud={} feat_auth={} feat_messaging={} feat_cache={}
            \\  feat_storage={} feat_search={} feat_mobile={} feat_gateway={}
            \\  feat_benchmarks={} feat_compute={} feat_documents={} feat_desktop={}
        , .{
            flags.feat_ai,            flags.feat_gpu,     flags.feat_database,  flags.feat_network,
            flags.feat_observability, flags.feat_web,     flags.feat_pages,     flags.feat_analytics,
            flags.feat_cloud,         flags.feat_auth,    flags.feat_messaging, flags.feat_cache,
            flags.feat_storage,       flags.feat_search,  flags.feat_mobile,    flags.feat_gateway,
            flags.feat_benchmarks,    flags.feat_compute, flags.feat_documents, flags.feat_desktop,
        }),
    });
    const doc2 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\AI Sub-features:
            \\  feat_llm={} feat_training={} feat_vision={} feat_explore={} feat_reasoning={} feat_external_ai={}
            \\Protocols:
            \\  feat_lsp={} feat_mcp={} feat_acp={} feat_ha={}
            \\GPU Backends:
            \\  metal={} cuda={} vulkan={} webgpu={} opengl={}
            \\  opengles={} webgl2={} stdgpu={} fpga={} tpu={}
        , .{
            flags.feat_llm,   flags.feat_training, flags.feat_vision, flags.feat_explore, flags.feat_reasoning, flags.feat_external_ai,
            flags.feat_lsp,   flags.feat_mcp,      flags.feat_acp,    flags.feat_ha,      flags.gpu_metal,      flags.gpu_cuda,
            flags.gpu_vulkan, flags.gpu_webgpu,    flags.gpu_opengl,  flags.gpu_opengles, flags.gpu_webgl2,     flags.gpu_stdgpu,
            flags.gpu_fpga,   flags.gpu_tpu,
        }),
    });
    doc2.step.dependOn(&doc1.step);
    doctor_step.dependOn(&doc2.step);

    const doctor_full_step = b.step("doctor-full", "Report toolchain, target, feature, and workflow diagnostics");
    const doctor_full = b.addSystemCommand(&.{
        "bash",
        "-c",
        b.fmt(
            \\set -e
            \\echo "ABI Doctor Full"
            \\echo "==============="
            \\echo "Package: {s}"
            \\echo "Zig (build): {s}"
            \\printf "Zig path: "; command -v zig || true
            \\printf "OS: "; uname -a
            \\echo "Optimize: {s}"
            \\echo "Target: requested by zig build -Dtarget (native if unset)"
            \\echo "GPU backend option: {s}"
            \\echo
            \\echo "Recommended next commands:"
            \\echo "  ./build.sh quick"
            \\echo "  ./build.sh dev"
            \\echo "  ./build.sh mcp-health"
            \\echo "  ./zig-out/bin/abi doctor"
        , .{
            pkg_version,
            builtin.zig_version_string,
            @tagName(optimize),
            gpu_backend_str orelse "stdgpu(default)",
        }),
    });
    doctor_full.step.dependOn(&doc2.step);
    doctor_full_step.dependOn(&doctor_full.step);
}
