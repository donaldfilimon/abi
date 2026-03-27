const std = @import("std");
const build_flags = @import("flags.zig");
const linking = @import("linking.zig");

pub const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
    package_version: []const u8 = "0.1.0",
};

pub const Steps = struct {
    typecheck_step: *std.Build.Step,
    cross_check_step: *std.Build.Step,
};

const CrossTarget = struct {
    arch: std.Target.Cpu.Arch,
    os: std.Target.Os.Tag,
    name: []const u8,
};

pub fn addSteps(ctx: Context) Steps {
    const typecheck_root_module = ctx.b.createModule(.{
        .root_source_file = ctx.b.path("src/root.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
    });
    typecheck_root_module.addImport("build_options", ctx.build_options_module);
    const typecheck_root = ctx.b.addObject(.{
        .name = "abi-typecheck",
        .root_module = typecheck_root_module,
    });
    linking.linkIfDarwin(typecheck_root, .static_lib, true, true);

    const gpu_policy_contract_module = ctx.b.createModule(.{
        .root_source_file = ctx.b.path("src/features/gpu/policy/target_contract.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
    });
    const gpu_policy_contract = ctx.b.addObject(.{
        .name = "abi-gpu-policy-contract",
        .root_module = gpu_policy_contract_module,
    });
    linking.linkIfDarwin(gpu_policy_contract, .static_lib, true, true);

    const typecheck_step = ctx.b.step("typecheck", "Compile-only validation for the requested target");
    typecheck_step.dependOn(&typecheck_root.step);
    typecheck_step.dependOn(&gpu_policy_contract.step);

    // Platform feature availability matrix:
    //
    //  Feature       | macOS | Linux | Windows | WASM/WASI | Freestanding
    //  --------------|-------|-------|---------|-----------|-------------
    //  feat_gpu      |  yes  |  yes  |   yes   |    no     |     no
    //  feat_ai       |  yes  |  yes  |   yes   |   yes     |    yes
    //  feat_database |  yes  |  yes  |   yes   |    no     |     no
    //  feat_network  |  yes  |  yes  |   yes   |    no     |     no
    //  feat_observability| yes |  yes  |   yes   |    no     |     no
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
    //  feat_acp      |  yes  |  yes  |   yes   |    no     |     no
    //  feat_ha       |  yes  |  yes  |   yes   |    no     |     no
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
    const cross_check_step = ctx.b.step("cross-check", "Verify cross-compilation for key targets");
    const cross_targets = [_]CrossTarget{
        .{ .arch = .aarch64, .os = .linux, .name = "aarch64-linux" },
        .{ .arch = .x86_64, .os = .linux, .name = "x86_64-linux" },
        .{ .arch = .wasm32, .os = .wasi, .name = "wasm32-wasi" },
        .{ .arch = .x86_64, .os = .macos, .name = "x86_64-macos" },
    };

    inline for (cross_targets) |ct| {
        const cross_mod = ctx.b.createModule(.{
            .root_source_file = ctx.b.path("src/root.zig"),
            .target = ctx.b.resolveTargetQuery(.{ .cpu_arch = ct.arch, .os_tag = ct.os }),
            .optimize = ctx.optimize,
        });
        cross_mod.addImport("build_options", crossBuildOptions(ctx.b, ct, ctx.package_version).createModule());

        const cross_lib = ctx.b.addLibrary(.{
            .name = "cross-" ++ ct.name,
            .root_module = cross_mod,
            .linkage = .static,
        });
        linking.linkIfDarwin(cross_lib, .static_lib, true, false);
        cross_check_step.dependOn(&cross_lib.step);
    }

    return .{
        .typecheck_step = typecheck_step,
        .cross_check_step = cross_check_step,
    };
}

/// Platform capabilities — single source of truth for what each OS supports.
/// Feature flags are derived from these capabilities, not from ad-hoc
/// `!is_wasm` / `!is_linux` expressions scattered across the flag list.
const PlatformCaps = struct {
    has_posix: bool, // filesystem, threads, sockets (everything except WASM)
    has_desktop: bool, // macOS-only NSStatusItem, native menus
    has_gpu_hw: bool, // can load GPU drivers (not WASM)
};

fn platformCaps(os: std.Target.Os.Tag) PlatformCaps {
    const is_wasm = os == .wasi;
    return .{
        .has_posix = !is_wasm,
        .has_desktop = os == .macos,
        .has_gpu_hw = !is_wasm,
    };
}

fn crossBuildOptions(b: *std.Build, ct: CrossTarget, package_version: []const u8) *std.Build.Step.Options {
    const caps = platformCaps(ct.os);
    const cross_opts = b.addOptions();

    build_flags.addAllBuildOptions(cross_opts, .{
        // ── Features requiring POSIX (filesystem, threads, sockets) ──
        .feat_gpu = caps.has_gpu_hw,
        .feat_database = caps.has_posix,
        .feat_network = caps.has_posix,
        .feat_observability = caps.has_posix,
        .feat_web = caps.has_posix,
        .feat_pages = caps.has_posix,
        .feat_cloud = caps.has_posix,
        .feat_storage = caps.has_posix,
        .feat_compute = caps.has_posix,
        .feat_tui = caps.has_posix,
        .feat_lsp = caps.has_posix,
        .feat_mcp = caps.has_posix,
        .feat_acp = caps.has_posix,
        .feat_ha = caps.has_posix,

        // ── Platform-specific ───────────────────────────────────────
        .feat_desktop = caps.has_desktop,
        .feat_mobile = false, // opt-in only for iOS cross-builds

        // ── Portable (work everywhere including WASM) ───────────────
        .feat_ai = true,
        .feat_analytics = true,
        .feat_auth = true,
        .feat_messaging = true,
        .feat_cache = true,
        .feat_search = true,
        .feat_gateway = true,
        .feat_benchmarks = true,
        .feat_documents = true,
        .feat_llm = true,
        .feat_training = true,
        .feat_vision = true,
        .feat_explore = true,
        .feat_reasoning = true,
        .feat_connectors = true,
        .feat_tasks = true,
        .feat_inference = true,

        // ── GPU backends (none enabled in cross-builds; host-specific) ──
        .gpu_metal = false,
        .gpu_cuda = false,
        .gpu_vulkan = false,
        .gpu_webgpu = false,
        .gpu_opengl = false,
        .gpu_opengles = false,
        .gpu_webgl2 = false,
        .gpu_stdgpu = caps.has_gpu_hw,
        .gpu_fpga = false,
        .gpu_tpu = false,
    }, package_version);

    return cross_opts;
}
