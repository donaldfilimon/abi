const std = @import("std");
const options_mod = @import("options.zig");
const modules = @import("modules.zig");
const targets = @import("targets.zig");

/// Register WASM build and check-wasm steps.
///
/// WASM builds disable several feature modules (database, network, gpu,
/// profiling, web, cloud, storage) since they require OS-level I/O.
/// Returns the check-wasm step on success, or null if the WASM entry point
/// (`src/bindings/wasm/abi_wasm.zig`) does not exist.
pub fn addWasmBuild(
    b: *std.Build,
    options: options_mod.BuildOptions,
    abi_module: *std.Build.Module,
    optimize: std.builtin.OptimizeMode,
) ?*std.Build.Step {
    if (!targets.pathExists(b, "src/bindings/wasm/abi_wasm.zig")) {
        _ = b.step("check-wasm", "Check WASM compilation (bindings not available)");
        _ = b.step("wasm", "Build WASM bindings (bindings not available)");
        return null;
    }

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    var wasm_opts = options;
    wasm_opts.feat_database = false;
    wasm_opts.feat_network = false;
    wasm_opts.feat_gpu = false;
    wasm_opts.feat_profiling = false;
    wasm_opts.feat_web = false;
    wasm_opts.feat_cloud = false;
    wasm_opts.feat_storage = false;
    wasm_opts.gpu_backends = &.{};

    const wasm_build_opts = modules.createBuildOptionsModule(b, wasm_opts);
    const wasm_shared_services = modules.createSharedServicesModule(b, wasm_build_opts, wasm_target, optimize);
    const wasm_core_module = modules.createCoreModule(b, wasm_target, optimize, wasm_build_opts);
    const abi_wasm = b.addModule("abi-wasm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    modules.wireAbiImports(abi_wasm, wasm_build_opts, wasm_shared_services, wasm_core_module);
    _ = abi_module;

    const wasm_lib = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bindings/wasm/abi_wasm.zig"),
            .target = wasm_target,
            .optimize = optimize,
        }),
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;
    wasm_lib.root_module.addImport("abi", abi_wasm);

    const check_wasm_step = b.step("check-wasm", "Check WASM compilation");
    check_wasm_step.dependOn(&wasm_lib.step);

    b.step("wasm", "Build WASM bindings").dependOn(
        &b.addInstallArtifact(wasm_lib, .{
            .dest_dir = .{ .override = .{ .custom = "wasm" } },
        }).step,
    );

    return check_wasm_step;
}
