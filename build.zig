const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Feature Flags - Enabled by default
    const feat_ai = b.option(bool, "feat-ai", "Enable AI features") orelse true;
    const feat_gpu = b.option(bool, "feat-gpu", "Enable GPU acceleration") orelse true;
    const feat_tui = b.option(bool, "feat-tui", "Enable TUI features") orelse true;
    const feat_accelerator = b.option(bool, "feat-accelerator", "Enable accelerator backend selection") orelse true;
    const feat_shader = b.option(bool, "feat-shader", "Enable Zig shader validation backend") orelse true;
    const feat_mlir = b.option(bool, "feat-mlir", "Enable textual MLIR lowering backend") orelse true;
    const feat_mobile = b.option(bool, "feat-mobile", "Enable mobile platform feature flag") orelse true;
    const feat_wdbx = b.option(bool, "feat-wdbx", "Enable WDBX vector store and block memory") orelse true;
    const feat_os_control = b.option(bool, "feat-os-control", "Enable OS command policy controls") orelse true;
    const feat_hash = b.option(bool, "feat-hash", "Enable stable portable hashing utilities") orelse true;
    const feat_metrics = b.option(bool, "feat-metrics", "Enable lightweight in-process metrics for observability") orelse true;
    const feat_telemetry = b.option(bool, "feat-telemetry", "Enable lightweight telemetry event emission") orelse true;
    const feat_nn = b.option(bool, "feat-nn", "Enable the pure-Zig nn char-LM trainer") orelse true;
    const feat_sea = b.option(bool, "feat-sea", "Enable SEA self-learning loop (evidence-augmented completion)") orelse true;
    const feat_foundationmodels = b.option(bool, "feat-foundationmodels", "Enable Apple FoundationModels on-device connector (macOS only)") orelse true;
    const test_filter = b.option([]const u8, "test-filter", "Only run tests whose names contain this text");
    const test_filters: []const []const u8 = if (test_filter) |filter| &.{filter} else &.{};

    const options = b.addOptions();
    options.addOption(bool, "feat_ai", feat_ai);
    options.addOption(bool, "feat_gpu", feat_gpu);
    options.addOption(bool, "feat_tui", feat_tui);
    options.addOption(bool, "feat_accelerator", feat_accelerator);
    options.addOption(bool, "feat_shader", feat_shader);
    options.addOption(bool, "feat_mlir", feat_mlir);
    options.addOption(bool, "feat_mobile", feat_mobile);
    options.addOption(bool, "feat_wdbx", feat_wdbx);
    options.addOption(bool, "feat_os_control", feat_os_control);
    options.addOption(bool, "feat_hash", feat_hash);
    options.addOption(bool, "feat_metrics", feat_metrics);
    options.addOption(bool, "feat_telemetry", feat_telemetry);
    options.addOption(bool, "feat_nn", feat_nn);
    options.addOption(bool, "feat_sea", feat_sea);
    options.addOption(bool, "feat_foundationmodels", feat_foundationmodels);
    const options_mod = options.createModule();

    // Apple FoundationModels Swift bridge (macOS + flag only). FoundationModels
    // is a Swift-only framework, so the on-device path needs a Swift `@c` shim
    // (`src/connectors/fm_shim.swift`) exposing C entry points. We build it once
    // here as a *dynamic library* with `swiftc -emit-library -parse-as-library`:
    // swiftc performs the full Swift autolink, so the dylib carries its own
    // runtime deps (`/usr/lib/swift/libswiftCore.dylib`, `libswift_Concurrency`,
    // FoundationModels.framework, …), all resolved by dyld at runtime from the OS.
    // Zig then links each module against just the dylib (`-labi_fm_shim`), which
    // only has to resolve the two exported C symbols — it does NOT need to know
    // about the swift runtime or the framework. (We chose a dylib over a bare `.o`
    // precisely because Zig's linker does not honor the Swift `-l`/`-framework`
    // autolink directives embedded in a Swift object file.) An `@rpath` install
    // name + a per-module rpath into the build-cache dir let the binary find the
    // dylib at runtime. The flag defaults on, but the shim is still strictly
    // gated on an arm64 macOS target.
    const FmShim = struct {
        /// Build-cache directory that holds `libabi_fm_shim.dylib`; used as both
        /// the link-time library search path and the runtime rpath.
        dir: std.Build.LazyPath,
    };
    const fm_shim: ?FmShim = if (target.result.os.tag == .macos and target.result.cpu.arch == .aarch64 and feat_foundationmodels) blk: {
        const sdk_path = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), " \r\n\t");

        const compile = b.addSystemCommand(&.{ "xcrun", "swiftc" });
        compile.addArgs(&.{ "-emit-library", "-O", "-parse-as-library", "-target", "arm64-apple-macosx26.0", "-sdk", sdk_path });
        compile.addArgs(&.{ "-Xlinker", "-install_name", "-Xlinker", "@rpath/libabi_fm_shim.dylib" });
        compile.addFileArg(b.path("src/connectors/fm_shim.swift"));
        compile.addArg("-o");
        const dylib = compile.addOutputFileArg("libabi_fm_shim.dylib");

        break :blk .{ .dir = dylib.dirname() };
    } else null;

    // Plugin Registry Generation. This tool is *run* during the build
    // (`addRunArtifact`), so it must be built for the host graph — on a cross
    // build the host cannot exec a target binary.
    const gen_plugin_registry = b.addExecutable(.{
        .name = "gen_plugin_registry",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_plugin_registry.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    const run_gen_plugin_registry = b.addRunArtifact(gen_plugin_registry);
    run_gen_plugin_registry.addArg("src/plugins");
    run_gen_plugin_registry.addArg("src/plugin_registry.zig");

    // ABI Module
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        // Explicit libc: sockets/getpid paths need it on Linux. On macOS Metal/
        // objc often pull libc transitively, but tests that do not link frameworks
        // (and Linux hosts) still require an explicit link.
        .link_libc = true,
    });
    abi_mod.addImport("build_options", options_mod);

    if (target.result.os.tag == .macos) {
        // When the macOS target is selected *explicitly* (e.g.
        // `-Dtarget=aarch64-macos`), Zig does not auto-add the host SDK search
        // paths that a native build inherits, so `objc`/frameworks fail to
        // resolve. Add the active SDK's framework + lib dirs in that case only,
        // on a macOS host. The native build (the default `./build.sh check`
        // gate) takes `isNative()` and is left completely untouched.
        if (builtin.os.tag == .macos and !target.query.isNative()) {
            const sdk = std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), " \r\n\t");
            abi_mod.addFrameworkPath(.{ .cwd_relative = b.fmt("{s}/System/Library/Frameworks", .{sdk}) });
            abi_mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/usr/lib", .{sdk}) });
        }

        // OS-keychain credential backend (Security.framework SecItem C API),
        // opt-in at runtime via ABI_CREDENTIALS_BACKEND=keychain — see
        // src/foundation/keychain.zig. Linked unconditionally for macOS
        // (not gated by feat_gpu): src/foundation/credentials.zig always
        // compiles regardless of the GPU flag.
        abi_mod.linkFramework("Security", .{});
        abi_mod.linkFramework("CoreFoundation", .{});

        if (feat_gpu) {
            abi_mod.linkFramework("Metal", .{});
            abi_mod.linkFramework("Foundation", .{});
            abi_mod.linkSystemLibrary("objc", .{});
        }
    }

    // Apple FoundationModels on-device connector (macOS + flag only). Links the
    // Swift shim dylib (`-labi_fm_shim`) and records an rpath so the binary finds
    // it at runtime; the dylib carries its own swift-runtime + FoundationModels
    // deps. Non-macOS / flag-off builds are untouched, keeping the default
    // `./build.sh check` link surface unchanged.
    if (fm_shim) |shim| {
        abi_mod.addLibraryPath(shim.dir);
        abi_mod.linkSystemLibrary("abi_fm_shim", .{});
        abi_mod.addRPath(shim.dir);
    }

    // CLI Executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            // link_libc is per-module (not inherited from abi_mod).
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    exe.step.dependOn(&run_gen_plugin_registry.step);
    b.installArtifact(exe);

    const mcp_exe = b.addExecutable(.{
        .name = "abi-mcp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    b.installArtifact(mcp_exe);

    // Steps
    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const cli_step = b.step("cli", "Build ABI CLI");
    cli_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    const mcp_step = b.step("mcp", "Build MCP server");
    mcp_step.dependOn(&b.addInstallArtifact(mcp_exe, .{}).step);

    // Tests
    const mod_tests = b.addTest(.{
        .root_module = abi_mod,
        .filters = test_filters,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const connector_test_mod = b.createModule(.{
        .root_source_file = b.path("src/connectors/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "build_options", .module = options_mod },
        },
    });
    // The connector test module compiles `fm.zig`; under the flag it links the
    // same Swift shim dylib here too (this module does not import abi_mod, so it
    // cannot inherit abi_mod's link directives).
    if (fm_shim) |shim| {
        connector_test_mod.addLibraryPath(shim.dir);
        connector_test_mod.linkSystemLibrary("abi_fm_shim", .{});
        connector_test_mod.addRPath(shim.dir);
    }
    const connector_tests = b.addTest(.{
        .root_module = connector_test_mod,
        .filters = test_filters,
    });
    const run_connector_tests = b.addRunArtifact(connector_tests);

    // The CLI ships as an executable, so the inline tests in `src/cli/*` (the
    // declarative registry + generic argument parser) have no test artifact and
    // would never run. `cli_test.zig` aggregates them and covers the migrated
    // argument specs; it mirrors the `abi` exe module imports.
    const cli_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli_test.zig"),
            .target = target,
            .optimize = optimize,
            // link_libc is per-module, not inherited from imported abi_mod;
            // needed so Linux cross builds resolve socket/getpid paths.
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    cli_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_cli_tests = b.addRunArtifact(cli_tests);

    const cli_test_step = b.step("test-cli", "Run CLI framework (registry + argument parser) tests");
    cli_test_step.dependOn(&run_cli_tests.step);

    // Plugin test aggregator — exercises the bundled plugin mod/stub refAllDecls
    // blocks (they are loaded at runtime by path, so their inline tests would
    // never run without an explicit aggregator).
    const plugin_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/plugins_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    plugin_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_plugin_tests = b.addRunArtifact(plugin_tests);

    const plugin_test_step = b.step("test-plugins", "Run bundled plugin refAllDecls coverage");
    plugin_test_step.dependOn(&run_plugin_tests.step);

    const feature_contract_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/contracts/feature_modules.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    feature_contract_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_feature_contract_tests = b.addRunArtifact(feature_contract_tests);

    const feature_contract_step = b.step("test-feature-contracts", "Run focused feature module contract tests");
    feature_contract_step.dependOn(&run_feature_contract_tests.step);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_connector_tests.step);
    test_step.dependOn(&run_cli_tests.step);
    test_step.dependOn(&run_plugin_tests.step);
    test_step.dependOn(&run_feature_contract_tests.step);

    // Integration Tests
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/integration_tests.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_integration_step = b.step("test-integration", "Run integration tests");
    test_integration_step.dependOn(&run_integration_tests.step);

    // Benchmarks
    const benchmarks = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/benchmarks.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    const run_benchmarks = b.addRunArtifact(benchmarks);

    const bench_step = b.step("benchmarks", "Run benchmark suite");
    bench_step.dependOn(&run_benchmarks.step);

    const cli_usage_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/usage.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mcp_handlers_mod = b.createModule(.{
        .root_source_file = b.path("src/mcp/handlers.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "abi", .module = abi_mod },
            .{ .name = "build_options", .module = options_mod },
        },
    });
    const contract_surface_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/contracts/surface.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "cli_usage", .module = cli_usage_mod },
                .{ .name = "mcp_handlers", .module = mcp_handlers_mod },
            },
        }),
        .filters = test_filters,
    });
    const run_contract_surface_tests = b.addRunArtifact(contract_surface_tests);

    const contract_mcp_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/contracts/mcp_tools.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "cli_usage", .module = cli_usage_mod },
                .{ .name = "mcp_handlers", .module = mcp_handlers_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    const run_contract_mcp_tests = b.addRunArtifact(contract_mcp_tests);

    // handlers.zig owns the tool catalog + per-tool validation specs but is only
    // imported as a module by the contract tests above, so its in-file tests
    // (FieldSpec/schema parity) would never run without their own artifact —
    // same rationale as the mcp_server tests below.
    const mcp_handlers_tests = b.addTest(.{
        .root_module = mcp_handlers_mod,
        .filters = test_filters,
    });
    mcp_handlers_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_mcp_handlers_tests = b.addRunArtifact(mcp_handlers_tests);

    const contract_mcp_step = b.step("test-mcp-contracts", "Run MCP tool contract tests");
    contract_mcp_step.dependOn(&run_contract_mcp_tests.step);
    contract_mcp_step.dependOn(&run_mcp_handlers_tests.step);
    test_step.dependOn(&run_mcp_handlers_tests.step);

    // The MCP server transport (stdio + HTTP) is only reachable through the
    // `abi-mcp` executable, so its in-file tests are wired here as their own
    // test artifact; without this the HTTP read-loop tests would never run.
    const mcp_server_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp/server.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
        .filters = test_filters,
    });
    mcp_server_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_mcp_server_tests = b.addRunArtifact(mcp_server_tests);

    const mcp_server_step = b.step("test-mcp-server", "Run MCP server transport tests");
    mcp_server_step.dependOn(&run_mcp_server_tests.step);
    test_step.dependOn(&run_mcp_server_tests.step);

    const contract_plugin_registry_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/contracts/plugin_registry.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
        .filters = test_filters,
    });
    contract_plugin_registry_tests.step.dependOn(&run_gen_plugin_registry.step);
    const run_contract_plugin_registry_tests = b.addRunArtifact(contract_plugin_registry_tests);

    const contract_public_docs_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/contracts/public_docs.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
        .filters = test_filters,
    });
    const run_contract_public_docs_tests = b.addRunArtifact(contract_public_docs_tests);

    const contract_step = b.step("test-contracts", "Run public API contract tests");
    contract_step.dependOn(&run_contract_surface_tests.step);
    contract_step.dependOn(&run_contract_mcp_tests.step);
    contract_step.dependOn(&run_mcp_server_tests.step);
    contract_step.dependOn(&run_contract_plugin_registry_tests.step);
    contract_step.dependOn(&run_contract_public_docs_tests.step);

    const run_contract_cli = b.addSystemCommand(&.{ "bash", "tools/run_contract_cli.sh" });
    run_contract_cli.step.dependOn(b.getInstallStep());

    const feature_stub_check = b.addSystemCommand(&.{ "bash", "tools/check_feature_stubs.sh" });
    feature_stub_check.step.dependOn(&exe.step);

    const tui_smoke = b.addSystemCommand(&.{ "bash", "tools/run_tui_smoke.sh" });
    tui_smoke.step.dependOn(b.getInstallStep());

    // Fmt and Parity Checks
    const fmt_check = b.addSystemCommand(&.{ "zig", "fmt", "--check", "src", "tests", "tools", "build.zig" });
    const fmt = b.addSystemCommand(&.{ "zig", "fmt", "src", "tests", "tools", "build.zig" });

    // check_parity is *run* during the build (`addRunArtifact`), so it must be
    // built for the host graph to stay executable under a cross build.
    const parity_exe = b.addExecutable(.{
        .name = "check_parity",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_parity.zig"),
            .target = b.graph.host,
            .optimize = optimize,
        }),
    });
    const parity_check = b.addRunArtifact(parity_exe);

    const check_step = b.step("check", "Run all checks (build + tests + lint + parity)");
    check_step.dependOn(&exe.step);
    check_step.dependOn(&mcp_exe.step);
    check_step.dependOn(test_step);
    check_step.dependOn(&run_feature_contract_tests.step);
    check_step.dependOn(contract_step);
    check_step.dependOn(&run_contract_cli.step);
    check_step.dependOn(&feature_stub_check.step);
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&parity_check.step);

    const full_check_step = b.step("full-check", "Run check, integration tests, benchmarks, dashboard smoke, and agent TUI smoke");
    full_check_step.dependOn(check_step);
    full_check_step.dependOn(test_integration_step);
    full_check_step.dependOn(bench_step);
    full_check_step.dependOn(&tui_smoke.step);

    const lint_step = b.step("lint", "Check Zig formatting");
    lint_step.dependOn(&fmt_check.step);

    const fix_step = b.step("fix", "Format Zig sources");
    fix_step.dependOn(&fmt.step);

    const check_parity_step = b.step("check-parity", "Check feature mod/stub API parity");
    check_parity_step.dependOn(&parity_check.step);

    // Opt-in cross-compile smoke: compiles+links the CLI for the supported
    // non-native targets (Linux/Windows/macOS). Deliberately NOT wired into
    // `check` — the cold cross builds are slow. See tools/cross_smoke.sh.
    const cross_smoke = b.addSystemCommand(&.{ "bash", "tools/cross_smoke.sh" });
    const cross_smoke_step = b.step("cross-smoke", "Compile-check the CLI for Linux/Windows/macOS cross targets (opt-in)");
    cross_smoke_step.dependOn(&cross_smoke.step);
}
