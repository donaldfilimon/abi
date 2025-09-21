const std = @import("std");

const VersionParseError = error{
    MissingVersion,
    MalformedVersion,
};

fn readPackageVersion(allocator: std.mem.Allocator) ![]const u8 {
    const contents = try std.fs.cwd().readFileAlloc(allocator, "build.zig.zon", 4 * 1024);
    defer allocator.free(contents);

    const prefix = ".version = \"";
    const start_index = std.mem.indexOf(u8, contents, prefix) orelse return VersionParseError.MissingVersion;
    const after_prefix = contents[start_index + prefix.len ..];
    const end_index = std.mem.indexOfScalar(u8, after_prefix, '"') orelse return VersionParseError.MalformedVersion;
    const version_slice = after_prefix[0..end_index];

    return try allocator.dupe(u8, version_slice);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const package_version = readPackageVersion(b.allocator) catch |err| {
        std.debug.panic("failed to read package version: {s}", .{@errorName(err)});
    };

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_version", package_version);

    const enable_vulkan = b.option(bool, "enable-vulkan", "Link Vulkan loader for GPU compute backends") orelse false;
    const enable_cuda = b.option(bool, "enable-cuda", "Link CUDA driver/runtime for GPU acceleration") orelse false;
    const enable_metal = b.option(bool, "enable-metal", "Link Apple Metal frameworks for GPU acceleration") orelse false;

    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addOptions("build_options", build_options);

    const main_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_module.addImport("abi", abi_mod);
    main_module.addOptions("build_options", build_options);

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = main_module,
    });
    b.installArtifact(exe);

    const unit_tests = b.addTest(.{
        .root_module = main_module,
    });

    if (enable_vulkan) {
        const vulkan_lib = switch (target.result.os.tag) {
            .windows => "vulkan-1",
            .macos => "vulkan",
            else => "vulkan",
        };
        exe.linkSystemLibrary(vulkan_lib);
        unit_tests.linkSystemLibrary(vulkan_lib);
    }

    if (enable_cuda) {
        const cuda_lib = switch (target.result.os.tag) {
            .windows => "nvcuda",
            .macos => "cuda",
            else => "cuda",
        };
        exe.linkSystemLibrary(cuda_lib);
        unit_tests.linkSystemLibrary(cuda_lib);
    }

    if (enable_metal) {
        switch (target.result.os.tag) {
            .macos, .ios, .tvos, .watchos => {
                exe.linkFramework("Metal");
                exe.linkFramework("MetalKit");
                exe.linkFramework("QuartzCore");
                unit_tests.linkFramework("Metal");
                unit_tests.linkFramework("MetalKit");
                unit_tests.linkFramework("QuartzCore");
            },
            else => {},
        }
    }

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);
}
