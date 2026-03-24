const std = @import("std");

pub const DarwinRole = enum {
    static_lib,
    executable,
    test_artifact,
    parity_test,
};

pub fn linkDarwinArtifact(
    artifact: *std.Build.Step.Compile,
    role: DarwinRole,
    feat_gpu: bool,
    gpu_metal: bool,
) void {
    switch (role) {
        .static_lib => {
            for ([_][]const u8{ "System", "c" }) |lib| {
                artifact.root_module.linkSystemLibrary(lib, .{});
            }

            if (feat_gpu) {
                artifact.root_module.linkFramework("Accelerate", .{});
            }

            artifact.root_module.linkFramework("IOKit", .{});
            artifact.root_module.linkSystemLibrary("objc", .{});

            if (gpu_metal) {
                for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |framework| {
                    artifact.root_module.linkFramework(framework, .{});
                }
            }
        },
        .executable => {
            artifact.root_module.linkSystemLibrary("System", .{});
            artifact.root_module.linkSystemLibrary("c", .{});
            artifact.root_module.linkSystemLibrary("objc", .{});
            artifact.root_module.linkFramework("IOKit", .{});
            artifact.root_module.linkFramework("CoreFoundation", .{});
            artifact.root_module.linkFramework("CoreGraphics", .{});
            if (feat_gpu) {
                artifact.root_module.linkFramework("Accelerate", .{});
            }
        },
        .test_artifact => {
            artifact.root_module.linkSystemLibrary("System", .{});
            artifact.root_module.linkSystemLibrary("c", .{});
            artifact.root_module.linkSystemLibrary("objc", .{});
            artifact.root_module.linkFramework("IOKit", .{});
            artifact.root_module.linkFramework("CoreFoundation", .{});
            artifact.root_module.linkFramework("CoreGraphics", .{});
            if (feat_gpu) {
                artifact.root_module.linkFramework("Accelerate", .{});
            }
            if (gpu_metal) {
                for ([_][]const u8{ "Metal", "MetalPerformanceShaders" }) |framework| {
                    artifact.root_module.linkFramework(framework, .{});
                }
            }
        },
        .parity_test => {
            artifact.root_module.linkSystemLibrary("c", .{});
            artifact.root_module.linkSystemLibrary("objc", .{});
            artifact.root_module.linkFramework("IOKit", .{});
            artifact.root_module.linkFramework("CoreFoundation", .{});
            artifact.root_module.linkFramework("CoreGraphics", .{});
            if (feat_gpu) {
                artifact.root_module.linkFramework("Accelerate", .{});
            }
        },
    }
}
