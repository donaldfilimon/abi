const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const exe = b.addExecutable(.{ .name = "test-smc", .root_source_file = b.path("src/test_smc.zig"), .target = b.standardTargetOptions(.{}), .optimize = .Debug, .use_llvm = true, .use_lld = true, .linkage = .dynamic, .libc = true });

    // Add necessary modules/packages if required.
    // The ABI framework structure seems complex; this test might need access to foundation.
    // For now, let's see if it builds with just path adjustments.
    exe.addIncludePath(b.path("src"));
    b.installArtifact(exe);
}
