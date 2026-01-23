const std = @import("std");
const loader = @import("loader.zig");

test "CUDA loader basic functionality" {
    // Ensure that attempting to load does not panic even when driver missing.
    const allocator = std.testing.allocator;
    const functions = loader.load(allocator) catch |err| {
        // If library not found, we expect LoadError.LibraryNotFound.
        try std.testing.expect(err == loader.LoadError.LibraryNotFound);
        return;
    };
    // If the library was loaded, basic core symbols must be present.
    try std.testing.expect(functions.core.cuInit != null);
    try std.testing.expect(functions.core.cuDeviceGetCount != null);
    // Clean up.
    loader.unload();
}
