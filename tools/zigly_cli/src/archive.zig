const std = @import("std");

pub const ArchiveError = error{
    ExtractionFailed,
    InvalidArchive,
    IoError,
};

pub fn extractTarball(allocator: std.mem.Allocator, io: std.Io, tarball_path: []const u8, out_dir: []const u8) !void {
    var child_args = std.ArrayListUnmanaged([]const u8).empty;

    // Use fast native tar extraction using multiple threads where possible
    try child_args.append(allocator, "tar");
    try child_args.append(allocator, "-xf");
    try child_args.append(allocator, tarball_path);
    try child_args.append(allocator, "-C");
    try child_args.append(allocator, out_dir);

    var child = try std.process.spawn(io, .{
        .argv = child_args.items,
    });

    const term = try child.wait(io);
    switch (term) {
        .exited => |code| {
            if (code != 0) return error.ExtractionFailed;
        },
        else => return error.ExtractionFailed,
    }
}
