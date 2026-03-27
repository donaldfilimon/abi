const std = @import("std");

pub const DownloadError = error{
    DownloadFailed,
    NetworkError,
    NoUrlProvided,
    FileSystemError,
};

pub fn downloadFile(allocator: std.mem.Allocator, io: std.Io, url: []const u8, out_path: []const u8) !void {
    var child_args = std.ArrayListUnmanaged([]const u8).empty;

    // We use unmanaged and pass allocator to append
    try child_args.append(allocator, "curl");
    try child_args.append(allocator, "-fL");
    try child_args.append(allocator, "--progress-bar");
    try child_args.append(allocator, "-o");
    try child_args.append(allocator, out_path);
    try child_args.append(allocator, url);

    var child = try std.process.spawn(io, .{
        .argv = child_args.items,
    });

    const term = try child.wait(io);
    switch (term) {
        .exited => |code| {
            if (code != 0) return error.DownloadFailed;
        },
        else => return error.DownloadFailed,
    }
}
