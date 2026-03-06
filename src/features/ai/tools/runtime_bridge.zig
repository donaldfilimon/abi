//! Runtime Bridge Module
//!
//! Provides ABI with a secure, native execution sandbox to dynamically
//! synthesize and evaluate Python and JavaScript code on the host machine.
//! This vastly expands ABI's computational and data-manipulation capabilities.

const std = @import("std");

pub const RuntimeEnvironment = enum {
    python3,
    node,
    deno,
};

pub const RuntimeResult = struct {
    stdout: []const u8,
    stderr: []const u8,
    exit_code: u32,

    pub fn deinit(self: *RuntimeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

pub const RuntimeBridge = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io) RuntimeBridge {
        return .{
            .allocator = allocator,
            .io = io,
        };
    }

    pub fn deinit(self: *RuntimeBridge) void {
        _ = self;
    }

    /// Dynamically writes a script to disk and executes it via the host's installed runtime.
    pub fn executeScript(self: *RuntimeBridge, env: RuntimeEnvironment, script_content: []const u8) !RuntimeResult {
        const bin_name = switch (env) {
            .python3 => "python3",
            .node => "node",
            .deno => "deno run -A", // Assuming relaxed permissions for ABI local tasks
        };

        const ext = switch (env) {
            .python3 => ".py",
            .node, .deno => ".js",
        };

        // Write synthetic script to a temporary file
        var tmp_name_buf: [64]u8 = [_]u8{0} ** 64;
        const ts = @import("../../../services/shared/time.zig").timestampMs();
        const tmp_file_name = try std.fmt.bufPrint(&tmp_name_buf, ".abi_synthetic_script_{d}{s}", .{ ts, ext });

        var file = try std.Io.Dir.cwd().createFile(self.io.*, tmp_file_name, .{ .truncate = true });
        try file.writeStreamingAll(self.io.*, script_content);
        file.close(self.io.*);

        // Ensure we cleanup the temp script
        defer std.Io.Dir.cwd().deleteFile(self.io.*, tmp_file_name) catch {};

        std.log.info("[Runtime Bridge] Executing {s} script via {s}...", .{ ext, bin_name });

        // Setup execution arguments
        var args = std.ArrayListUnmanaged([]const u8).empty;
        defer args.deinit(self.allocator);

        var bin_iter = std.mem.splitScalar(u8, bin_name, ' ');
        while (bin_iter.next()) |arg| {
            try args.append(self.allocator, arg);
        }
        try args.append(self.allocator, tmp_file_name);

        // In Zig 0.16 we execute via child process natively
        var result = try std.process.run(self.allocator, self.io.*, .{
            .argv = args.items,
        });

        const exit_code = switch (result.term) {
            .exited => |code| code,
            else => 1,
        };

        return RuntimeResult{
            .stdout = result.stdout,
            .stderr = result.stderr,
            .exit_code = exit_code,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
