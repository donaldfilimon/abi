const std = @import("std");

pub const LoggingSink = enum {
    stdout,
    stderr,
    file,
};

pub const ProfileConfig = struct {
    sink: LoggingSink = .stdout,
    path: []const u8 = "",
};

pub const ProfileEntry = struct {
    label: []const u8,
    duration_ns: u64,
    timestamp_ns: ?u64 = null,
};

pub const ProfileError =
    std.Io.File.OpenError ||
    std.Io.File.StatError ||
    std.Io.File.WritePositionalError ||
    error{
        MissingPath,
        InvalidSink,
    };

pub const ProfileWriter = struct {
    sink: LoggingSink,
    io_backend: std.Io.Threaded,
    io: std.Io,
    file: ?std.Io.File = null,
    file_offset: u64 = 0,

    pub fn init(config: ProfileConfig) ProfileError!ProfileWriter {
        var io_backend = std.Io.Threaded.init(std.heap.page_allocator, .{
            .environ = std.process.Environ.empty,
        });
        const io = io_backend.io();

        var writer = ProfileWriter{
            .sink = config.sink,
            .io_backend = io_backend,
            .io = io,
            .file = null,
            .file_offset = 0,
        };
        if (config.sink == .file) {
            if (config.path.len == 0) return error.MissingPath;
            var file = try std.Io.Dir.cwd().createFile(io, config.path, .{
                .truncate = false,
            });
            errdefer file.close(io);
            const stat = try file.stat(io);
            writer.file_offset = stat.size;
            writer.file = file;
        }
        return writer;
    }

    pub fn deinit(self: *ProfileWriter) void {
        if (self.file) |file| {
            file.close(self.io);
        }
        self.io_backend.deinit();
        self.* = undefined;
    }

    pub fn writeLine(self: *ProfileWriter, line: []const u8) ProfileError!void {
        switch (self.sink) {
            .stdout => {
                const file = std.Io.File.stdout();
                try file.writeStreamingAll(self.io, line);
                try file.writeStreamingAll(self.io, "\n");
            },
            .stderr => {
                const file = std.Io.File.stderr();
                try file.writeStreamingAll(self.io, line);
                try file.writeStreamingAll(self.io, "\n");
            },
            .file => {
                const file = self.file orelse {
                    std.log.err("Invalid file sink: no file provided", .{});
                    return error.InvalidSink;
                };
                try file.writePositionalAll(self.io, line, self.file_offset);
                self.file_offset += line.len;
                try file.writePositionalAll(self.io, "\n", self.file_offset);
                self.file_offset += 1;
            },
        }
    }

    pub fn writeEntry(self: *ProfileWriter, entry: ProfileEntry) ProfileError!void {
        var buffer: [256]u8 = undefined;
        const line = try formatEntry(&buffer, entry);
        try self.writeLine(line);
    }
};

pub fn writeProfileLine(sink: LoggingSink, line: []const u8) void {
    const config = ProfileConfig{ .sink = sink };
    var writer = ProfileWriter.init(config) catch |err| {
        std.log.warn("Failed to initialize profile writer: {t}", .{err});
        return;
    };
    defer writer.deinit();
    writer.writeLine(line) catch |err| {
        std.log.warn("Failed to write profile line: {t}", .{err});
    };
}

pub fn writeProfileLineWithConfig(config: ProfileConfig, line: []const u8) ProfileError!void {
    var writer = try ProfileWriter.init(config);
    defer writer.deinit();
    try writer.writeLine(line);
}

pub fn writeProfileEntryWithConfig(config: ProfileConfig, entry: ProfileEntry) ProfileError!void {
    var writer = try ProfileWriter.init(config);
    defer writer.deinit();
    try writer.writeEntry(entry);
}

pub fn formatEntry(buffer: []u8, entry: ProfileEntry) ![]u8 {
    if (entry.timestamp_ns) |timestamp| {
        return std.fmt.bufPrint(
            buffer,
            "{s} duration_ns={d} timestamp_ns={d}",
            .{ entry.label, entry.duration_ns, timestamp },
        );
    }
    return std.fmt.bufPrint(
        buffer,
        "{s} duration_ns={d}",
        .{ entry.label, entry.duration_ns },
    );
}

test "profile writer writes to file sink" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const file = try tmp.dir.createFile(io, "profile.log", .{ .truncate = true });
    file.close(io);

    const path_z = try tmp.dir.realPathFileAlloc(io, "profile.log", allocator);
    defer allocator.free(path_z);
    const path = path_z[0..path_z.len];

    var writer = try ProfileWriter.init(.{ .sink = .file, .path = path });
    defer writer.deinit();
    try writer.writeLine("hello");

    const contents = try tmp.dir.readFileAlloc(
        io,
        "profile.log",
        allocator,
        .limited(1024),
    );
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("hello\n", contents);
}

test "profile entry formatting" {
    var buffer: [128]u8 = undefined;
    const line = try formatEntry(&buffer, .{
        .label = "step",
        .duration_ns = 1200,
        .timestamp_ns = 42,
    });
    try std.testing.expectEqualStrings("step duration_ns=1200 timestamp_ns=42", line);
}
