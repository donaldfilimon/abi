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

pub const ProfileError = std.fs.File.OpenError || std.fs.File.WriteError || error{
    MissingPath,
    InvalidSink,
};

pub const ProfileWriter = struct {
    sink: LoggingSink,
    file: ?std.fs.File = null,

    pub fn init(config: ProfileConfig) ProfileError!ProfileWriter {
        var writer = ProfileWriter{
            .sink = config.sink,
            .file = null,
        };
        if (config.sink == .file) {
            if (config.path.len == 0) return error.MissingPath;
            var file = try std.fs.cwd().createFile(config.path, .{
                .truncate = false,
            });
            errdefer file.close();
            try file.seekFromEnd(0);
            writer.file = file;
        }
        return writer;
    }

    pub fn deinit(self: *ProfileWriter) void {
        if (self.file) |file| {
            file.close();
        }
        self.* = undefined;
    }

    pub fn writeLine(self: *ProfileWriter, line: []const u8) ProfileError!void {
        switch (self.sink) {
            .stdout => {
                const file = std.fs.File.stdout();
                try file.writeAll(line);
                try file.writeAll("\n");
            },
            .stderr => {
                const file = std.fs.File.stderr();
                try file.writeAll(line);
                try file.writeAll("\n");
            },
            .file => {
                const file = self.file orelse return error.InvalidSink;
                try file.writeAll(line);
                try file.writeAll("\n");
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
    var writer = ProfileWriter.init(config) catch return;
    defer writer.deinit();
    _ = writer.writeLine(line) catch {};
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
    const file = try tmp.dir.createFile("profile.log", .{ .truncate = true });
    file.close();

    const path = try tmp.dir.realpathAlloc(allocator, "profile.log");
    defer allocator.free(path);

    var writer = try ProfileWriter.init(.{ .sink = .file, .path = path });
    defer writer.deinit();
    try writer.writeLine("hello");

    const contents = try tmp.dir.readFileAlloc(allocator, "profile.log", .limited(1024));
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
