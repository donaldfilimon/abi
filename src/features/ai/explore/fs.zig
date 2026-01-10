const std = @import("std");
const ExploreConfig = @import("config.zig").ExploreConfig;
const FileFilter = @import("config.zig").FileFilter;

pub const FileStats = struct {
    path: []const u8,
    size_bytes: u64,
    mtime: i128,
    ctime: i128,
    is_directory: bool,
    is_symlink: bool,
    mode: u16,

    pub fn fromDirEntry(allocator: std.mem.Allocator, io: std.Io, base_path: []const u8, entry: std.Io.Dir.Entry) !FileStats {
        const full_path = try std.fs.path.join(allocator, &.{ base_path, entry.name });
        errdefer allocator.free(full_path);

        var stats: FileStats = undefined;

        if (entry.kind == .directory) {
            stats = FileStats{
                .path = full_path,
                .size_bytes = 0,
                .mtime = 0,
                .ctime = 0,
                .is_directory = true,
                .is_symlink = false,
                .mode = 0,
            };
        } else if (entry.kind == .file) {
            const file = try std.Io.Dir.cwd().openFile(io, full_path, .{});
            defer file.close(io);

            const stat = try file.stat(io);
            stats = FileStats{
                .path = full_path,
                .size_bytes = stat.size,
                .mtime = 0,
                .ctime = 0,
                .is_directory = false,
                .is_symlink = false,
                .mode = 0,
            };
        } else {
            stats = FileStats{
                .path = full_path,
                .size_bytes = 0,
                .mtime = 0,
                .ctime = 0,
                .is_directory = false,
                .is_symlink = false,
                .mode = 0,
            };
        }

        return stats;
    }
};

pub const FileVisitor = struct {
    allocator: std.mem.Allocator,
    config: *const ExploreConfig,
    files: std.ArrayListUnmanaged(FileStats),
    directories: std.ArrayListUnmanaged([]const u8),
    visited_paths: std.StringHashMapUnmanaged(bool),
    symlink_count: usize = 0,
    max_symlinks: usize = 32,
    error_count: usize = 0,
    io_backend: std.Io.Threaded,

    pub fn init(allocator: std.mem.Allocator, config: *const ExploreConfig) !FileVisitor {
        return FileVisitor{
            .allocator = allocator,
            .config = config,
            .files = std.ArrayListUnmanaged(FileStats){},
            .directories = std.ArrayListUnmanaged([]const u8){},
            .visited_paths = std.StringHashMapUnmanaged(bool){},
            .io_backend = std.Io.Threaded.init(allocator, .{
                .environ = std.process.Environ.empty,
            }),
        };
    }

    pub fn deinit(self: *FileVisitor) void {
        for (self.files.items) |*stat| {
            self.allocator.free(stat.path);
        }
        self.files.deinit(self.allocator);

        for (self.directories.items) |dir| {
            self.allocator.free(dir);
        }
        self.directories.deinit(self.allocator);

        self.visited_paths.deinit(self.allocator);
        self.io_backend.deinit();
    }

    pub fn visit(self: *FileVisitor, root_path: []const u8) !void {
        try self.visited_paths.put(self.allocator, root_path, true);
        try self.directories.append(self.allocator, try self.allocator.dupe(u8, root_path));

        while (self.directories.pop()) |dir| {
            try self.walkDirectory(dir);
        }
    }

    fn walkDirectory(self: *FileVisitor, dir_path: []const u8) !void {
        const io = self.io_backend.io();
        var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{}) catch {
            self.error_count += 1;
            return;
        };
        defer dir.close(io);

        var iterator = dir.iterate();
        while (iterator.next(io) catch {
            self.error_count += 1;
            return;
        }) |entry| {
            const full_path = std.fs.path.join(self.allocator, &.{ dir_path, entry.name }) catch {
                self.error_count += 1;
                continue;
            };
            defer self.allocator.free(full_path);

            if (self.shouldSkip(entry.name)) continue;

            if (entry.kind == .directory) {
                if (self.visited_paths.contains(full_path)) continue;
                try self.visited_paths.put(self.allocator, full_path, true);
                try self.directories.append(self.allocator, full_path);
            } else if (entry.kind == .file) {
                const stats = FileStats.fromDirEntry(self.allocator, io, dir_path, entry) catch {
                    self.error_count += 1;
                    continue;
                };
                try self.files.append(self.allocator, stats);
            }
        }
    }

    fn shouldSkip(self: *FileVisitor, name: []const u8) bool {
        if (!self.config.include_hidden) {
            if (name.len > 0 and name[0] == '.') return true;
        }

        for (self.config.exclude_patterns) |pattern| {
            if (matchesGlob(pattern, name)) return true;
        }

        return false;
    }

    pub fn getFiles(self: *FileVisitor) []FileStats {
        return self.files.items;
    }

    pub fn getFileCount(self: *FileVisitor) usize {
        return self.files.items.len;
    }
};

fn matchesGlob(pattern: []const u8, name: []const u8) bool {
    if (pattern.len == 0) return false;

    if (std.mem.endsWith(u8, pattern, "*")) {
        const prefix = pattern[0 .. pattern.len - 1];
        return std.mem.startsWith(u8, name, prefix);
    }

    if (std.mem.startsWith(u8, pattern, "*")) {
        const suffix = pattern[1..];
        return std.mem.endsWith(u8, name, suffix);
    }

    return std.mem.eql(u8, pattern, name);
}

pub fn getFileExtension(filename: []const u8) []const u8 {
    return std.fs.path.extension(filename);
}

pub fn getFileBasename(filename: []const u8) []const u8 {
    const basename = std.fs.path.basename(filename);
    const ext = getFileExtension(basename);
    if (ext.len > 0) {
        return basename[0 .. basename.len - ext.len];
    }
    return basename;
}

pub fn determineFileType(filename: []const u8) []const u8 {
    if (std.mem.endsWith(u8, filename, ".test.zig")) return "test";
    if (std.mem.endsWith(u8, filename, ".test")) return "test";
    if (std.mem.endsWith(u8, filename, "._test")) return "test";
    if (std.mem.endsWith(u8, filename, ".spec")) return "test";

    const ext = getFileExtension(filename);

    const source_exts = [_][]const u8{ ".zig", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".py", ".js", ".ts" };
    for (source_exts) |e| {
        if (std.mem.eql(u8, ext, e)) return "source";
    }

    const doc_exts = [_][]const u8{ ".md", ".txt", ".rst", ".adoc" };
    for (doc_exts) |e| {
        if (std.mem.eql(u8, ext, e)) return "documentation";
    }

    const config_exts = [_][]const u8{ ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf" };
    for (config_exts) |e| {
        if (std.mem.eql(u8, ext, e)) return "config";
    }

    const data_exts = [_][]const u8{ ".csv", ".xml", ".sql" };
    for (data_exts) |e| {
        if (std.mem.eql(u8, ext, e)) return "data";
    }

    const binary_exts = [_][]const u8{ ".exe", ".bin", ".so", ".dll", ".a", ".o" };
    for (binary_exts) |e| {
        if (std.mem.eql(u8, ext, e)) return "binary";
    }

    return "other";
}

pub fn readFileContent(allocator: std.mem.Allocator, io: std.Io, path: []const u8, max_size: ?usize) ![]const u8 {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch {
        return error.FileNotFound;
    };
    defer file.close(io);

    const stat = try file.stat(io);
    const max_bytes = max_size orelse stat.size;
    const size = @min(stat.size, max_bytes);

    const content = try file.readToEndAlloc(allocator, io, size);
    return content;
}

pub fn readFileLines(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !std.ArrayListUnmanaged([]const u8) {
    const content = try readFileContent(allocator, io, path, null);
    defer allocator.free(content);

    var lines = std.ArrayListUnmanaged([]const u8){};
    var start: usize = 0;

    for (content, 0..) |c, i| {
        if (c == '\n') {
            const line = try allocator.dupe(u8, content[start..i]);
            try lines.append(allocator, line);
            start = i + 1;
        }
    }

    if (start < content.len) {
        const line = try allocator.dupe(u8, content[start..]);
        try lines.append(allocator, line);
    }

    return lines;
}

pub fn countLines(content: []const u8) usize {
    var count: usize = 1;
    for (content) |c| {
        if (c == '\n') count += 1;
    }
    return count;
}
