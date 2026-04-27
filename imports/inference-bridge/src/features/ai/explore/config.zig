const std = @import("std");

pub const ExploreLevel = enum(u2) {
    quick = 0,
    medium = 1,
    thorough = 2,
    deep = 3,
};

pub const ExploreConfig = struct {
    level: ExploreLevel = .medium,
    max_files: usize = 10000,
    max_depth: usize = 20,
    timeout_ms: u64 = 60000,
    include_patterns: []const []const u8 = &.{},
    exclude_patterns: []const []const u8 = &.{
        "*.git",
        "node_modules",
        "build",
        "target",
        ".zig-cache",
        "*.o",
        "*.a",
    },
    case_sensitive: bool = false,
    use_regex: bool = false,
    parallel_io: bool = true,
    worker_count: ?usize = null,
    follow_symlinks: bool = false,
    include_hidden: bool = false,
    file_size_limit_bytes: ?u64 = null,
    output_format: OutputFormat = .human,

    pub fn defaultForLevel(level: ExploreLevel) ExploreConfig {
        return switch (level) {
            .quick => ExploreConfig{
                .level = .quick,
                .max_files = 1000,
                .max_depth = 3,
                .timeout_ms = 10000,
                .parallel_io = false,
            },
            .medium => ExploreConfig{
                .level = .medium,
                .max_files = 5000,
                .max_depth = 10,
                .timeout_ms = 30000,
                .parallel_io = true,
            },
            .thorough => ExploreConfig{
                .level = .thorough,
                .max_files = 10000,
                .max_depth = 20,
                .timeout_ms = 60000,
                .parallel_io = true,
                .worker_count = 4,
            },
            .deep => ExploreConfig{
                .level = .deep,
                .max_files = 50000,
                .max_depth = 50,
                .timeout_ms = 300000,
                .parallel_io = true,
                .worker_count = null,
            },
        };
    }
};

pub const OutputFormat = enum {
    human,
    json,
    compact,
    yaml,
};

pub const FileType = enum {
    source,
    header,
    test_file,
    documentation,
    config,
    data,
    binary,
    other,
};

pub const FileFilter = struct {
    extensions: ?[]const []const u8 = null,
    exclude_extensions: ?[]const []const u8 = null,
    min_size_bytes: ?u64 = null,
    max_size_bytes: ?u64 = null,
    modified_after: ?i128 = null,
    modified_before: ?i128 = null,

    pub fn matches(self: *const FileFilter, stats: anytype) bool {
        if (self.extensions) |exts| {
            const path = if (@hasField(@TypeOf(stats), "path")) stats.path else return false;
            const ext = std.Io.Dir.path.extension(path);
            var found = false;
            for (exts) |e| {
                if (std.mem.eql(u8, ext, e)) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }

        if (self.exclude_extensions) |exts| {
            const path = if (@hasField(@TypeOf(stats), "path")) stats.path else return false;
            const ext = std.Io.Dir.path.extension(path);
            for (exts) |e| {
                if (std.mem.eql(u8, ext, e)) {
                    return false;
                }
            }
        }

        if (self.min_size_bytes) |min| {
            if (stats.size < min) return false;
        }

        if (self.max_size_bytes) |max| {
            if (stats.size > max) return false;
        }

        if (self.modified_after) |after| {
            if (stats.mtime < after) return false;
        }

        if (self.modified_before) |before| {
            if (stats.mtime > before) return false;
        }

        return true;
    }
};

pub const SearchScope = struct {
    paths: []const []const u8 = &.{"."},
    recursive: bool = true,
    filter: ?FileFilter = null,
};

pub const SearchOptions = struct {
    scope: SearchScope,
    patterns: []const []const u8,
    config: ExploreConfig,
};

test {
    std.testing.refAllDecls(@This());
}
