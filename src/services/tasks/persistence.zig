//! Task Persistence
//!
//! ZON serialization and file I/O for task data.

const std = @import("std");
const types = @import("types.zig");

const Task = types.Task;
const Status = types.Status;
const Priority = types.Priority;
const Category = types.Category;
const ManagerError = types.ManagerError;

/// Helper structs for ZON persistence
const PersistedTask = struct {
    id: u64,
    title: []const u8,
    status: []const u8,
    priority: []const u8,
    category: []const u8,
    created_at: i64,
    updated_at: i64,
    description: ?[]const u8 = null,
    due_date: ?i64 = null,
    completed_at: ?i64 = null,
    parent_id: ?u64 = null,
};

const PersistedData = struct {
    next_id: u64,
    tasks: []const PersistedTask,
};

/// Duplicate a string and store it in the strings list.
pub fn dupeString(allocator: std.mem.Allocator, strings: *std.ArrayListUnmanaged([]u8), s: []const u8) ManagerError![]const u8 {
    const duped = try allocator.dupe(u8, s);
    try strings.append(allocator, duped);
    return duped;
}

/// Save tasks to a ZON file.
pub fn save(
    allocator: std.mem.Allocator,
    storage_path: []const u8,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    next_id: u64,
) ManagerError!void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const dir_path = std.fs.path.dirname(storage_path) orelse ".";
    std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| {
        std.log.warn("persistence: failed to create directory '{s}': {t}", .{ dir_path, err });
    };

    var file = try std.Io.Dir.cwd().createFile(io, storage_path, .{ .truncate = true });
    defer file.close(io);

    var persisted_tasks = std.ArrayList(PersistedTask).init(allocator);
    defer persisted_tasks.deinit();

    var iter = tasks.iterator();
    while (iter.next()) |entry| {
        const t = entry.value_ptr.*;
        try persisted_tasks.append(.{
            .id = t.id,
            .title = t.title,
            .status = t.status.toString(),
            .priority = t.priority.toString(),
            .category = t.category.toString(),
            .created_at = t.created_at,
            .updated_at = t.updated_at,
            .description = t.description,
            .due_date = t.due_date,
            .completed_at = t.completed_at,
            .parent_id = t.parent_id,
        });
    }

    const data = PersistedData{
        .next_id = next_id,
        .tasks = persisted_tasks.items,
    };

    var zon_buffer = std.ArrayList(u8).init(allocator);
    defer zon_buffer.deinit();
    var writer = zon_buffer.writer();
    try std.zon.stringify.serialize(data, .{}, &writer);

    try file.writeStreamingAll(io, zon_buffer.items);
}

/// Load tasks from a ZON file.
pub fn load(
    allocator: std.mem.Allocator,
    storage_path: []const u8,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    next_id: *u64,
    strings: *std.ArrayListUnmanaged([]u8),
) ManagerError!void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        storage_path,
        allocator,
        .limited(1024 * 1024),
    ) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer allocator.free(contents);

    const contents_z = try allocator.dupeZ(u8, contents);
    defer allocator.free(contents_z);

    const parsed = std.zon.parse.fromSlice(PersistedData, allocator, contents_z, null, .{}) catch {
        return error.ParseError;
    };
    defer parsed.deinit();

    const data = parsed.value;
    next_id.* = data.next_id;

    for (data.tasks) |pt| {
        const task = Task{
            .id = pt.id,
            .title = try dupeString(allocator, strings, pt.title),
            .status = Status.fromString(pt.status) orelse .pending,
            .priority = Priority.fromString(pt.priority) orelse .normal,
            .category = Category.fromString(pt.category) orelse .personal,
            .created_at = pt.created_at,
            .updated_at = pt.updated_at,
            .description = if (pt.description) |d| try dupeString(allocator, strings, d) else null,
            .due_date = pt.due_date,
            .completed_at = pt.completed_at,
            .parent_id = pt.parent_id,
        };
        try tasks.put(allocator, task.id, task);
    }
}

test {
    std.testing.refAllDecls(@This());
}
