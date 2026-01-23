//! Task Persistence
//!
//! JSON serialization and file I/O for task data.

const std = @import("std");
const types = @import("types.zig");

const Task = types.Task;
const Status = types.Status;
const Priority = types.Priority;
const Category = types.Category;
const ManagerError = types.ManagerError;

/// Write a JSON-escaped string to the buffer.
pub fn writeJsonString(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\n"),
            '\r' => try buf.appendSlice(allocator, "\r"),
            '\t' => try buf.appendSlice(allocator, "\t"),
            else => try buf.append(allocator, c),
        }
    }
}

/// Write a single task as JSON to the buffer.
pub fn writeTask(allocator: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), task: Task) !void {
    try buf.appendSlice(allocator, "    {");
    try buf.print(allocator, "\"id\":{d},", .{task.id});
    try buf.appendSlice(allocator, "\"title\":\"");
    try writeJsonString(allocator, buf, task.title);
    try buf.appendSlice(allocator, "\",");
    try buf.print(allocator, "\"status\":\"{s}\",", .{task.status.toString()});
    try buf.print(allocator, "\"priority\":\"{s}\",", .{task.priority.toString()});
    try buf.print(allocator, "\"category\":\"{s}\",", .{task.category.toString()});
    try buf.print(allocator, "\"created_at\":{d},", .{task.created_at});
    try buf.print(allocator, "\"updated_at\":{d}", .{task.updated_at});
    if (task.description) |d| {
        try buf.appendSlice(allocator, ",\"description\":\"");
        try writeJsonString(allocator, buf, d);
        try buf.append(allocator, '"');
    }
    if (task.due_date) |d| {
        try buf.print(allocator, ",\"due_date\":{d}", .{d});
    }
    if (task.completed_at) |c| {
        try buf.print(allocator, ",\"completed_at\":{d}", .{c});
    }
    if (task.parent_id) |p| {
        try buf.print(allocator, ",\"parent_id\":{d}", .{p});
    }
    try buf.append(allocator, '}');
}

/// Parse a task from JSON object.
pub fn parseTask(
    allocator: std.mem.Allocator,
    obj: std.json.ObjectMap,
    dupeStringFn: *const fn (std.mem.Allocator, []const u8) ManagerError![]const u8,
) ManagerError!Task {
    const id: u64 = @intCast(obj.get("id").?.integer);
    const title = try dupeStringFn(allocator, obj.get("title").?.string);
    const status = Status.fromString(obj.get("status").?.string) orelse .pending;
    const priority = Priority.fromString(obj.get("priority").?.string) orelse .normal;
    const category = Category.fromString(obj.get("category").?.string) orelse .personal;
    const created_at: i64 = obj.get("created_at").?.integer;
    const updated_at: i64 = obj.get("updated_at").?.integer;

    var task = Task{
        .id = id,
        .title = title,
        .status = status,
        .priority = priority,
        .category = category,
        .created_at = created_at,
        .updated_at = updated_at,
    };

    if (obj.get("description")) |d| {
        task.description = try dupeStringFn(allocator, d.string);
    }
    if (obj.get("due_date")) |d| {
        task.due_date = d.integer;
    }
    if (obj.get("completed_at")) |c| {
        task.completed_at = c.integer;
    }
    if (obj.get("parent_id")) |p| {
        task.parent_id = @intCast(p.integer);
    }

    return task;
}

/// Save tasks to a JSON file.
pub fn save(
    allocator: std.mem.Allocator,
    storage_path: []const u8,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    next_id: u64,
) ManagerError!void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const dir_path = std.fs.path.dirname(storage_path) orelse ".";
    std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};

    var file = std.Io.Dir.cwd().createFile(io, storage_path, .{ .truncate = true }) catch |err| {
        return err;
    };
    defer file.close(io);

    var json_buffer = std.ArrayListUnmanaged(u8).empty;
    defer json_buffer.deinit(allocator);

    try json_buffer.appendSlice(allocator, "{\n  \"next_id\": ");
    try json_buffer.print(allocator, "{d}", .{next_id});
    try json_buffer.appendSlice(allocator, ",\n  \"tasks\": [\n");

    var first = true;
    var iter = tasks.iterator();
    while (iter.next()) |entry| {
        if (!first) try json_buffer.appendSlice(allocator, ",\n");
        first = false;
        try writeTask(allocator, &json_buffer, entry.value_ptr.*);
    }

    try json_buffer.appendSlice(allocator, "\n  ]\n}\n");

    try file.writeStreamingAll(io, json_buffer.items);
}

/// Load tasks from a JSON file.
/// Returns the next_id value from the file.
pub fn load(
    allocator: std.mem.Allocator,
    storage_path: []const u8,
    tasks: *std.AutoHashMapUnmanaged(u64, Task),
    dupeStringFn: *const fn (std.mem.Allocator, []const u8) ManagerError![]const u8,
) ManagerError!u64 {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const content = std.Io.Dir.cwd().readFileAlloc(
        io,
        storage_path,
        allocator,
        .limited(1024 * 1024),
    ) catch |err| {
        return err;
    };
    defer allocator.free(content);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch {
        return error.ParseError;
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    var next_id: u64 = 1;
    if (root.get("next_id")) |nid| {
        next_id = @intCast(nid.integer);
    }

    if (root.get("tasks")) |tasks_val| {
        for (tasks_val.array.items) |task_val| {
            const obj = task_val.object;
            const task = try parseTask(allocator, obj, dupeStringFn);
            try tasks.put(allocator, task.id, task);
        }
    }

    return next_id;
}
