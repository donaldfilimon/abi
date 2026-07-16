pub const TaskStatus = enum(u8) {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

pub const TaskPriority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    critical = 3,
};

pub const TaskFn = *const fn (ctx: ?*anyopaque) anyerror!void;

pub const Task = struct {
    id: u64,
    name: []const u8,
    priority: TaskPriority,
    status: TaskStatus,
    fn_ptr: TaskFn,
    ctx: ?*anyopaque,
    created_at: i64,
    started_at: i64,
    completed_at: i64,
    error_msg: ?[]const u8,
};
