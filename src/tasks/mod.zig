//! Task Management Module
//!
//! Provides unified task tracking for personal tasks, project roadmap
//! items, and distributed compute jobs.
//!
//! ## Usage
//!
//! ```zig
//! const tasks = @import("tasks/mod.zig");
//!
//! var manager = try tasks.Manager.init(allocator, .{});
//! defer manager.deinit();
//!
//! const id = try manager.add("Fix bug", .{ .priority = .high });
//! try manager.complete(id);
//! ```

pub const types = @import("types.zig");

pub const Task = types.Task;
pub const Priority = types.Priority;
pub const Status = types.Status;
pub const Category = types.Category;
pub const Filter = types.Filter;
pub const SortBy = types.SortBy;
pub const Stats = types.Stats;

// Manager will be added in Task 1.2
pub const Manager = @import("manager.zig").Manager;
pub const ManagerError = @import("manager.zig").ManagerError;
pub const ManagerConfig = @import("manager.zig").ManagerConfig;
pub const AddOptions = @import("manager.zig").AddOptions;

// Roadmap integration
pub const roadmap = @import("roadmap.zig");

test {
    _ = types;
    _ = @import("manager.zig");
    _ = @import("roadmap.zig");
}
