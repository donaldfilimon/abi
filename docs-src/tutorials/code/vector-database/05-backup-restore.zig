//! Vector Database Tutorial - Example 5: Backup & Restore
//!
//! Run with: zig run docs/tutorials/code/vector-database/05-backup-restore.zig

const std = @import("std");
// In a real project, you would use: const abi = @import("abi");
// For tutorial purposes, we use a relative path.
const abi = @import("../../../../src/abi.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.initDefault(allocator);
    defer framework.deinit();

    var db = try abi.database.openOrCreate(allocator, "production_db");
    defer abi.database.close(&db);

    const embedding = [_]f32{ 1.0, 2.0, 3.0 };
    try abi.database.insert(&db, 1, &embedding, "Critical business data");
    std.debug.print("Inserted data\n", .{});

    const backup_name = "backup_20260117.db";
    try abi.database.backup(&db, backup_name);
    std.debug.print("Backup created: backups/{s}\n", .{backup_name});

    // Restore from backup (example - don't run in same session)
    // try abi.database.restore(&db, backup_name);
    // std.debug.print("Restored from backup\n", .{});
}
