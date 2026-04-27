//! Integration Tests: Tasks Feature
//!
//! Verifies the tasks module exports and type availability through the
//! public `abi.tasks` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const tasks = abi.tasks;

test "tasks: public types have expected variants" {
    try std.testing.expect(tasks.Priority.high == .high);
    try std.testing.expect(tasks.Status.pending == .pending);
    try std.testing.expect(tasks.Category.bug == .bug);
}

test "tasks: error set is accessible" {
    const E = tasks.ManagerError;
    try std.testing.expect(@as(E, error.TaskNotFound) == error.TaskNotFound);
}

test "tasks: sub-namespaces are accessible" {
    comptime {
        _ = tasks.types;
        _ = tasks.roadmap;
    }
}
