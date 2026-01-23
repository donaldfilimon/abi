//! Roadmap Integration
//!
//! Roadmap items and import functionality for project tracking.

const std = @import("std");
const types = @import("types.zig");

pub const RoadmapItem = struct {
    title: []const u8,
    category: []const u8,
    timeline: []const u8,
    description: ?[]const u8 = null,
};

pub const incomplete_items = [_]RoadmapItem{
    .{
        .title = "Record video tutorials",
        .category = "Documentation",
        .timeline = "Short-term",
        .description = "Record and produce video tutorials from existing scripts in docs/tutorials/videos/",
    },
    .{
        .title = "FPGA/ASIC hardware acceleration research",
        .category = "Research & Innovation",
        .timeline = "Long-term (2027+)",
        .description = "Experimental hardware acceleration using FPGA and ASIC for vector operations",
    },
    .{
        .title = "Novel index structures research",
        .category = "Research & Innovation",
        .timeline = "Long-term (2027+)",
        .description = "Research and implement novel index structures for improved search performance",
    },
    .{
        .title = "AI-optimized workloads",
        .category = "Research & Innovation",
        .timeline = "Long-term (2027+)",
        .description = "Optimize workloads specifically for AI/ML inference patterns",
    },
    .{
        .title = "Academic collaborations",
        .category = "Research & Innovation",
        .timeline = "Long-term (2027+)",
        .description = "Research partnerships, paper publications, conference presentations",
    },
    .{
        .title = "Community governance RFC process",
        .category = "Community & Growth",
        .timeline = "Long-term (2027+)",
        .description = "Establish RFC process, voting mechanism, contribution recognition",
    },
    .{
        .title = "Education and certification program",
        .category = "Community & Growth",
        .timeline = "Long-term (2027+)",
        .description = "Training courses, certification program, university partnerships",
    },
    .{
        .title = "Commercial support services",
        .category = "Enterprise Features",
        .timeline = "Long-term (2028+)",
        .description = "SLA offerings, priority support, custom development services",
    },
    .{
        .title = "AWS Lambda integration",
        .category = "Cloud Integration",
        .timeline = "Long-term (2028+)",
        .description = "Deploy ABI functions to AWS Lambda",
    },
    .{
        .title = "Google Cloud Functions integration",
        .category = "Cloud Integration",
        .timeline = "Long-term (2028+)",
        .description = "Deploy ABI functions to Google Cloud Functions",
    },
    .{
        .title = "Azure Functions integration",
        .category = "Cloud Integration",
        .timeline = "Long-term (2028+)",
        .description = "Deploy ABI functions to Azure Functions",
    },
};

/// Import all roadmap items that don't already exist as tasks.
/// Returns the number of items imported.
pub fn importAll(
    allocator: std.mem.Allocator,
    tasks: *std.AutoHashMapUnmanaged(u64, types.Task),
    next_id: *u64,
    dupeStringFn: *const fn (std.mem.Allocator, []const u8) types.ManagerError![]const u8,
    time_utils: anytype,
) !usize {
    var count: usize = 0;
    for (incomplete_items) |item| {
        var exists = false;
        var iter = tasks.iterator();
        while (iter.next()) |entry| {
            if (std.mem.eql(u8, entry.value_ptr.title, item.title)) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            const now = time_utils.unixSeconds();
            const id = next_id.*;
            next_id.* += 1;

            const owned_title = try dupeStringFn(allocator, item.title);
            const owned_desc = if (item.description) |d| try dupeStringFn(allocator, d) else null;

            const task = types.Task{
                .id = id,
                .title = owned_title,
                .description = owned_desc,
                .priority = if (std.mem.eql(u8, item.timeline, "Short-term")) .high else .low,
                .category = .roadmap,
                .created_at = now,
                .updated_at = now,
            };

            try tasks.put(allocator, id, task);
            count += 1;
        }
    }
    return count;
}

// ============================================================================
// Tests
// ============================================================================

test "Roadmap incomplete_items count" {
    try std.testing.expectEqual(@as(usize, 11), incomplete_items.len);
}
