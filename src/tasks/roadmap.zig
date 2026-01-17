//! Roadmap Item Importer
//!
//! Imports incomplete roadmap items from ROADMAP.md as tasks.

const std = @import("std");
const Manager = @import("manager.zig").Manager;
const types = @import("types.zig");

pub const RoadmapItem = struct {
    title: []const u8,
    category: []const u8,
    timeline: []const u8,
    description: ?[]const u8 = null,
};

/// Predefined roadmap items from ROADMAP.md analysis
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

/// Import all incomplete roadmap items as tasks
pub fn importAll(manager: *Manager) !usize {
    var count: usize = 0;

    for (incomplete_items) |item| {
        // Check if already exists (by title)
        var exists = false;
        var iter = manager.tasks.iterator();
        while (iter.next()) |entry| {
            if (std.mem.eql(u8, entry.value_ptr.title, item.title)) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            _ = try manager.add(item.title, .{
                .description = item.description,
                .category = .roadmap,
                .priority = if (std.mem.eql(u8, item.timeline, "Short-term")) .high else .low,
            });
            count += 1;
        }
    }

    return count;
}

test "RoadmapItem struct" {
    const item = RoadmapItem{
        .title = "Test item",
        .category = "Testing",
        .timeline = "Short-term",
        .description = "Test description",
    };
    try std.testing.expectEqualStrings("Test item", item.title);
    try std.testing.expectEqualStrings("Testing", item.category);
    try std.testing.expectEqualStrings("Short-term", item.timeline);
    try std.testing.expectEqualStrings("Test description", item.description.?);
}

test "incomplete_items count" {
    try std.testing.expectEqual(@as(usize, 11), incomplete_items.len);
}

test "importAll adds roadmap items" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_roadmap_tasks.json",
        .auto_save = false,
    });
    defer manager.deinit();

    const count = try importAll(&manager);
    try std.testing.expectEqual(@as(usize, 11), count);

    // Verify tasks were added with correct category
    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 11), stats.total);

    // Check a specific task exists
    var found = false;
    var iter = manager.tasks.iterator();
    while (iter.next()) |entry| {
        if (std.mem.eql(u8, entry.value_ptr.title, "Record video tutorials")) {
            try std.testing.expectEqual(types.Category.roadmap, entry.value_ptr.category);
            try std.testing.expectEqual(types.Priority.high, entry.value_ptr.priority);
            found = true;
            break;
        }
    }
    try std.testing.expect(found);
}

test "importAll skips existing items" {
    var manager = try Manager.init(std.testing.allocator, .{
        .storage_path = ".zig-cache/test_roadmap_tasks2.json",
        .auto_save = false,
    });
    defer manager.deinit();

    // First import
    const count1 = try importAll(&manager);
    try std.testing.expectEqual(@as(usize, 11), count1);

    // Second import should add nothing
    const count2 = try importAll(&manager);
    try std.testing.expectEqual(@as(usize, 0), count2);

    // Total should still be 11
    const stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 11), stats.total);
}
