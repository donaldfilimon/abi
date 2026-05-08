const std = @import("std");
const self_learning = @import("self_learning.zig");

const ExperienceBuffer = self_learning.ExperienceBuffer;
const LearningExperience = self_learning.LearningExperience;
const RewardModel = self_learning.RewardModel;
const SelfLearningSystem = self_learning.SelfLearningSystem;

// ============================================================================
// Tests
// ============================================================================

test "ExperienceBuffer basic operations" {
    const allocator = std.testing.allocator;
    var buffer = ExperienceBuffer.init(allocator, 100);
    defer buffer.deinit();

    const input = try allocator.dupe(u32, &[_]u32{ 1, 2, 3 });
    const output = try allocator.dupe(u32, &[_]u32{ 4, 5, 6 });

    const exp = LearningExperience{
        .id = 0,
        .exp_type = .text_conversation,
        .input = input,
        .output = output,
        .reward = 0.5,
        .confidence = 0.8,
        .feedback = .positive,
        .timestamp = 0,
        .log_probs = null,
        .value = 0,
        .advantage = 0,
        .done = true,
        .image_data = null,
        .video_data = null,
        .audio_data = null,
        .document_content = null,
        .raw_data = null,
        .content_type = null,
        .metadata = .{},
    };

    try buffer.add(exp);
    try std.testing.expectEqual(@as(usize, 1), buffer.len());
}

test "RewardModel computation" {
    const allocator = std.testing.allocator;
    var model = try RewardModel.init(allocator, 64);
    defer model.deinit();

    var embedding: [64]f32 = undefined;
    for (&embedding) |*e| {
        e.* = 0.1;
    }

    const reward = model.computeReward(&embedding);
    try std.testing.expect(reward >= -1.0 and reward <= 1.0);
}

test "SelfLearningSystem initialization" {
    const allocator = std.testing.allocator;
    var system = try SelfLearningSystem.init(allocator, .{});
    defer system.deinit();

    try std.testing.expectEqual(@as(u64, 0), system.stats.total_experiences);
}

test {
    std.testing.refAllDecls(@This());
}
