const std = @import("std");
const abi = @import("abi");

test "demo training of testingllm" {
    const allocator = std.testing.allocator;
    // Simple config for a tiny model
    const config = abi.ai.TrainingConfig{
        .epochs = 2,
        .batch_size = 2,
        .sample_count = 8,
        .model_size = 16,
        .learning_rate = 0.01,
        .optimizer = .sgd,
        .checkpoint_interval = 1,
        .max_checkpoints = 2,
        .checkpoint_path = "./testingllm.ckpt",
    };
    var result = try abi.ai.trainWithResult(allocator, config);
    defer result.deinit();
    // Ensure checkpoints were saved
    try std.testing.expect(result.checkpoints.count() > 0);
    // Print a short summary (visible in test output)
    std.debug.print("Training completed: epochs={d}, loss={d}\n", .{ result.report.epochs, result.report.final_loss });
}
