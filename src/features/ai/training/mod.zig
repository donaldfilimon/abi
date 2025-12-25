const std = @import("std");

pub const TrainingError = error{
    InvalidConfiguration,
};

pub const TrainingConfig = struct {
    epochs: u32 = 1,
    batch_size: u32 = 16,
    sample_count: u32 = 256,
    learning_rate: f32 = 0.001,

    pub fn validate(self: TrainingConfig) TrainingError!void {
        if (self.epochs == 0) return TrainingError.InvalidConfiguration;
        if (self.batch_size == 0) return TrainingError.InvalidConfiguration;
        if (self.sample_count == 0) return TrainingError.InvalidConfiguration;
        if (self.learning_rate <= 0) return TrainingError.InvalidConfiguration;
    }
};

pub const TrainingReport = struct {
    epochs: u32,
    batches: u32,
    final_loss: f32,
    final_accuracy: f32,
};

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) !void {
    _ = try trainAndReport(allocator, config);
}

pub fn trainAndReport(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingReport {
    try config.validate();

    const batches_per_epoch = (config.sample_count + config.batch_size - 1) / config.batch_size;
    const scratch_len = @min(@as(usize, config.batch_size), 1024);
    const scratch = try allocator.alloc(f32, scratch_len);
    defer allocator.free(scratch);
    std.mem.set(f32, scratch, 0);

    var loss: f32 = 1.0;
    var accuracy: f32 = 0.5;

    var epoch: u32 = 0;
    while (epoch < config.epochs) : (epoch += 1) {
        var batch: u32 = 0;
        while (batch < batches_per_epoch) : (batch += 1) {
            const step = config.learning_rate * 0.1;
            loss = if (loss > step) loss - step else 0.0;
            accuracy = @min(1.0, accuracy + config.learning_rate * 0.05);
        }
    }

    return .{
        .epochs = config.epochs,
        .batches = batches_per_epoch,
        .final_loss = loss,
        .final_accuracy = accuracy,
    };
}
