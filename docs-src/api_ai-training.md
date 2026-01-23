# ai-training API Reference

> Training pipelines and fine-tuning

**Source:** [`src/ai/training/mod.zig`](../../src/ai/training/mod.zig)

---

Training Sub-module

Training pipeline utilities, gradient aggregation, and checkpointing.

Provides neural network training with SGD, Adam optimizers, learning rate scheduling,
gradient clipping, loss functions, and mixed precision support.

---

## API

### `pub const Context`

<sup>**type**</sup>

Training context for framework integration.

### `pub fn train(self: *Context, train_config: TrainingConfig) !TrainingResult`

<sup>**fn**</sup>

Run training with the given configuration.

### `pub fn getCheckpointStore(self: *Context) !*CheckpointStore`

<sup>**fn**</sup>

Get or create checkpoint store.

### `pub fn saveCheckpoint(self: *Context, name: []const u8, data: anytype) !void`

<sup>**fn**</sup>

Save a checkpoint.

### `pub fn loadCheckpointData(self: *Context, name: []const u8, comptime T: type) !T`

<sup>**fn**</sup>

Load a checkpoint.

---

*Generated automatically by `zig build gendocs`*
