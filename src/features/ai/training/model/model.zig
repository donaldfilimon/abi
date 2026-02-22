//! Trainable LLM model.
//!
//! This module re-exports the canonical `TrainableModel` implementation from
//! `trainable_model/core.zig`.  The split model directory (`model/`) provides
//! config, layer, weights, cache, and utility types.  The full model struct
//! (with forward, backward, checkpoint, etc.) lives in `trainable_model/`.

const core = @import("../trainable_model/core.zig");

pub const TrainableModel = core.TrainableModel;

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
