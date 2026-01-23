//! Text generation module.
//!
//! Provides token sampling, text generation, and streaming output.

const std = @import("std");

pub const sampler = @import("sampler.zig");
pub const generator = @import("generator.zig");
pub const batch = @import("batch.zig");
pub const streaming = @import("streaming.zig");

// Re-exports
pub const Sampler = sampler.Sampler;
pub const SamplerConfig = sampler.SamplerConfig;
pub const TopKTopP = sampler.TopKTopP;

pub const Generator = generator.Generator;
pub const GeneratorConfig = generator.GeneratorConfig;
pub const GenerationResult = generator.GenerationResult;

pub const BatchGenerator = batch.BatchGenerator;

// Streaming re-exports
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamingState = streaming.StreamingState;
pub const StreamingStats = streaming.StreamingStats;
pub const StreamingCallbacks = streaming.StreamingCallbacks;
pub const StreamingConfig = streaming.StreamingConfig;
pub const StreamingError = streaming.StreamingError;
pub const StreamingResponse = streaming.StreamingResponse;
pub const TokenEvent = streaming.TokenEvent;
pub const SSEFormatter = streaming.SSEFormatter;
pub const streamToStdout = streaming.streamToStdout;
pub const printCompletionStats = streaming.printCompletionStats;
pub const collectStreamingResponse = streaming.collectStreamingResponse;

test "generation module imports" {
    _ = sampler;
    _ = generator;
    _ = batch;
    _ = streaming;
}
