//! Text generation module.
//!
//! Provides token sampling, text generation, and streaming output.

const std = @import("std");

pub const sampler = @import("sampler.zig");
pub const generator = @import("generator.zig");
pub const batch = @import("batch.zig");

// Re-exports
pub const Sampler = sampler.Sampler;
pub const SamplerConfig = sampler.SamplerConfig;
pub const TopKTopP = sampler.TopKTopP;

pub const Generator = generator.Generator;
pub const GeneratorConfig = generator.GeneratorConfig;
pub const GenerationResult = generator.GenerationResult;

pub const BatchGenerator = batch.BatchGenerator;

test "generation module imports" {
    _ = sampler;
    _ = generator;
    _ = batch;
}
