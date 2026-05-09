//! Embedding Backends Module
//!
//! Provides implementations of the embedding backend interface for
//! various providers.
const std = @import("std");

pub const openai = @import("openai.zig");

pub const OpenAIBackend = openai.OpenAIBackend;
pub const OpenAIModel = openai.Model;

test {
    std.testing.refAllDecls(@This());
}
