//! Async streaming generation support.
//!
//! Provides asynchronous token generation with callback-based streaming,
//! compatible with web servers and interactive applications.

const std = @import("std");
const streaming_mod = @import("streaming/mod.zig");

// Re-export everything from the sub-module
pub const types = streaming_mod.types;
pub const generator = streaming_mod.generator;
pub const sse = streaming_mod.sse;
pub const response = streaming_mod.response;

pub const StreamingError = streaming_mod.StreamingError;
pub const StreamingState = streaming_mod.StreamingState;
pub const TokenEvent = streaming_mod.TokenEvent;
pub const StreamingStats = streaming_mod.StreamingStats;
pub const StreamingCallbacks = streaming_mod.StreamingCallbacks;
pub const StreamingConfig = streaming_mod.StreamingConfig;

pub const StreamingGenerator = streaming_mod.StreamingGenerator;
pub const streamToStdout = streaming_mod.streamToStdout;
pub const streamEventToWriter = streaming_mod.streamEventToWriter;
pub const printCompletionStats = streaming_mod.printCompletionStats;
pub const writeCompletionStats = streaming_mod.writeCompletionStats;

pub const SSEFormatter = streaming_mod.SSEFormatter;

pub const StreamingResponse = streaming_mod.StreamingResponse;
pub const collectStreamingResponse = streaming_mod.collectStreamingResponse;

test {
    std.testing.refAllDecls(@This());
}
