const std = @import("std");

pub const types = @import("types.zig");
pub const generator = @import("generator.zig");
pub const sse = @import("sse.zig");
pub const response = @import("response.zig");

// Re-export common types and functions for convenience
pub const StreamingError = types.StreamingError;
pub const StreamingState = types.StreamingState;
pub const TokenEvent = types.TokenEvent;
pub const StreamingStats = types.StreamingStats;
pub const StreamingCallbacks = types.StreamingCallbacks;
pub const StreamingConfig = types.StreamingConfig;

pub const StreamingGenerator = generator.StreamingGenerator;
pub const streamToStdout = generator.streamToStdout;
pub const streamEventToWriter = generator.streamEventToWriter;
pub const printCompletionStats = generator.printCompletionStats;
pub const writeCompletionStats = generator.writeCompletionStats;

pub const SSEFormatter = sse.SSEFormatter;

pub const StreamingResponse = response.StreamingResponse;
pub const collectStreamingResponse = response.collectStreamingResponse;

test {
    std.testing.refAllDecls(@This());
    _ = types;
    _ = generator;
    _ = sse;
    _ = response;
}
