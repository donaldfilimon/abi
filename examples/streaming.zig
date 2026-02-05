//! Streaming Example
//!
//! Demonstrates the ABI streaming API including SSE encoding,
//! stream events, and circuit breaker patterns for resilience.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("ABI Streaming API Example\n", .{});
    std.debug.print("=========================\n\n", .{});

    // Create stream events
    std.debug.print("Creating stream events:\n", .{});

    // Start event
    const start_event = abi.ai.streaming.StreamEvent.startEvent();
    std.debug.print("  Start event type: {t}\n", .{start_event.event_type});

    // Token events
    const tokens = [_][]const u8{ "Hello", ",", " ", "world", "!" };
    for (tokens, 0..) |text, i| {
        const token = abi.ai.streaming.StreamToken{
            .id = @intCast(i),
            .text = text,
            .is_end = (i == tokens.len - 1),
            .sequence_index = i,
        };
        const event = abi.ai.streaming.StreamEvent.tokenEvent(token);
        std.debug.print("  Token event: \"{s}\" (id={d}, is_end={})\n", .{
            text,
            token.id,
            token.is_end,
        });
        _ = event; // Use event
    }

    // End event
    const end_event = abi.ai.streaming.StreamEvent.endEvent();
    std.debug.print("  End event type: {t}\n", .{end_event.event_type});

    // Error event
    const error_event = abi.ai.streaming.StreamEvent.errorEvent("Connection timeout");
    std.debug.print("  Error event: {s}\n", .{error_event.error_message orelse "none"});

    // Heartbeat event
    const heartbeat = abi.ai.streaming.StreamEvent.heartbeatEvent();
    std.debug.print("  Heartbeat event type: {t}\n\n", .{heartbeat.event_type});

    // SSE Encoding
    std.debug.print("SSE Encoding:\n", .{});
    var encoder = abi.ai.streaming.sse.SseEncoder.init(allocator, .{
        .include_timestamp = true,
        .include_id = true,
    });

    const sample_token = abi.ai.streaming.StreamToken{
        .id = 42,
        .text = "example",
        .is_end = false,
        .sequence_index = 0,
    };
    const sample_event = abi.ai.streaming.StreamEvent.tokenEvent(sample_token);
    const encoded = try encoder.encode(sample_event);
    defer allocator.free(encoded);
    std.debug.print("  Encoded SSE message:\n{s}\n", .{encoded});

    // Stream event types overview
    std.debug.print("Available StreamEventType values:\n", .{});
    const event_types = [_]abi.ai.streaming.StreamEventType{
        .token,
        .start,
        .end,
        .error_event,
        .metadata,
        .heartbeat,
    };
    for (event_types) |et| {
        std.debug.print("  - {t}\n", .{et});
    }

    // Backend types (for routing)
    std.debug.print("\nAvailable BackendType values:\n", .{});
    const backend_types = [_]abi.ai.streaming.BackendType{
        .local,
        .openai,
        .ollama,
        .anthropic,
    };
    for (backend_types) |bt| {
        std.debug.print("  - {t}\n", .{bt});
    }

    std.debug.print("\nStreaming example completed successfully!\n", .{});
}
