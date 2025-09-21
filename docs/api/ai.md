# AI and Machine Learning API

This document provides comprehensive API documentation for the `abi.features.ai`
namespace that is re-exported from `src/mod.zig`.

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The AI module provides multi-persona agents, neural networks, and machine learning utilities.

### Personas

- **Helpful**: General-purpose assistant
- **Creative**: Artistic and imaginative responses
- **Analytical**: Data-driven analysis
- **Casual**: Informal conversation

## Examples

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const ai = abi.features.ai;
    var agent = try ai.agent.Agent.init(allocator, .{ .name = "Storyteller", .persona = .creative });
    defer agent.deinit();

    const reply = try agent.process("Tell me a story", allocator);
    defer allocator.free(reply);

    std.log.info("Agent reply: {s}", .{reply});
}
```

