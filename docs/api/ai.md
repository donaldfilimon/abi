# AI and Machine Learning API

This document provides comprehensive API documentation for the `ai` module.

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
var agent = try abi.ai.Agent.init(allocator, .creative);
defer agent.deinit();

const response = try agent.generate("Tell me a story", .{});
```

