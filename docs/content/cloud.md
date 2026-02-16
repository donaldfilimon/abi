---
title: "Cloud"
description: "Cloud function adapters for AWS, GCP, and Azure"
section: "Infrastructure"
order: 6
---

# Cloud

The Cloud module provides unified adapters for deploying ABI applications as
serverless functions across AWS Lambda, Google Cloud Functions, and Azure
Functions -- write your handler once and run it on any provider.

- **Build flag:** `-Denable-cloud=true` (default: enabled)
- **Namespace:** `abi.cloud`
- **Source:** `src/features/cloud/`

## Overview

The cloud module normalizes the differences between serverless providers behind
a common event/response model. Your function handler receives a `CloudEvent`
and returns a `CloudResponse`, regardless of whether it runs on AWS, GCP, or
Azure. The module handles provider detection, event parsing, and response
formatting automatically.

Key capabilities:

- **Unified event model** -- `CloudEvent` normalizes incoming events across all three providers
- **Unified response model** -- `CloudResponse` provides consistent response formatting
- **Provider-specific adapters** -- Optimized parsing and serialization for each cloud runtime
- **Auto-detection** -- `detectProvider()` reads environment variables to identify the runtime
- **Response builder** -- Fluent builder API for constructing responses with headers, CORS, and content types
- **Framework integration** -- `Context` struct with auto-detection and handler wrapping

### Supported Providers

| Provider | Adapter Module | Detection Variables |
|----------|---------------|---------------------|
| AWS Lambda | `abi.cloud.aws_lambda` | `AWS_LAMBDA_RUNTIME_API`, `AWS_LAMBDA_FUNCTION_NAME` |
| Google Cloud Functions | `abi.cloud.gcp_functions` | `K_SERVICE`, `FUNCTION_NAME`, `GOOGLE_CLOUD_PROJECT` |
| Azure Functions | `abi.cloud.azure_functions` | `FUNCTIONS_WORKER_RUNTIME`, `AZURE_FUNCTIONS_ENVIRONMENT`, `WEBSITE_SITE_NAME` |

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");
const cloud = abi.cloud;

/// Your function handler -- same code works on all providers.
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    const body = event.body orelse "{}";
    _ = body;

    return try cloud.CloudResponse.json(allocator,
        \\{"message": "Hello from the cloud!"}
    );
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Auto-detect provider and run
    try cloud.runHandler(allocator, handler);
}
```

### Targeting a Specific Provider

```zig
// Deploy to AWS Lambda
try cloud.aws_lambda.runHandler(allocator, handler);

// Deploy to Google Cloud Functions (specify port)
try cloud.gcp_functions.runHandler(allocator, handler, 8080);

// Deploy to Azure Functions
try cloud.azure_functions.runHandler(allocator, handler);
```

### Using the Context API

```zig
var ctx = try cloud.Context.init(allocator, .{
    .memory_mb = 512,
    .timeout_seconds = 60,
    .tracing_enabled = true,
});
defer ctx.deinit();

// Check detected provider
if (ctx.getProvider()) |provider| {
    std.debug.print("Running on: {s}\n", .{provider.name()});
}

// Check if running in a cloud environment
if (ctx.isCloudEnvironment()) {
    // Cloud-specific initialization
}
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context with auto-detection and handler wrapping |
| `CloudConfig` | Memory, timeout, tracing, logging, cold start optimization settings |
| `CloudEvent` | Normalized incoming event: request ID, provider, HTTP method, path, headers, body |
| `CloudResponse` | Normalized response: status, headers, body, content type |
| `CloudProvider` | Enum: `aws_lambda`, `gcp_functions`, `azure_functions` |
| `CloudHandler` | Handler function type: `fn (*CloudEvent, Allocator) !CloudResponse` |
| `HttpMethod` | GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS |
| `InvocationMetadata` | Provider-specific invocation metadata |
| `CloudError` | Error set for cloud operations |

### Provider Adapters

| Module | Description |
|--------|-------------|
| `abi.cloud.aws_lambda` | AWS Lambda runtime adapter |
| `abi.cloud.gcp_functions` | Google Cloud Functions HTTP adapter |
| `abi.cloud.azure_functions` | Azure Functions adapter |

### Response Builder

| Method | Description |
|--------|-------------|
| `ResponseBuilder.init(allocator)` | Create a new builder |
| `.status(code)` | Set HTTP status code |
| `.header(name, value)` | Add a response header |
| `.json()` | Set content type to `application/json` |
| `.text()` | Set content type to `text/plain` |
| `.html()` | Set content type to `text/html` |
| `.body(content)` | Set response body |
| `.cors(origin)` | Add CORS headers for the specified origin |
| `.build()` | Finalize and return the `CloudResponse` |

### Key Functions

| Function | Description |
|----------|-------------|
| `detectProvider() ?CloudProvider` | Auto-detect cloud provider from environment variables |
| `detectProviderWithAllocator(allocator) ?CloudProvider` | Auto-detect with explicit allocator |
| `runHandler(allocator, handler) !void` | Run handler on the detected provider |
| `isEnabled() bool` | Returns `true` if cloud is compiled in |
| `isInitialized() bool` | Returns `true` if the module is initialized |

## Configuration

Cloud is configured through the `CloudConfig` struct:

```zig
const config = abi.cloud.CloudConfig{
    .provider = .auto,          // auto-detect from environment
    .memory_mb = 256,           // memory allocation in MB
    .timeout_seconds = 30,      // function timeout
    .tracing_enabled = false,   // distributed tracing
    .logging_enabled = true,    // structured logging
    .log_level = .info,         // log verbosity
};
```

| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `.auto` | Target provider (auto-detect if not specified) |
| `memory_mb` | 256 | Memory allocation in MB |
| `timeout_seconds` | 30 | Function timeout in seconds |
| `tracing_enabled` | `false` | Enable distributed tracing |
| `logging_enabled` | `true` | Enable structured logging |
| `log_level` | `.info` | Log level for execution |

## CLI Commands

The cloud module does not have a dedicated CLI command. Use the cloud API
programmatically or through the Framework builder.

## Examples

See `examples/cloud.zig` for a complete working example that enumerates
providers, creates cloud events, and demonstrates the configuration API:

```bash
zig build run-cloud
```

## Disabling at Build Time

```bash
# Compile without cloud support
zig build -Denable-cloud=false
```

When disabled, all public functions return `error.CloudDisabled` and
`isEnabled()` returns `false`. The stub module preserves identical type
signatures -- including full stub implementations of provider adapters and
the `ResponseBuilder` -- so downstream code compiles without conditional guards.

## Related

- [Web](web.html) -- HTTP client utilities
- [Gateway](gateway.html) -- API gateway for inbound routing
- [Network](network.html) -- Distributed compute and node management
