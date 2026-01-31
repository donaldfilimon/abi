# Cloud Functions Module
> **Last reviewed:** 2026-01-31

Unified adapters for deploying ABI applications as serverless functions across major cloud providers.

## Overview

The Cloud module provides a consistent interface for deploying Zig-based ABI applications as serverless functions on:

- **AWS Lambda** - Amazon's serverless compute platform
- **Google Cloud Functions** - Google's serverless platform
- **Azure Functions** - Microsoft's serverless platform

Write once, deploy anywhere. Your function handler works identically across all three providers.

## Features

- **Unified Event Model**: Common `CloudEvent` struct that normalizes events across all providers
- **Unified Response Model**: Common `CloudResponse` struct for consistent response handling
- **Automatic Provider Detection**: Detects which cloud environment your code is running on
- **Provider-Specific Adapters**: Optimized parsing and formatting for AWS Lambda, GCP Functions, and Azure Functions
- **Context Extraction**: Access provider-specific metadata and context
- **Response Builder**: Fluent API for constructing responses with headers, status codes, and content types
- **CORS Support**: Built-in CORS header management

## Supported Providers

| Provider | Runtime | Detection |
|----------|---------|-----------|
| AWS Lambda | `provided.al2023` | `AWS_LAMBDA_RUNTIME_API`, `AWS_LAMBDA_FUNCTION_NAME` |
| Google Cloud Functions | `zig-runtime` | `K_SERVICE`, `FUNCTION_NAME`, `GOOGLE_CLOUD_PROJECT` |
| Azure Functions | `custom` | `FUNCTIONS_WORKER_RUNTIME`, `AZURE_FUNCTIONS_ENVIRONMENT`, `WEBSITE_SITE_NAME` |

## Types

### CloudEvent

Unified representation of an incoming request/event:

```zig
pub struct CloudEvent {
    request_id: []const u8,           // Unique request identifier
    provider: CloudProvider,           // Source cloud provider
    method: ?HttpMethod,               // HTTP method (GET, POST, etc.)
    path: ?[]const u8,                 // Request path
    query_params: ?StringHashMap,      // Query parameters
    headers: ?StringHashMap,           // Request headers (case-insensitive)
    body: ?[]const u8,                 // Raw request body
    json_body: ?json.Value,            // Parsed JSON body
    source: ?[]const u8,               // Event source ARN/resource
    event_type: ?[]const u8,           // Event type (s3:ObjectCreated, etc.)
    timestamp: i64,                    // Event timestamp
    context: ProviderContext,          // Provider-specific metadata
}
```

**Useful methods:**

- `getHeader(key)` - Get header value (case-insensitive)
- `getQueryParam(key)` - Get query parameter value
- `isHttpRequest()` - Check if this is an HTTP request
- `isJsonRequest()` - Check if content-type is JSON
- `getContentType()` - Get content-type header value

### CloudResponse

HTTP response to return from a function:

```zig
pub struct CloudResponse {
    status_code: u16,                  // HTTP status code (default: 200)
    headers: StringHashMap,            // Response headers
    body: []const u8,                  // Response body
    is_base64_encoded: bool,           // Whether body is base64 encoded
}
```

**Static constructors:**

- `json(allocator, body)` - Create JSON response (status 200)
- `text(allocator, body)` - Create plain text response (status 200)
- `err(allocator, status_code, message)` - Create error response with JSON error object

### CloudProvider

Enumeration of supported cloud providers:

```zig
pub enum CloudProvider {
    aws_lambda,
    gcp_functions,
    azure_functions,
}
```

## Quick Start

### Basic Handler

```zig
const std = @import("std");
const abi = @import("abi");
const cloud = abi.cloud;

fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    // Access request data uniformly across all providers
    const body = event.body orelse "{}";

    // Return a JSON response
    return try cloud.CloudResponse.json(allocator,
        \\{"message": "Hello from the cloud!"}
    );
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Automatically detects and runs on the appropriate cloud provider
    try cloud.runHandler(allocator, handler);
}
```

### Using Response Builder

```zig
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    var builder = cloud.ResponseBuilder.init(allocator);

    var response = builder
        .status(201)
        .json()
        .header("X-Custom-Header", "value")
        .cors("https://example.com")
        .body("{\"id\": 123}")
        .build();

    return response;
}
```

### HTTP-Triggered Function

```zig
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    if (!event.isHttpRequest()) {
        return try cloud.CloudResponse.err(allocator, 400, "Expected HTTP request");
    }

    const method = event.method.?;
    const path = event.path orelse "/";

    const message = try std.fmt.allocPrint(allocator,
        "Got {s} {s}",
        .{ method.toString(), path }
    );
    defer allocator.free(message);

    return try cloud.CloudResponse.text(allocator, message);
}
```

### Provider Detection

```zig
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    const provider_name = event.provider.name();  // "AWS Lambda", etc.

    switch (event.provider) {
        .aws_lambda => {
            // AWS-specific: access event.context.function_arn, etc.
        },
        .gcp_functions => {
            // GCP-specific: access event.context.project_id, etc.
        },
        .azure_functions => {
            // Azure-specific: access event.context.function_name, etc.
        },
    }

    return try cloud.CloudResponse.json(allocator,
        try std.fmt.allocPrint(allocator,
            "\\{{\"provider\":\"{s}\"}}",
            .{ provider_name }
        )
    );
}
```

## Configuration

### CloudConfig

Configure cloud function behavior:

```zig
const config = cloud.CloudConfig{
    .memory_mb = 512,
    .timeout_seconds = 60,
    .tracing_enabled = true,
    .logging_enabled = true,
    .log_level = .info,
    .cors = .{
        .allowed_origins = &.{ "https://example.com", "https://app.example.com" },
        .allowed_methods = &.{ .GET, .POST, .OPTIONS },
        .max_age_seconds = 3600,
    },
};
```

## Deployment

See deployment templates for provider-specific configurations:

- `deploy/aws/template.yaml` - AWS SAM (Serverless Application Model) template
- `deploy/gcp/cloudfunctions.yaml` - Google Cloud Functions configuration
- `deploy/azure/function.json` - Azure Functions configuration

## Error Handling

CloudError provides detailed error types:

```zig
pub enum CloudError {
    CloudDisabled,              // Feature disabled at build time
    InvalidEvent,               // Malformed event
    EventParseFailed,           // Event parsing error
    ResponseSerializeFailed,    // Response serialization error
    HandlerFailed,              // Handler execution error
    TimeoutExceeded,            // Function timeout
    InvalidConfig,              // Invalid configuration
    ProviderError,              // Provider-specific error
    OutOfMemory,                // Memory allocation failed
}
```

## Provider-Specific Context

Each provider includes specific metadata accessible via `event.context`:

**AWS Lambda:**
- `function_arn` - Function's ARN
- `log_group` - CloudWatch log group
- `log_stream` - CloudWatch log stream
- `remaining_time_ms` - Milliseconds until timeout

**Google Cloud Functions:**
- `project_id` - GCP project ID
- `region` - Function region

**Azure Functions:**
- `invocation_id` - Invocation ID
- `function_name` - Function name

## Integration with ABI Framework

```zig
const abi = @import("abi");

var cloud_context = try abi.cloud.Context.init(allocator, cloud_config);
defer cloud_context.deinit();

// Check if running in cloud environment
if (cloud_context.isCloudEnvironment()) {
    const provider = cloud_context.getProvider();
    std.debug.print("Running on {s}\n", .{ provider.?.name() });
}

// Get the detected provider
if (cloud_context.getProvider()) |provider| {
    std.debug.print("Cloud provider: {s}\n", .{ provider.name() });
}
```

## Testing

Test your handler locally before deployment:

```bash
# Run cloud module tests
zig test src/cloud/types.zig

# Or include in full test suite
zig build test --summary all
```

## See Also

- `CloudEvent` - Event type details
- `CloudResponse` - Response type details
- `ResponseBuilder` - Fluent response building
- AWS Lambda runtime: `src/cloud/aws_lambda.zig`
- GCP Functions runtime: `src/cloud/gcp_functions.zig`
- Azure Functions runtime: `src/cloud/azure_functions.zig`
