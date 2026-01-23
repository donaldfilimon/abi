---
title: Cloud Deployment Guide
description: Deploy ABI applications as serverless functions on AWS Lambda, Google Cloud Functions, and Azure Functions
---

# Cloud Deployment Guide

The ABI framework provides unified cloud function adapters for deploying your applications as serverless functions across major cloud providers. This guide covers deployment to AWS Lambda, Google Cloud Functions, and Azure Functions.

## Overview

The cloud module (`src/cloud/`) provides:

- **Unified Event Model**: A common `CloudEvent` struct that normalizes incoming events across all providers
- **Unified Response Model**: A common `CloudResponse` struct for consistent response handling
- **Provider-Specific Adapters**: Optimized parsing and formatting for each cloud provider
- **Context Extraction**: Access to provider-specific context and metadata
- **Auto-Detection**: Automatic provider detection based on environment variables

## Quick Start

### 1. Write Your Handler

Create a handler function that works on all providers:

```zig
const std = @import("std");
const abi = @import("abi");
const cloud = abi.cloud;

/// Your function handler - same code works on all providers
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    // Access request data uniformly
    const method = if (event.method) |m| m.toString() else "N/A";
    const path = event.path orelse "/";
    const body = event.body orelse "{}";

    // Log the request
    std.log.info("Request: {s} {s}", .{method, path});

    // Process the request
    var response_body = std.ArrayList(u8).init(allocator);
    defer response_body.deinit();

    try std.json.stringify(.{
        .message = "Hello from the cloud!",
        .provider = event.provider.name(),
        .request_id = event.request_id,
    }, .{}, response_body.writer());

    // Return a JSON response
    return try cloud.CloudResponse.json(allocator, try response_body.toOwnedSlice());
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Auto-detect provider and run
    try cloud.runHandler(allocator, handler);
}
```

### 2. Build for Your Target Platform

```bash
# For AWS Lambda (x86_64 Linux)
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast

# For AWS Lambda (ARM64 Linux - recommended for cost)
zig build -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseFast

# For GCP Cloud Functions (x86_64 Linux)
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast

# For Azure Functions (depends on your Function App plan)
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast
```

### 3. Deploy

See provider-specific sections below for deployment instructions.

## AWS Lambda

### Prerequisites

- AWS CLI configured with appropriate credentials
- AWS SAM CLI installed
- IAM permissions for Lambda, API Gateway, CloudWatch Logs

### Deployment with SAM

1. Navigate to the deployment directory:

```bash
cd deploy/aws
```

2. Build and deploy:

```bash
sam build && sam deploy --guided
```

3. Follow the prompts to configure your deployment.

### SAM Template Structure

The `template.yaml` includes:

- **HttpFunction**: HTTP API Gateway triggered function
- **QueueFunction**: SQS triggered function
- **ScheduledFunction**: CloudWatch Events (cron) triggered function

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AWS_LAMBDA_RUNTIME_API` | Runtime API endpoint (set by Lambda) |
| `AWS_LAMBDA_FUNCTION_NAME` | Function name (set by Lambda) |
| `LOG_LEVEL` | Logging level (debug, info, warn, error) |
| `ABI_CLOUD_PROVIDER` | Set to `aws_lambda` |

### Event Parsing

The AWS adapter handles:

- API Gateway REST API events (v1)
- API Gateway HTTP API events (v2)
- SQS events
- SNS events
- S3 events
- CloudWatch Events

```zig
// Access AWS-specific context
const function_arn = event.context.function_arn;
const remaining_time = event.context.remaining_time_ms;
```

## Google Cloud Functions

### Prerequisites

- Google Cloud SDK installed
- GCP project configured
- Cloud Functions API enabled

### Deployment

1. Build for Linux:

```bash
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast
```

2. Deploy with gcloud:

```bash
gcloud functions deploy abi-function \
  --gen2 \
  --runtime=provided \
  --region=us-central1 \
  --source=./zig-out/bin \
  --entry-point=handler \
  --trigger-http \
  --allow-unauthenticated
```

### Configuration Options

See `deploy/gcp/cloudfunctions.yaml` for full configuration options including:

- HTTP triggers
- Pub/Sub triggers
- Cloud Storage triggers
- Firestore triggers

### Environment Variables

| Variable | Description |
|----------|-------------|
| `K_SERVICE` | Service name (set by Cloud Run) |
| `FUNCTION_NAME` | Function name (legacy) |
| `PORT` | HTTP server port (set by runtime) |
| `GCP_PROJECT` | GCP project ID |
| `FUNCTION_REGION` | Deployment region |

### Event Parsing

The GCP adapter handles:

- HTTP requests
- CloudEvents (structured)
- Pub/Sub messages
- Cloud Storage events

```zig
// Access GCP-specific context
const project_id = event.context.project_id;
const region = event.context.region;
```

## Azure Functions

### Prerequisites

- Azure CLI installed
- Azure Functions Core Tools installed
- Azure subscription configured

### Deployment

1. Build for your target platform:

```bash
zig build -Dtarget=x86_64-linux-gnu -Doptimize=ReleaseFast
```

2. Deploy with Azure CLI:

```bash
# Create a Function App (if not exists)
az functionapp create \
  --resource-group myResourceGroup \
  --consumption-plan-location westus \
  --runtime custom \
  --functions-version 4 \
  --name abi-function-app \
  --storage-account mystorageaccount

# Deploy
func azure functionapp publish abi-function-app
```

### Configuration Files

- `function.json`: Trigger and binding configuration
- `host.json`: Host-level configuration
- `local.settings.json`: Local development settings

### Environment Variables

| Variable | Description |
|----------|-------------|
| `FUNCTIONS_HTTPWORKER_PORT` | Custom handler port |
| `FUNCTION_NAME` | Function name |
| `WEBSITE_SITE_NAME` | App Service name |
| `AZURE_FUNCTIONS_ENVIRONMENT` | Environment (Development, Production) |

### Event Parsing

The Azure adapter handles:

- HTTP triggers
- Timer triggers
- Blob Storage triggers
- Queue triggers
- Event Hub triggers
- Service Bus triggers

```zig
// Access Azure-specific context
const invocation_id = event.context.invocation_id;
const function_name = event.context.function_name;
```

## Advanced Usage

### Response Builder

Use the fluent response builder for complex responses:

```zig
var response = cloud.ResponseBuilder.init(allocator)
    .status(201)
    .json()
    .header("X-Custom-Header", "value")
    .cors("*")
    .body("{\"created\":true}")
    .build();
```

### Error Handling

Return structured error responses:

```zig
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
    // Validate request
    if (event.body == null) {
        return cloud.CloudResponse.err(allocator, 400, "Request body is required");
    }

    // Process or return error
    processRequest(event) catch |err| {
        return cloud.CloudResponse.err(allocator, 500, @errorName(err));
    };

    return cloud.CloudResponse.json(allocator, "{\"success\":true}");
}
```

### Provider Detection

Detect the current provider at runtime:

```zig
const provider = cloud.detectProvider();
if (provider) |p| {
    std.log.info("Running on {s}", .{p.name()});
} else {
    std.log.info("Not running in a cloud environment");
}
```

### Framework Integration

Use the cloud module with the full ABI framework:

```zig
const abi = @import("abi");

fn handler(event: *abi.cloud.CloudEvent, allocator: std.mem.Allocator) !abi.cloud.CloudResponse {
    // Initialize framework for this request
    var fw = try abi.initDefault(allocator);
    defer fw.deinit();

    // Use framework features
    if (fw.isEnabled(.ai)) {
        // Use AI features
    }

    if (fw.isEnabled(.database)) {
        // Use database features
    }

    return abi.cloud.CloudResponse.json(allocator, "{\"ok\":true}");
}
```

## Configuration Reference

### CloudConfig

```zig
const config = abi.cloud.CloudConfig{
    .memory_mb = 512,
    .timeout_seconds = 30,
    .tracing_enabled = true,
    .logging_enabled = true,
    .log_level = .info,
    .cors = .{
        .allowed_origins = &.{"https://example.com"},
        .allowed_methods = &.{ .GET, .POST },
    },
};
```

### CORS Configuration

Enable CORS for HTTP functions:

```zig
.cors = .{
    .allowed_origins = &.{"*"},
    .allowed_methods = &.{ .GET, .POST, .PUT, .DELETE, .OPTIONS },
    .allowed_headers = &.{ "Content-Type", "Authorization" },
    .max_age_seconds = 86400,
    .allow_credentials = false,
},
```

## Troubleshooting

### Common Issues

**Function times out**
- Increase the timeout in your deployment configuration
- Check for blocking operations or infinite loops
- Ensure database connections are properly managed

**Cold start latency**
- Use provisioned concurrency (AWS) or minimum instances (GCP)
- Reduce function size by disabling unused features
- Use ARM64 architecture on AWS for faster cold starts

**Memory errors**
- Increase memory allocation
- Check for memory leaks in your handler
- Use `GeneralPurposeAllocator` with leak detection in debug builds

**Permission denied**
- Verify IAM roles/service accounts have required permissions
- Check VPC configuration if accessing private resources

### Logging

Enable debug logging for troubleshooting:

```bash
# Environment variable
LOG_LEVEL=debug
```

View logs:

```bash
# AWS
aws logs tail /aws/lambda/abi-function --follow

# GCP
gcloud functions logs read abi-function

# Azure
func azure functionapp logstream abi-function-app
```

## Best Practices

1. **Keep handlers small**: Focus on request handling, delegate business logic
2. **Use connection pooling**: Reuse database connections across invocations
3. **Handle cold starts**: Initialize expensive resources outside the handler
4. **Set appropriate timeouts**: Match timeout to expected execution time
5. **Use structured logging**: Include request IDs for tracing
6. **Implement retries**: Handle transient failures gracefully
7. **Validate inputs**: Check request data before processing
8. **Return appropriate status codes**: Use HTTP semantics correctly

## See Also

- [API Reference](api/index.md) - Full API documentation
- [Architecture Overview](architecture/overview.md) - System architecture
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
