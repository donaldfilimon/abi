---
title: cloud API
purpose: Generated API reference for cloud
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# cloud

> Cloud Functions Module

Provides unified adapters for deploying ABI applications as serverless
functions across major cloud providers: AWS Lambda, Google Cloud Functions,
and Azure Functions.

## Features

- **Unified Event Model**: Common `CloudEvent` struct that normalizes events
across all providers
- **Unified Response Model**: Common `CloudResponse` struct for consistent
response handling
- **Provider-Specific Adapters**: Optimized parsing and formatting for each
cloud provider
- **Context Extraction**: Access provider-specific context and metadata

## Quick Start

```zig
const std = @import("std");
const abi = @import("abi");
const cloud = abi.cloud;

/// Your function handler - same code works on all providers
fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
// Access request data uniformly
const body = event.body orelse "{}";

// Return a JSON response
return try cloud.CloudResponse.json(allocator,
\\{"message": "Hello from the cloud!"}
);
}

pub fn main(init: std.process.Init) !void {
const arena = init.arena.allocator();

// Deploy to AWS Lambda
try cloud.aws_lambda.runHandler(arena, handler);

// Or Google Cloud Functions
// try cloud.gcp_functions.runHandler(arena, handler, 8080);

// Or Azure Functions
// try cloud.azure_functions.runHandler(arena, handler);
}
```

## Deployment

See the deployment templates in `deploy/` for provider-specific configurations:
- `deploy/aws/template.yaml` - AWS SAM template
- `deploy/gcp/cloudfunctions.yaml` - GCP configuration
- `deploy/azure/function.json` - Azure Functions configuration

**Source:** [`src/features/cloud/mod.zig`](../../src/features/cloud/mod.zig)

**Build flag:** `-Dfeat_cloud=true`

---

## API

### <a id="pub-const-error"></a>`pub const Error`

<sup>**const**</sup> | [source](../../src/features/cloud/mod.zig#L76)

Cloud module errors.

### <a id="pub-const-context"></a>`pub const Context`

<sup>**const**</sup> | [source](../../src/features/cloud/mod.zig#L83)

Cloud module context for Framework integration.

### <a id="pub-fn-init-allocator-std-mem-allocator-cfg-cloudconfig-context"></a>`pub fn init(allocator: std.mem.Allocator, cfg: CloudConfig) !*Context`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L89)

Initialize the cloud context.

### <a id="pub-fn-deinit-self-context-void"></a>`pub fn deinit(self: *Context) void`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L105)

Deinitialize the cloud context.

### <a id="pub-fn-getprovider-self-const-context-cloudprovider"></a>`pub fn getProvider(self: *const Context) ?CloudProvider`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L110)

Get the detected cloud provider.

### <a id="pub-fn-iscloudenvironment-self-const-context-bool"></a>`pub fn isCloudEnvironment(self: *const Context) bool`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L115)

Check if running in a cloud function environment.

### <a id="pub-fn-wraphandler-self-context-comptime-handler-fn-cloudevent-std-mem-allocator-anyerror-cloudresponse-cloudhandler"></a>`pub fn wrapHandler( self: *Context, comptime handler: fn (*CloudEvent, std.mem.Allocator) anyerror!CloudResponse, ) CloudHandler`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L120)

Create a handler wrapper that integrates with the ABI framework.

### <a id="pub-fn-detectprovider-cloudprovider"></a>`pub fn detectProvider() ?CloudProvider`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L135)

Detect which cloud provider environment we're running in.
This function requires an allocator to read environment variables in Zig 0.16.

### <a id="pub-fn-detectproviderwithallocator-allocator-std-mem-allocator-cloudprovider"></a>`pub fn detectProviderWithAllocator(allocator: std.mem.Allocator) ?CloudProvider`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L140)

Detect which cloud provider environment we're running in (with explicit allocator).

### <a id="pub-fn-runhandler-allocator-std-mem-allocator-handler-cloudhandler-void"></a>`pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L173)

Run a handler on the detected cloud provider.
Automatically selects the appropriate runtime based on environment detection.

### <a id="pub-const-responsebuilder"></a>`pub const ResponseBuilder`

<sup>**const**</sup> | [source](../../src/features/cloud/mod.zig#L188)

Create a response helper for common response patterns.

### <a id="pub-fn-status-self-responsebuilder-code-u16-responsebuilder"></a>`pub fn status(self: *ResponseBuilder, code: u16) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L200)

Set HTTP status code.

### <a id="pub-fn-header-self-responsebuilder-key-const-u8-value-const-u8-responsebuilder"></a>`pub fn header(self: *ResponseBuilder, key: []const u8, value: []const u8) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L206)

Add a header.

### <a id="pub-fn-json-self-responsebuilder-responsebuilder"></a>`pub fn json(self: *ResponseBuilder) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L214)

Set content type to JSON.

### <a id="pub-fn-text-self-responsebuilder-responsebuilder"></a>`pub fn text(self: *ResponseBuilder) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L219)

Set content type to plain text.

### <a id="pub-fn-html-self-responsebuilder-responsebuilder"></a>`pub fn html(self: *ResponseBuilder) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L224)

Set content type to HTML.

### <a id="pub-fn-body-self-responsebuilder-content-const-u8-responsebuilder"></a>`pub fn body(self: *ResponseBuilder, content: []const u8) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L229)

Set the response body.

### <a id="pub-fn-cors-self-responsebuilder-origin-const-u8-responsebuilder"></a>`pub fn cors(self: *ResponseBuilder, origin: []const u8) *ResponseBuilder`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L235)

Add CORS headers.

### <a id="pub-fn-build-self-responsebuilder-cloudresponse"></a>`pub fn build(self: *ResponseBuilder) CloudResponse`

<sup>**fn**</sup> | [source](../../src/features/cloud/mod.zig#L243)

Build the final response.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
