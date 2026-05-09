//! Cloud Provider Detection
//!
//! Detects which cloud provider environment the application is running in
//! by checking environment variables for AWS Lambda, GCP Functions, and Azure Functions.

const std = @import("std");
const types = @import("types.zig");
const CloudProvider = types.CloudProvider;

/// Detect which cloud provider environment we're running in.
/// This function requires an allocator to read environment variables in Zig 0.17.
pub fn detectProvider() ?CloudProvider {
    return detectProviderWithAllocator(std.heap.page_allocator);
}

/// Detect which cloud provider environment we're running in (with explicit allocator).
pub fn detectProviderWithAllocator(allocator: std.mem.Allocator) ?CloudProvider {
    // Get environment map using Zig 0.17 API
    var env_map = std.process.Environ.createMap(std.process.Environ.empty, allocator) catch return null;
    defer env_map.deinit();

    // AWS Lambda
    if (env_map.get("AWS_LAMBDA_RUNTIME_API") != null or
        env_map.get("AWS_LAMBDA_FUNCTION_NAME") != null)
    {
        return .aws_lambda;
    }

    // Google Cloud Functions
    if (env_map.get("K_SERVICE") != null or
        env_map.get("FUNCTION_NAME") != null or
        env_map.get("GOOGLE_CLOUD_PROJECT") != null)
    {
        return .gcp_functions;
    }

    // Azure Functions
    if (env_map.get("FUNCTIONS_WORKER_RUNTIME") != null or
        env_map.get("AZURE_FUNCTIONS_ENVIRONMENT") != null or
        env_map.get("WEBSITE_SITE_NAME") != null)
    {
        return .azure_functions;
    }

    return null;
}
