//! Stub Parity Verification
//!
//! Verifies that stub modules export the same public symbols as their
//! corresponding real modules. This ensures stubs stay in sync with
//! real implementations.
//!
//! Run with: zig build test --summary all

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const abi = @import("abi");

// ============================================================================
// Database Module Parity
// ============================================================================

test "database stub parity - types exist" {
    // These types should exist in both real and stub implementations
    // Since we're testing via abi module, we verify the public API surface
    const Database = abi.database;

    // Verify key exported types exist
    try testing.expect(@hasDecl(Database, "DatabaseHandle"));
    try testing.expect(@hasDecl(Database, "SearchResult"));
    try testing.expect(@hasDecl(Database, "Context"));
    try testing.expect(@hasDecl(Database, "Stats"));
    try testing.expect(@hasDecl(Database, "VectorView"));

    // Verify key functions exist
    try testing.expect(@hasDecl(Database, "open"));
    try testing.expect(@hasDecl(Database, "search"));
    try testing.expect(@hasDecl(Database, "insert"));
    try testing.expect(@hasDecl(Database, "isEnabled"));

    // Verify sub-modules
    try testing.expect(@hasDecl(Database, "wdbx"));
    try testing.expect(@hasDecl(Database, "fulltext"));
    try testing.expect(@hasDecl(Database, "hybrid"));
    try testing.expect(@hasDecl(Database, "filter"));
    try testing.expect(@hasDecl(Database, "batch"));
    try testing.expect(@hasDecl(Database, "clustering"));
    try testing.expect(@hasDecl(Database, "formats"));
}

// ============================================================================
// GPU Module Parity
// ============================================================================

test "gpu stub parity - types exist" {
    const Gpu = abi.gpu;

    // Core API surface (both mod.zig and stub.zig must have these)
    try testing.expect(@hasDecl(Gpu, "Context"));
    try testing.expect(@hasDecl(Gpu, "isEnabled"));

    // Error types (shared between both implementations)
    try testing.expect(@hasDecl(Gpu, "GpuError"));
    try testing.expect(@hasDecl(Gpu, "MemoryError"));
    try testing.expect(@hasDecl(Gpu, "KernelError"));

    // Key re-exported types
    try testing.expect(@hasDecl(Gpu, "Backend"));
    try testing.expect(@hasDecl(Gpu, "Device"));
    try testing.expect(@hasDecl(Gpu, "DeviceType"));
}

// ============================================================================
// Network Module Parity
// ============================================================================

test "network stub parity - types exist" {
    const Network = abi.network;

    try testing.expect(@hasDecl(Network, "Context"));
    try testing.expect(@hasDecl(Network, "isEnabled"));
}

// ============================================================================
// Web Module Parity
// ============================================================================

test "web stub parity - types exist" {
    const Web = abi.web;

    try testing.expect(@hasDecl(Web, "Context"));
    try testing.expect(@hasDecl(Web, "isEnabled"));
}

// ============================================================================
// Observability Module Parity
// ============================================================================

test "observability stub parity - types exist" {
    const Observability = abi.observability;

    try testing.expect(@hasDecl(Observability, "Context"));
    try testing.expect(@hasDecl(Observability, "isEnabled"));
}

// ============================================================================
// Analytics Module Parity
// ============================================================================

test "analytics stub parity - types exist" {
    const Analytics = abi.analytics;

    // Core types
    try testing.expect(@hasDecl(Analytics, "Engine"));
    try testing.expect(@hasDecl(Analytics, "Event"));
    try testing.expect(@hasDecl(Analytics, "AnalyticsConfig"));
    try testing.expect(@hasDecl(Analytics, "AnalyticsError"));

    // Extended types
    try testing.expect(@hasDecl(Analytics, "Funnel"));
    try testing.expect(@hasDecl(Analytics, "Experiment"));
}

// ============================================================================
// Cloud Module Parity
// ============================================================================

test "cloud stub parity - types exist" {
    const Cloud = abi.cloud;

    // Core types
    try testing.expect(@hasDecl(Cloud, "CloudEvent"));
    try testing.expect(@hasDecl(Cloud, "CloudResponse"));
    try testing.expect(@hasDecl(Cloud, "CloudProvider"));
    try testing.expect(@hasDecl(Cloud, "CloudHandler"));
    try testing.expect(@hasDecl(Cloud, "CloudConfig"));
    try testing.expect(@hasDecl(Cloud, "CloudError"));
    try testing.expect(@hasDecl(Cloud, "HttpMethod"));
    try testing.expect(@hasDecl(Cloud, "InvocationMetadata"));

    // Structs
    try testing.expect(@hasDecl(Cloud, "Context"));
    try testing.expect(@hasDecl(Cloud, "ResponseBuilder"));

    // Functions
    try testing.expect(@hasDecl(Cloud, "detectProvider"));
    try testing.expect(@hasDecl(Cloud, "detectProviderWithAllocator"));
    try testing.expect(@hasDecl(Cloud, "runHandler"));
    try testing.expect(@hasDecl(Cloud, "init"));
    try testing.expect(@hasDecl(Cloud, "deinit"));
    try testing.expect(@hasDecl(Cloud, "isEnabled"));
    try testing.expect(@hasDecl(Cloud, "isInitialized"));

    // Sub-modules
    try testing.expect(@hasDecl(Cloud, "aws_lambda"));
    try testing.expect(@hasDecl(Cloud, "gcp_functions"));
    try testing.expect(@hasDecl(Cloud, "azure_functions"));
}

// ============================================================================
// AI Module Parity
// ============================================================================

test "ai stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const AI = abi.ai;

    try testing.expect(@hasDecl(AI, "Context"));
    try testing.expect(@hasDecl(AI, "isEnabled"));
}

// ============================================================================
// AI Sub-module Parity Tests
// ============================================================================

test "ai/llm stub parity - types exist" {
    if (!build_options.enable_llm) return;

    const Llm = abi.ai.llm;
    _ = Llm; // Module exists and is accessible
}

test "ai/agents stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Agents = abi.ai.agents;
    _ = Agents; // Module exists and is accessible
}

test "ai/embeddings stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Embeddings = abi.ai.embeddings;
    _ = Embeddings; // Module exists and is accessible
}

test "ai/training stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Training = abi.ai.training;
    _ = Training; // Module exists and is accessible
}

// ============================================================================
// GPU Backend VTable Parity
// ============================================================================

/// Required methods for all GPU backend implementations.
/// These must match the VTable signatures in interface.zig.
const vtable_required_methods = [_][]const u8{
    "init",
    "deinit",
    "getDeviceCount",
    "getDeviceCaps",
    "allocate",
    "free",
    "copyToDevice",
    "copyFromDevice",
    "copyToDeviceAsync",
    "copyFromDeviceAsync",
    "compileKernel",
    "launchKernel",
    "destroyKernel",
    "synchronize",
};

fn verifyBackendHasMethods(comptime Backend: type) !void {
    inline for (vtable_required_methods) |method_name| {
        if (!@hasDecl(Backend, method_name)) {
            @compileError("GPU backend " ++ @typeName(Backend) ++ " missing required method: " ++ method_name);
        }
    }
}

test "gpu backend vtable parity - all backends implement required methods" {
    if (!build_options.enable_gpu) return error.SkipZigTest;

    const gpu_mod = abi.gpu;

    // Verify each backend type exports all VTable-required methods
    if (@hasDecl(gpu_mod, "backends")) {
        const backends = gpu_mod.backends;

        if (@hasDecl(backends, "CudaBackend"))
            try verifyBackendHasMethods(backends.CudaBackend);
        if (@hasDecl(backends, "MetalBackend"))
            try verifyBackendHasMethods(backends.MetalBackend);
        if (@hasDecl(backends, "OpenGLBackend"))
            try verifyBackendHasMethods(backends.OpenGLBackend);
        if (@hasDecl(backends, "OpenGLESBackend"))
            try verifyBackendHasMethods(backends.OpenGLESBackend);
        if (@hasDecl(backends, "WebGpuBackend"))
            try verifyBackendHasMethods(backends.WebGpuBackend);
        if (@hasDecl(backends, "FpgaBackend"))
            try verifyBackendHasMethods(backends.FpgaBackend);
        if (@hasDecl(backends, "VulkanBackend"))
            try verifyBackendHasMethods(backends.VulkanBackend);
    }
}

// ============================================================================
// Parity Verification Helpers
// ============================================================================

/// Verify a module has the expected minimal API surface for Context pattern
fn verifyContextPattern(comptime Module: type) !void {
    try testing.expect(@hasDecl(Module, "Context"));
    try testing.expect(@hasDecl(Module, "isEnabled"));
}

// ============================================================================
// Comprehensive Module Surface Test
// ============================================================================

// ============================================================================
// Auth Module Parity
// ============================================================================

test "auth stub parity - types exist" {
    const Auth = abi.auth;

    try testing.expect(@hasDecl(Auth, "AuthConfig"));
    try testing.expect(@hasDecl(Auth, "AuthError"));
    try testing.expect(@hasDecl(Auth, "Token"));
    try testing.expect(@hasDecl(Auth, "Session"));
    try testing.expect(@hasDecl(Auth, "Permission"));
    try testing.expect(@hasDecl(Auth, "Context"));
    try testing.expect(@hasDecl(Auth, "init"));
    try testing.expect(@hasDecl(Auth, "isEnabled"));
    try testing.expect(@hasDecl(Auth, "createToken"));
    try testing.expect(@hasDecl(Auth, "verifyToken"));
    try testing.expect(@hasDecl(Auth, "createSession"));
    try testing.expect(@hasDecl(Auth, "checkPermission"));

    // Security sub-modules (re-exported from services/shared/security)
    try testing.expect(@hasDecl(Auth, "jwt"));
    try testing.expect(@hasDecl(Auth, "api_keys"));
    try testing.expect(@hasDecl(Auth, "rbac"));
    try testing.expect(@hasDecl(Auth, "cors"));
    try testing.expect(@hasDecl(Auth, "rate_limit"));
    try testing.expect(@hasDecl(Auth, "encryption"));
    try testing.expect(@hasDecl(Auth, "tls"));
    try testing.expect(@hasDecl(Auth, "mtls"));
    try testing.expect(@hasDecl(Auth, "certificates"));
    try testing.expect(@hasDecl(Auth, "secrets"));
    try testing.expect(@hasDecl(Auth, "session"));
    try testing.expect(@hasDecl(Auth, "audit"));
    try testing.expect(@hasDecl(Auth, "password"));
    try testing.expect(@hasDecl(Auth, "validation"));
    try testing.expect(@hasDecl(Auth, "ip_filter"));
    try testing.expect(@hasDecl(Auth, "headers"));
}

// ============================================================================
// Messaging Module Parity
// ============================================================================

test "messaging stub parity - types exist" {
    const Messaging = abi.messaging;

    try testing.expect(@hasDecl(Messaging, "MessagingConfig"));
    try testing.expect(@hasDecl(Messaging, "MessagingError"));
    try testing.expect(@hasDecl(Messaging, "Message"));
    try testing.expect(@hasDecl(Messaging, "Channel"));
    try testing.expect(@hasDecl(Messaging, "DeliveryResult"));
    try testing.expect(@hasDecl(Messaging, "MessagingStats"));
    try testing.expect(@hasDecl(Messaging, "TopicInfo"));
    try testing.expect(@hasDecl(Messaging, "DeadLetter"));
    try testing.expect(@hasDecl(Messaging, "SubscriberCallback"));
    try testing.expect(@hasDecl(Messaging, "Context"));
    try testing.expect(@hasDecl(Messaging, "init"));
    try testing.expect(@hasDecl(Messaging, "isEnabled"));
    try testing.expect(@hasDecl(Messaging, "publish"));
    try testing.expect(@hasDecl(Messaging, "subscribe"));
    try testing.expect(@hasDecl(Messaging, "unsubscribe"));
    try testing.expect(@hasDecl(Messaging, "listTopics"));
    try testing.expect(@hasDecl(Messaging, "topicStats"));
    try testing.expect(@hasDecl(Messaging, "getDeadLetters"));
    try testing.expect(@hasDecl(Messaging, "clearDeadLetters"));
    try testing.expect(@hasDecl(Messaging, "messagingStats"));
}

// ============================================================================
// Cache Module Parity
// ============================================================================

test "cache stub parity - types exist" {
    const Cache = abi.cache;

    try testing.expect(@hasDecl(Cache, "CacheConfig"));
    try testing.expect(@hasDecl(Cache, "CacheError"));
    try testing.expect(@hasDecl(Cache, "CacheEntry"));
    try testing.expect(@hasDecl(Cache, "CacheStats"));
    try testing.expect(@hasDecl(Cache, "EvictionPolicy"));
    try testing.expect(@hasDecl(Cache, "Context"));
    try testing.expect(@hasDecl(Cache, "init"));
    try testing.expect(@hasDecl(Cache, "isEnabled"));
    try testing.expect(@hasDecl(Cache, "get"));
    try testing.expect(@hasDecl(Cache, "put"));
    try testing.expect(@hasDecl(Cache, "putWithTtl"));
    try testing.expect(@hasDecl(Cache, "delete"));
    try testing.expect(@hasDecl(Cache, "contains"));
    try testing.expect(@hasDecl(Cache, "clear"));
    try testing.expect(@hasDecl(Cache, "size"));
    try testing.expect(@hasDecl(Cache, "stats"));
}

// ============================================================================
// Storage Module Parity
// ============================================================================

test "storage stub parity - types exist" {
    const Storage = abi.storage;

    try testing.expect(@hasDecl(Storage, "StorageConfig"));
    try testing.expect(@hasDecl(Storage, "StorageBackend"));
    try testing.expect(@hasDecl(Storage, "StorageError"));
    try testing.expect(@hasDecl(Storage, "StorageObject"));
    try testing.expect(@hasDecl(Storage, "ObjectMetadata"));
    try testing.expect(@hasDecl(Storage, "StorageStats"));
    try testing.expect(@hasDecl(Storage, "Context"));
    try testing.expect(@hasDecl(Storage, "init"));
    try testing.expect(@hasDecl(Storage, "isEnabled"));
    try testing.expect(@hasDecl(Storage, "putObject"));
    try testing.expect(@hasDecl(Storage, "putObjectWithMetadata"));
    try testing.expect(@hasDecl(Storage, "getObject"));
    try testing.expect(@hasDecl(Storage, "deleteObject"));
    try testing.expect(@hasDecl(Storage, "objectExists"));
    try testing.expect(@hasDecl(Storage, "listObjects"));
    try testing.expect(@hasDecl(Storage, "stats"));
}

// ============================================================================
// Search Module Parity
// ============================================================================

test "search stub parity - types exist" {
    const Search = abi.search;

    try testing.expect(@hasDecl(Search, "SearchConfig"));
    try testing.expect(@hasDecl(Search, "SearchError"));
    try testing.expect(@hasDecl(Search, "SearchResult"));
    try testing.expect(@hasDecl(Search, "SearchIndex"));
    try testing.expect(@hasDecl(Search, "SearchStats"));
    try testing.expect(@hasDecl(Search, "Context"));
    try testing.expect(@hasDecl(Search, "init"));
    try testing.expect(@hasDecl(Search, "isEnabled"));
    try testing.expect(@hasDecl(Search, "createIndex"));
    try testing.expect(@hasDecl(Search, "deleteIndex"));
    try testing.expect(@hasDecl(Search, "indexDocument"));
    try testing.expect(@hasDecl(Search, "deleteDocument"));
    try testing.expect(@hasDecl(Search, "query"));
    try testing.expect(@hasDecl(Search, "stats"));
}

// ============================================================================
// Gateway Module Parity
// ============================================================================

test "gateway stub parity - types exist" {
    const Gateway = abi.gateway;

    try testing.expect(@hasDecl(Gateway, "GatewayConfig"));
    try testing.expect(@hasDecl(Gateway, "GatewayError"));
    try testing.expect(@hasDecl(Gateway, "GatewayStats"));
    try testing.expect(@hasDecl(Gateway, "HttpMethod"));
    try testing.expect(@hasDecl(Gateway, "Route"));
    try testing.expect(@hasDecl(Gateway, "MiddlewareType"));
    try testing.expect(@hasDecl(Gateway, "MatchResult"));
    try testing.expect(@hasDecl(Gateway, "RateLimitResult"));
    try testing.expect(@hasDecl(Gateway, "RateLimitConfig"));
    try testing.expect(@hasDecl(Gateway, "RateLimitAlgorithm"));
    try testing.expect(@hasDecl(Gateway, "CircuitBreakerConfig"));
    try testing.expect(@hasDecl(Gateway, "CircuitBreakerState"));
    try testing.expect(@hasDecl(Gateway, "Context"));
    try testing.expect(@hasDecl(Gateway, "init"));
    try testing.expect(@hasDecl(Gateway, "isEnabled"));
    try testing.expect(@hasDecl(Gateway, "addRoute"));
    try testing.expect(@hasDecl(Gateway, "removeRoute"));
    try testing.expect(@hasDecl(Gateway, "getRoutes"));
    try testing.expect(@hasDecl(Gateway, "matchRoute"));
    try testing.expect(@hasDecl(Gateway, "checkRateLimit"));
    try testing.expect(@hasDecl(Gateway, "recordUpstreamResult"));
    try testing.expect(@hasDecl(Gateway, "stats"));
    try testing.expect(@hasDecl(Gateway, "getCircuitState"));
    try testing.expect(@hasDecl(Gateway, "resetCircuit"));
}

// ============================================================================
// Pages Module Parity
// ============================================================================

test "pages stub parity - types exist" {
    const Pages = abi.pages;

    try testing.expect(@hasDecl(Pages, "PagesConfig"));
    try testing.expect(@hasDecl(Pages, "PagesError"));
    try testing.expect(@hasDecl(Pages, "HttpMethod"));
    try testing.expect(@hasDecl(Pages, "Page"));
    try testing.expect(@hasDecl(Pages, "PageContent"));
    try testing.expect(@hasDecl(Pages, "PageMatch"));
    try testing.expect(@hasDecl(Pages, "RenderResult"));
    try testing.expect(@hasDecl(Pages, "PagesStats"));
    try testing.expect(@hasDecl(Pages, "MetadataEntry"));
    try testing.expect(@hasDecl(Pages, "TemplateVar"));
    try testing.expect(@hasDecl(Pages, "TemplateRef"));
    try testing.expect(@hasDecl(Pages, "Context"));
    try testing.expect(@hasDecl(Pages, "init"));
    try testing.expect(@hasDecl(Pages, "isEnabled"));
    try testing.expect(@hasDecl(Pages, "isInitialized"));
    try testing.expect(@hasDecl(Pages, "addPage"));
    try testing.expect(@hasDecl(Pages, "removePage"));
    try testing.expect(@hasDecl(Pages, "getPage"));
    try testing.expect(@hasDecl(Pages, "matchPage"));
    try testing.expect(@hasDecl(Pages, "renderPage"));
    try testing.expect(@hasDecl(Pages, "listPages"));
    try testing.expect(@hasDecl(Pages, "stats"));
}

// ============================================================================
// Benchmarks Module Parity
// ============================================================================

test "benchmarks stub parity - types exist" {
    const Benchmarks = abi.benchmarks;

    try testing.expect(@hasDecl(Benchmarks, "Config"));
    try testing.expect(@hasDecl(Benchmarks, "BenchmarksError"));
    try testing.expect(@hasDecl(Benchmarks, "Context"));
    try testing.expect(@hasDecl(Benchmarks, "isEnabled"));
}

// ============================================================================
// Mobile Module Parity
// ============================================================================

test "mobile stub parity - types exist" {
    const Mobile = abi.mobile;

    try testing.expect(@hasDecl(Mobile, "MobileConfig"));
    try testing.expect(@hasDecl(Mobile, "MobilePlatform"));
    try testing.expect(@hasDecl(Mobile, "MobileError"));
    try testing.expect(@hasDecl(Mobile, "LifecycleState"));
    try testing.expect(@hasDecl(Mobile, "SensorData"));
    try testing.expect(@hasDecl(Mobile, "Context"));
    try testing.expect(@hasDecl(Mobile, "init"));
    try testing.expect(@hasDecl(Mobile, "isEnabled"));
    try testing.expect(@hasDecl(Mobile, "getLifecycleState"));
    try testing.expect(@hasDecl(Mobile, "readSensor"));
    try testing.expect(@hasDecl(Mobile, "sendNotification"));
}

// ============================================================================
// AI Core Module Parity
// ============================================================================

test "ai_core stub parity - types exist" {
    const AiCore = abi.ai_core;

    try testing.expect(@hasDecl(AiCore, "Context"));
    try testing.expect(@hasDecl(AiCore, "Error"));
    try testing.expect(@hasDecl(AiCore, "isEnabled"));
    try testing.expect(@hasDecl(AiCore, "Agent"));
    try testing.expect(@hasDecl(AiCore, "ToolRegistry"));
    try testing.expect(@hasDecl(AiCore, "PromptBuilder"));
    try testing.expect(@hasDecl(AiCore, "ModelRegistry"));
    try testing.expect(@hasDecl(AiCore, "createAgent"));
    try testing.expect(@hasDecl(AiCore, "createRegistry"));
}

// ============================================================================
// AI Inference Module Parity
// ============================================================================

test "ai_inference stub parity - types exist" {
    const Inference = abi.inference;

    try testing.expect(@hasDecl(Inference, "Context"));
    try testing.expect(@hasDecl(Inference, "Error"));
    try testing.expect(@hasDecl(Inference, "isEnabled"));
    try testing.expect(@hasDecl(Inference, "llm"));
    try testing.expect(@hasDecl(Inference, "embeddings"));
    try testing.expect(@hasDecl(Inference, "streaming"));
    try testing.expect(@hasDecl(Inference, "transformer"));
}

// ============================================================================
// AI Training Module Parity
// ============================================================================

test "ai_training stub parity - types exist" {
    const Training = abi.training;

    try testing.expect(@hasDecl(Training, "Context"));
    try testing.expect(@hasDecl(Training, "Error"));
    try testing.expect(@hasDecl(Training, "isEnabled"));
    try testing.expect(@hasDecl(Training, "TrainingConfig"));
    try testing.expect(@hasDecl(Training, "TrainableModel"));
    try testing.expect(@hasDecl(Training, "train"));
    try testing.expect(@hasDecl(Training, "trainWithResult"));
}

// ============================================================================
// AI Reasoning Module Parity
// ============================================================================

test "ai_reasoning stub parity - types exist" {
    const Reasoning = abi.reasoning;

    try testing.expect(@hasDecl(Reasoning, "Context"));
    try testing.expect(@hasDecl(Reasoning, "Error"));
    try testing.expect(@hasDecl(Reasoning, "isEnabled"));
    try testing.expect(@hasDecl(Reasoning, "abbey"));
    try testing.expect(@hasDecl(Reasoning, "explore"));
    try testing.expect(@hasDecl(Reasoning, "orchestration"));
    try testing.expect(@hasDecl(Reasoning, "documents"));
}

// ============================================================================
// Comprehensive Module Surface Test
// ============================================================================

test "all feature modules have consistent API surface" {
    // All feature modules should follow the Context + isEnabled pattern
    try verifyContextPattern(abi.database);
    try verifyContextPattern(abi.gpu);
    try verifyContextPattern(abi.network);
    try verifyContextPattern(abi.web);
    try verifyContextPattern(abi.observability);
    try verifyContextPattern(abi.analytics);
    try verifyContextPattern(abi.cloud);
    try verifyContextPattern(abi.auth);
    try verifyContextPattern(abi.messaging);
    try verifyContextPattern(abi.cache);
    try verifyContextPattern(abi.storage);
    try verifyContextPattern(abi.search);
    try verifyContextPattern(abi.mobile);
    try verifyContextPattern(abi.gateway);
    try verifyContextPattern(abi.pages);

    if (build_options.enable_ai) {
        try verifyContextPattern(abi.ai);
    }

    // Split AI modules
    try verifyContextPattern(abi.ai_core);
    try verifyContextPattern(abi.inference);
    try verifyContextPattern(abi.training);
    try verifyContextPattern(abi.reasoning);
}
