pub const required = [_][]const u8{
    // Core types
    "Context",
    "Gpu",
    "GpuConfig",
    "GpuError",
    "Backend",

    // Buffer types
    "Buffer",
    "UnifiedBuffer",
    "BufferOptions",
    "BufferFlags",

    // Device types
    "Device",
    "DeviceType",

    // Stream types
    "Stream",
    "StreamOptions",
    "Event",
    "EventOptions",

    // Execution types
    "ExecutionResult",
    "LaunchConfig",
    "HealthStatus",

    // DSL types
    "KernelBuilder",

    // Namespaced API surface
    "backends",
    "dispatch",
    "devices",
    "runtime",
    "policy",
    "multi",
    "factory",

    // Module functions
    "init",
    "deinit",
    "isEnabled",
    "isInitialized",
};
