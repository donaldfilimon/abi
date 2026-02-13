//! ABI Framework C Bindings
//!
//! This module provides C-compatible FFI exports for the ABI framework.
//! All functions use C calling conventions and return C-compatible types.
//!
//! The bindings follow a handle-based pattern where opaque pointers are
//! returned to C code for managing framework resources.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// ============================================================================
// Error Handling
// ============================================================================

/// C-compatible error codes matching abi_errors.h
pub const ABI_OK: c_int = 0;
pub const ABI_ERROR_INIT_FAILED: c_int = -1;
pub const ABI_ERROR_ALREADY_INITIALIZED: c_int = -2;
pub const ABI_ERROR_NOT_INITIALIZED: c_int = -3;
pub const ABI_ERROR_OUT_OF_MEMORY: c_int = -4;
pub const ABI_ERROR_INVALID_ARGUMENT: c_int = -5;
pub const ABI_ERROR_FEATURE_DISABLED: c_int = -6;
pub const ABI_ERROR_TIMEOUT: c_int = -7;
pub const ABI_ERROR_IO: c_int = -8;
pub const ABI_ERROR_GPU_UNAVAILABLE: c_int = -9;
pub const ABI_ERROR_DATABASE_ERROR: c_int = -10;
pub const ABI_ERROR_NETWORK_ERROR: c_int = -11;
pub const ABI_ERROR_AI_ERROR: c_int = -12;
pub const ABI_ERROR_UNKNOWN: c_int = -99;

/// Get human-readable error message for an error code.
export fn abi_error_string(err: c_int) [*:0]const u8 {
    return switch (err) {
        ABI_OK => "Success",
        ABI_ERROR_INIT_FAILED => "Initialization failed",
        ABI_ERROR_ALREADY_INITIALIZED => "Already initialized",
        ABI_ERROR_NOT_INITIALIZED => "Not initialized",
        ABI_ERROR_OUT_OF_MEMORY => "Out of memory",
        ABI_ERROR_INVALID_ARGUMENT => "Invalid argument",
        ABI_ERROR_FEATURE_DISABLED => "Feature disabled at compile time",
        ABI_ERROR_TIMEOUT => "Operation timed out",
        ABI_ERROR_IO => "I/O error",
        ABI_ERROR_GPU_UNAVAILABLE => "GPU not available",
        ABI_ERROR_DATABASE_ERROR => "Database error",
        ABI_ERROR_NETWORK_ERROR => "Network error",
        ABI_ERROR_AI_ERROR => "AI operation error",
        else => "Unknown error",
    };
}

// ============================================================================
// Memory Management
// ============================================================================

/// Global allocator for C bindings (uses libc malloc/free for compatibility)
var c_allocator: std.mem.Allocator = std.heap.c_allocator;

/// Opaque framework handle
const FrameworkHandle = opaque {};

/// Framework wrapper that holds the actual Framework instance
const FrameworkWrapper = struct {
    framework: abi.Framework,
    allocator: std.mem.Allocator,
};

// ============================================================================
// Framework Lifecycle
// ============================================================================

/// Initialize the ABI framework with default options.
export fn abi_init(out_framework: *?*FrameworkHandle) c_int {
    return abi_init_with_options(null, out_framework);
}

/// Initialize the ABI framework with custom options.
export fn abi_init_with_options(
    options: ?*const Options,
    out_framework: *?*FrameworkHandle,
) c_int {
    out_framework.* = null;

    // Allocate wrapper
    const wrapper = c_allocator.create(FrameworkWrapper) catch {
        return ABI_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.destroy(wrapper);

    // Build config from options
    var config = abi.Config.defaults();
    if (options) |opts| {
        if (!opts.enable_ai) config.ai = null;
        if (!opts.enable_gpu) config.gpu = null;
        if (!opts.enable_database) config.database = null;
        if (!opts.enable_network) config.network = null;
        if (!opts.enable_web) config.web = null;
        if (!opts.enable_profiling) config.observability = null;
    }

    // Initialize framework
    wrapper.framework = abi.Framework.init(c_allocator, config) catch |err| {
        c_allocator.destroy(wrapper);
        return mapError(err);
    };
    wrapper.allocator = c_allocator;

    out_framework.* = @ptrCast(wrapper);
    return ABI_OK;
}

/// Shutdown the framework and release all resources.
export fn abi_shutdown(framework: ?*FrameworkHandle) void {
    if (framework) |fw| {
        const wrapper: *FrameworkWrapper = @ptrCast(@alignCast(fw));
        wrapper.framework.deinit();
        wrapper.allocator.destroy(wrapper);
    }
}

/// Get the version string.
export fn abi_version() [*:0]const u8 {
    return @ptrCast(abi.version().ptr);
}

/// Get detailed version information.
export fn abi_version_info(out_version: *VersionInfo) void {
    const ver = abi.version();
    // Parse version string "X.Y.Z"
    var iter = std.mem.splitScalar(u8, ver, '.');
    out_version.major = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;
    out_version.minor = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;
    out_version.patch = std.fmt.parseInt(c_int, iter.next() orelse "0", 10) catch 0;
    out_version.full = @ptrCast(ver.ptr);
}

/// Check if a feature is enabled.
export fn abi_is_feature_enabled(framework: ?*FrameworkHandle, feature: [*:0]const u8) bool {
    _ = framework;
    const feature_str = std.mem.sliceTo(feature, 0);

    if (std.mem.eql(u8, feature_str, "ai")) return build_options.enable_ai;
    if (std.mem.eql(u8, feature_str, "gpu")) return build_options.enable_gpu;
    if (std.mem.eql(u8, feature_str, "database")) return build_options.enable_database;
    if (std.mem.eql(u8, feature_str, "network")) return build_options.enable_network;
    if (std.mem.eql(u8, feature_str, "web")) return build_options.enable_web;
    if (std.mem.eql(u8, feature_str, "profiling")) return build_options.enable_profiling;

    return false;
}

/// Get the current framework state as a string.
export fn abi_get_state(framework: ?*FrameworkHandle) [*:0]const u8 {
    if (framework) |fw| {
        const wrapper: *FrameworkWrapper = @ptrCast(@alignCast(fw));
        return @tagName(wrapper.framework.state);
    }
    return "unknown";
}

/// Get the number of features that initialized successfully.
export fn abi_enabled_feature_count(framework: ?*FrameworkHandle) c_int {
    if (framework) |fw| {
        const wrapper: *FrameworkWrapper = @ptrCast(@alignCast(fw));
        var count: c_int = 0;
        if (wrapper.framework.gpu != null) count += 1;
        if (wrapper.framework.ai != null) count += 1;
        if (wrapper.framework.database != null) count += 1;
        if (wrapper.framework.network != null) count += 1;
        if (wrapper.framework.web != null) count += 1;
        if (wrapper.framework.observability != null) count += 1;
        if (wrapper.framework.analytics != null) count += 1;
        if (wrapper.framework.cloud != null) count += 1;
        return count;
    }
    return 0;
}

// ============================================================================
// SIMD Operations
// ============================================================================

/// SIMD capability flags
const SimdCaps = extern struct {
    sse: bool = false,
    sse2: bool = false,
    sse3: bool = false,
    ssse3: bool = false,
    sse4_1: bool = false,
    sse4_2: bool = false,
    avx: bool = false,
    avx2: bool = false,
    avx512f: bool = false,
    neon: bool = false,
};

/// Query CPU SIMD capabilities.
export fn abi_simd_get_caps(out_caps: *SimdCaps) void {
    const caps = abi.simd.getSimdCapabilities();
    // Map from internal capabilities to C struct
    const has_simd = caps.has_simd;
    const is_x86 = caps.arch == .x86_64;
    const is_arm = caps.arch == .aarch64;

    out_caps.* = .{
        // On x86_64, basic SSE is generally available
        .sse = if (is_x86) has_simd else false,
        .sse2 = if (is_x86) has_simd else false,
        .sse3 = if (is_x86) has_simd else false,
        .ssse3 = if (is_x86) has_simd else false,
        .sse4_1 = if (is_x86) has_simd else false,
        .sse4_2 = if (is_x86) has_simd else false,
        .avx = if (is_x86) (caps.vector_size >= 8) else false,
        .avx2 = if (is_x86) (caps.vector_size >= 8) else false,
        .avx512f = if (is_x86) (caps.vector_size >= 16) else false,
        .neon = is_arm,
    };
}

/// Check if any SIMD instruction set is available.
export fn abi_simd_available() bool {
    return abi.simd.hasSimdSupport();
}

/// Vector element-wise addition: result[i] = a[i] + b[i]
export fn abi_simd_vector_add(
    a: [*]const f32,
    b: [*]const f32,
    result: [*]f32,
    len: usize,
) void {
    const a_slice = a[0..len];
    const b_slice = b[0..len];
    const result_slice = result[0..len];
    abi.simd.vectorAdd(a_slice, b_slice, result_slice);
}

/// Vector dot product: sum(a[i] * b[i])
export fn abi_simd_vector_dot(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return abi.simd.vectorDot(a[0..len], b[0..len]);
}

/// Vector L2 norm: sqrt(sum(v[i]^2))
export fn abi_simd_vector_l2_norm(v: [*]const f32, len: usize) f32 {
    return abi.simd.vectorL2Norm(v[0..len]);
}

/// Cosine similarity between two vectors.
export fn abi_simd_cosine_similarity(a: [*]const f32, b: [*]const f32, len: usize) f32 {
    return abi.simd.cosineSimilarity(a[0..len], b[0..len]);
}

// ============================================================================
// Database Operations
// ============================================================================

/// Opaque database handle
const DatabaseHandle = opaque {};

/// Database configuration
const DatabaseConfig = extern struct {
    name: [*:0]const u8 = "default",
    dimension: usize = 384,
    initial_capacity: usize = 1000,
};

/// Create a new vector database.
export fn abi_database_create(
    config: ?*const DatabaseConfig,
    out_db: *?*DatabaseHandle,
) c_int {
    if (!build_options.enable_database) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    out_db.* = null;

    const cfg = config orelse &DatabaseConfig{};
    const name = std.mem.sliceTo(cfg.name, 0);

    // Create database wrapper
    const db = c_allocator.create(DatabaseWrapper) catch {
        return ABI_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.destroy(db);

    // Initialize database using the formats.VectorDatabase API
    db.* = .{
        .handle = abi.database.formats.VectorDatabase.init(c_allocator, name, cfg.dimension),
        .allocator = c_allocator,
    };

    out_db.* = @ptrCast(db);
    return ABI_OK;
}

/// Close a database and release resources.
export fn abi_database_close(db: ?*DatabaseHandle) void {
    if (db) |d| {
        const wrapper: *DatabaseWrapper = @ptrCast(@alignCast(d));
        wrapper.handle.deinit();
        wrapper.allocator.destroy(wrapper);
    }
}

/// Insert a vector into the database.
export fn abi_database_insert(
    db: ?*DatabaseHandle,
    id: u64,
    vector: [*]const f32,
    vector_len: usize,
    metadata: ?[*:0]const u8,
) c_int {
    if (!build_options.enable_database) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const d = db orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *DatabaseWrapper = @ptrCast(@alignCast(d));

    const meta_slice: ?[]const u8 = if (metadata) |m| std.mem.sliceTo(m, 0) else null;
    wrapper.handle.insert(id, vector[0..vector_len], meta_slice) catch |err| {
        return mapError(err);
    };

    return ABI_OK;
}

/// Search for similar vectors.
export fn abi_database_search(
    db: ?*DatabaseHandle,
    query: [*]const f32,
    query_len: usize,
    k: usize,
    out_results: [*]SearchResult,
    out_count: *usize,
) c_int {
    if (!build_options.enable_database) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const d = db orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *DatabaseWrapper = @ptrCast(@alignCast(d));

    const results = wrapper.handle.search(query[0..query_len], k) catch |err| {
        out_count.* = 0;
        return mapError(err);
    };
    defer c_allocator.free(results);

    const count = @min(results.len, k);
    for (0..count) |i| {
        out_results[i] = .{
            .id = results[i].id,
            .score = results[i].score,
        };
    }
    out_count.* = count;

    return ABI_OK;
}

/// Delete a vector from the database.
export fn abi_database_delete(db: ?*DatabaseHandle, id: u64) c_int {
    if (!build_options.enable_database) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const d = db orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *DatabaseWrapper = @ptrCast(@alignCast(d));

    const deleted = wrapper.handle.delete(id);
    if (!deleted) {
        return ABI_ERROR_INVALID_ARGUMENT; // Vector not found
    }

    return ABI_OK;
}

/// Get the number of vectors in the database.
export fn abi_database_count(db: ?*DatabaseHandle, out_count: *usize) c_int {
    if (!build_options.enable_database) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const d = db orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *DatabaseWrapper = @ptrCast(@alignCast(d));

    out_count.* = wrapper.handle.vectors.items.len;
    return ABI_OK;
}

// ============================================================================
// GPU Operations
// ============================================================================

/// Opaque GPU handle
const GpuHandle = opaque {};

/// GPU configuration
const GpuConfig = extern struct {
    backend: c_int = 0, // 0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu
    device_index: c_int = 0,
    enable_profiling: bool = false,
};

/// Initialize a GPU context.
export fn abi_gpu_init(config: ?*const GpuConfig, out_gpu: *?*GpuHandle) c_int {
    if (!build_options.enable_gpu) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    out_gpu.* = null;

    const cfg = config orelse &GpuConfig{};

    // Map backend enum (0=auto/vulkan, 1=cuda, 2=vulkan, 3=metal, 4=webgpu)
    const backend: ?abi.gpu.Backend = switch (cfg.backend) {
        1 => .cuda,
        2 => .vulkan,
        3 => .metal,
        4 => .webgpu,
        else => null, // auto-detect
    };

    // Create GPU wrapper
    const gpu = c_allocator.create(GpuWrapper) catch {
        return ABI_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.destroy(gpu);

    gpu.* = .{
        .handle = abi.gpu.Gpu.init(c_allocator, .{
            .preferred_backend = backend,
            .enable_profiling = cfg.enable_profiling,
        }) catch |err| {
            c_allocator.destroy(gpu);
            return mapError(err);
        },
        .allocator = c_allocator,
    };

    out_gpu.* = @ptrCast(gpu);
    return ABI_OK;
}

/// Shutdown GPU context and release resources.
export fn abi_gpu_shutdown(gpu: ?*GpuHandle) void {
    if (gpu) |g| {
        const wrapper: *GpuWrapper = @ptrCast(@alignCast(g));
        wrapper.handle.deinit();
        wrapper.allocator.destroy(wrapper);
    }
}

/// Check if any GPU backend is available.
export fn abi_gpu_is_available() bool {
    if (!build_options.enable_gpu) {
        return false;
    }
    // Check available backends
    const backends = abi.gpu.availableBackends(c_allocator) catch return false;
    defer c_allocator.free(backends);
    return backends.len > 0;
}

/// Get the active GPU backend name.
export fn abi_gpu_backend_name(gpu: ?*GpuHandle) [*:0]const u8 {
    if (!build_options.enable_gpu) {
        return "disabled";
    }
    if (gpu) |g| {
        const wrapper: *GpuWrapper = @ptrCast(@alignCast(g));
        if (wrapper.handle.getBackend()) |backend| {
            // Convert the slice to a null-terminated string
            return switch (backend) {
                .cuda => "cuda",
                .vulkan => "vulkan",
                .metal => "metal",
                .webgpu => "webgpu",
                .stdgpu => "stdgpu",
                .opengl => "opengl",
                .opengles => "opengles",
                .webgl2 => "webgl2",
                .fpga => "fpga",
            };
        }
    }
    return "none";
}

// ============================================================================
// Agent Operations
// ============================================================================

/// Opaque agent handle
const AgentHandle = opaque {};

/// Agent backend type (matches AgentBackend enum)
pub const ABI_AGENT_BACKEND_ECHO: c_int = 0;
pub const ABI_AGENT_BACKEND_OPENAI: c_int = 1;
pub const ABI_AGENT_BACKEND_OLLAMA: c_int = 2;
pub const ABI_AGENT_BACKEND_HUGGINGFACE: c_int = 3;
pub const ABI_AGENT_BACKEND_LOCAL: c_int = 4;

/// Agent status codes
pub const ABI_AGENT_STATUS_READY: c_int = 0;
pub const ABI_AGENT_STATUS_BUSY: c_int = 1;
pub const ABI_AGENT_STATUS_ERROR: c_int = 2;

/// Agent configuration for C API
const AgentConfig = extern struct {
    /// Agent name (required, null-terminated)
    name: [*:0]const u8 = "agent",
    /// Backend type (ABI_AGENT_BACKEND_*)
    backend: c_int = ABI_AGENT_BACKEND_ECHO,
    /// Model name (e.g., "gpt-4", "llama3.2")
    model: [*:0]const u8 = "gpt-4",
    /// System prompt (optional, null for no system prompt)
    system_prompt: ?[*:0]const u8 = null,
    /// Temperature for generation (0.0 - 2.0)
    temperature: f32 = 0.7,
    /// Top-p for generation (0.0 - 1.0)
    top_p: f32 = 0.9,
    /// Maximum tokens for generation
    max_tokens: u32 = 1024,
    /// Enable conversation history
    enable_history: bool = true,
};

/// Agent response structure
const AgentResponse = extern struct {
    /// Response text (null-terminated, owned by agent - valid until next send or destroy)
    text: ?[*:0]const u8 = null,
    /// Length of response text (excluding null terminator)
    length: usize = 0,
    /// Number of tokens used (if available from backend)
    tokens_used: u64 = 0,
};

/// Agent statistics
const AgentStats = extern struct {
    /// Total messages in history
    history_length: usize = 0,
    /// Number of user messages
    user_messages: usize = 0,
    /// Number of assistant messages
    assistant_messages: usize = 0,
    /// Total characters in history
    total_characters: usize = 0,
    /// Total tokens used in session
    total_tokens_used: u64 = 0,
};

/// Create a new AI agent.
export fn abi_agent_create(
    config: ?*const AgentConfig,
    out_agent: *?*AgentHandle,
) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    out_agent.* = null;

    const cfg = config orelse &AgentConfig{};

    // Map C backend enum to Zig backend type
    const backend: abi.ai.agent.AgentBackend = switch (cfg.backend) {
        ABI_AGENT_BACKEND_OPENAI => .openai,
        ABI_AGENT_BACKEND_OLLAMA => .ollama,
        ABI_AGENT_BACKEND_HUGGINGFACE => .huggingface,
        ABI_AGENT_BACKEND_LOCAL => .local,
        else => .echo,
    };

    // Convert C strings to Zig slices
    const name_slice = std.mem.sliceTo(cfg.name, 0);
    const model_slice = std.mem.sliceTo(cfg.model, 0);
    const system_prompt_slice: ?[]const u8 = if (cfg.system_prompt) |sp|
        std.mem.sliceTo(sp, 0)
    else
        null;

    // Create agent wrapper
    const wrapper = c_allocator.create(AgentWrapper) catch {
        return ABI_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.destroy(wrapper);

    // Create the agent
    const agent_ptr = c_allocator.create(abi.ai.Agent) catch {
        c_allocator.destroy(wrapper);
        return ABI_ERROR_OUT_OF_MEMORY;
    };
    errdefer c_allocator.destroy(agent_ptr);

    agent_ptr.* = abi.ai.Agent.init(c_allocator, .{
        .name = name_slice,
        .backend = backend,
        .model = model_slice,
        .system_prompt = system_prompt_slice,
        .temperature = cfg.temperature,
        .top_p = cfg.top_p,
        .max_tokens = cfg.max_tokens,
        .enable_history = cfg.enable_history,
    }) catch |err| {
        c_allocator.destroy(agent_ptr);
        c_allocator.destroy(wrapper);
        return mapAgentError(err);
    };

    wrapper.* = .{
        .handle = agent_ptr,
        .allocator = c_allocator,
        .last_response = null,
    };

    out_agent.* = @ptrCast(wrapper);
    return ABI_OK;
}

/// Destroy an agent and release all resources.
export fn abi_agent_destroy(agent: ?*AgentHandle) void {
    if (!build_options.enable_ai) {
        return;
    }

    if (agent) |a| {
        const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

        // Free last response if any
        if (wrapper.last_response) |resp| {
            wrapper.allocator.free(resp);
        }

        // Deinit and destroy the agent
        wrapper.handle.deinit();
        wrapper.allocator.destroy(wrapper.handle);
        wrapper.allocator.destroy(wrapper);
    }
}

/// Send a message to the agent and get a response.
/// The response text is owned by the agent and valid until the next send or destroy.
export fn abi_agent_send(
    agent: ?*AgentHandle,
    message: [*:0]const u8,
    out_response: *AgentResponse,
) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const a = agent orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    // Free previous response if any
    if (wrapper.last_response) |resp| {
        wrapper.allocator.free(resp);
        wrapper.last_response = null;
    }

    // Convert message to slice
    const message_slice = std.mem.sliceTo(message, 0);

    // Process the message
    const response = wrapper.handle.process(message_slice, wrapper.allocator) catch |err| {
        out_response.* = .{};
        return mapAgentError(err);
    };

    // Store response for lifetime management
    wrapper.last_response = response;

    // Get token usage
    const stats = wrapper.handle.getStats();

    out_response.* = .{
        .text = @ptrCast(response.ptr),
        .length = response.len,
        .tokens_used = stats.total_tokens_used,
    };

    return ABI_OK;
}

/// Get the current status of the agent.
export fn abi_agent_get_status(agent: ?*AgentHandle) c_int {
    if (!build_options.enable_ai) {
        return ABI_AGENT_STATUS_ERROR;
    }

    const a = agent orelse return ABI_AGENT_STATUS_ERROR;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    // Agent is ready if it exists and has a valid handle
    _ = wrapper.handle;
    return ABI_AGENT_STATUS_READY;
}

/// Get agent statistics.
export fn abi_agent_get_stats(agent: ?*AgentHandle, out_stats: *AgentStats) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const a = agent orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    const stats = wrapper.handle.getStats();
    out_stats.* = .{
        .history_length = stats.history_length,
        .user_messages = stats.user_messages,
        .assistant_messages = stats.assistant_messages,
        .total_characters = stats.total_characters,
        .total_tokens_used = stats.total_tokens_used,
    };

    return ABI_OK;
}

/// Clear the agent's conversation history.
export fn abi_agent_clear_history(agent: ?*AgentHandle) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const a = agent orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    wrapper.handle.clearHistory();
    return ABI_OK;
}

/// Set the agent's temperature parameter.
export fn abi_agent_set_temperature(agent: ?*AgentHandle, temperature: f32) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const a = agent orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    wrapper.handle.setTemperature(temperature) catch {
        return ABI_ERROR_INVALID_ARGUMENT;
    };

    return ABI_OK;
}

/// Set the agent's max tokens parameter.
export fn abi_agent_set_max_tokens(agent: ?*AgentHandle, max_tokens: u32) c_int {
    if (!build_options.enable_ai) {
        return ABI_ERROR_FEATURE_DISABLED;
    }

    const a = agent orelse return ABI_ERROR_NOT_INITIALIZED;
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    wrapper.handle.setMaxTokens(max_tokens) catch {
        return ABI_ERROR_INVALID_ARGUMENT;
    };

    return ABI_OK;
}

/// Get the agent's name.
export fn abi_agent_get_name(agent: ?*AgentHandle) [*:0]const u8 {
    if (!build_options.enable_ai) {
        return "disabled";
    }

    const a = agent orelse return "unknown";
    const wrapper: *AgentWrapper = @ptrCast(@alignCast(a));

    const name = wrapper.handle.name();
    // Note: This assumes the name is stored as a null-terminated string internally
    // or the slice remains valid for the agent's lifetime
    return @ptrCast(name.ptr);
}

// ============================================================================
// Memory Management Exports
// ============================================================================

/// Free a string allocated by ABI functions.
export fn abi_free_string(str: ?[*]u8) void {
    if (str) |s| {
        // Find null terminator to determine length
        var len: usize = 0;
        while (s[len] != 0) : (len += 1) {}
        c_allocator.free(s[0 .. len + 1]);
    }
}

/// Free a search results array.
export fn abi_free_results(results: ?[*]SearchResult, count: usize) void {
    if (results) |r| {
        c_allocator.free(r[0..count]);
    }
}

// ============================================================================
// Type Definitions
// ============================================================================

/// Framework initialization options.
const Options = extern struct {
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    enable_network: bool = true,
    enable_web: bool = true,
    enable_profiling: bool = true,
};

/// Version information.
const VersionInfo = extern struct {
    major: c_int = 0,
    minor: c_int = 0,
    patch: c_int = 0,
    full: [*:0]const u8 = "",
};

/// Vector search result.
const SearchResult = extern struct {
    id: u64,
    score: f32,
};

// ============================================================================
// Internal Wrapper Types
// ============================================================================

/// Database wrapper for opaque handle
const DatabaseWrapper = struct {
    handle: if (build_options.enable_database) abi.database.formats.VectorDatabase else void,
    allocator: std.mem.Allocator,
};

/// GPU wrapper for opaque handle
const GpuWrapper = struct {
    handle: if (build_options.enable_gpu) abi.gpu.Gpu else void,
    allocator: std.mem.Allocator,
};

/// Agent wrapper for opaque handle
const AgentWrapper = struct {
    handle: if (build_options.enable_ai) *abi.ai.Agent else void,
    allocator: std.mem.Allocator,
    /// Store the last response for C string lifetime management
    last_response: ?[]u8 = null,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Map Zig errors to C error codes.
fn mapError(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => ABI_ERROR_OUT_OF_MEMORY,
        error.Timeout => ABI_ERROR_TIMEOUT,
        error.IoError, error.AccessDenied, error.BrokenPipe => ABI_ERROR_IO,
        error.FeatureDisabled, error.FeatureNotEnabled => ABI_ERROR_FEATURE_DISABLED,
        error.InvalidArgument, error.InvalidConfiguration => ABI_ERROR_INVALID_ARGUMENT,
        else => ABI_ERROR_UNKNOWN,
    };
}

/// Map Agent-specific errors to C error codes.
fn mapAgentError(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => ABI_ERROR_OUT_OF_MEMORY,
        error.InvalidConfiguration => ABI_ERROR_INVALID_ARGUMENT,
        error.ConnectorNotAvailable => ABI_ERROR_AI_ERROR,
        error.GenerationFailed => ABI_ERROR_AI_ERROR,
        error.ApiKeyMissing => ABI_ERROR_AI_ERROR,
        error.HttpRequestFailed => ABI_ERROR_NETWORK_ERROR,
        error.InvalidApiResponse => ABI_ERROR_AI_ERROR,
        error.RateLimitExceeded => ABI_ERROR_AI_ERROR,
        error.Timeout => ABI_ERROR_TIMEOUT,
        error.ConnectionRefused => ABI_ERROR_NETWORK_ERROR,
        error.ModelNotFound => ABI_ERROR_AI_ERROR,
        else => ABI_ERROR_UNKNOWN,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "error string lookup" {
    const ok_str = abi_error_string(ABI_OK);
    try std.testing.expectEqualStrings("Success", std.mem.sliceTo(ok_str, 0));

    const oom_str = abi_error_string(ABI_ERROR_OUT_OF_MEMORY);
    try std.testing.expectEqualStrings("Out of memory", std.mem.sliceTo(oom_str, 0));
}

test "version info" {
    var info: VersionInfo = undefined;
    abi_version_info(&info);
    try std.testing.expect(info.major >= 0);
    try std.testing.expect(info.minor >= 0);
}

test "simd available check" {
    // Should not crash
    _ = abi_simd_available();
}

test "simd caps query" {
    var caps: SimdCaps = undefined;
    abi_simd_get_caps(&caps);
    // Should not crash
}

test "feature check" {
    const ai_enabled = abi_is_feature_enabled(null, "ai");
    try std.testing.expectEqual(build_options.enable_ai, ai_enabled);
}

test "agent create and destroy" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var agent: ?*AgentHandle = null;
    const config = AgentConfig{
        .name = "test-agent",
        .backend = ABI_AGENT_BACKEND_ECHO,
        .model = "test-model",
        .temperature = 0.5,
    };

    const result = abi_agent_create(&config, &agent);
    try std.testing.expectEqual(ABI_OK, result);
    try std.testing.expect(agent != null);

    // Verify status is ready
    const status = abi_agent_get_status(agent);
    try std.testing.expectEqual(ABI_AGENT_STATUS_READY, status);

    // Clean up
    abi_agent_destroy(agent);
}

test "agent send message" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var agent: ?*AgentHandle = null;
    const config = AgentConfig{
        .name = "echo-agent",
        .backend = ABI_AGENT_BACKEND_ECHO,
    };

    const create_result = abi_agent_create(&config, &agent);
    try std.testing.expectEqual(ABI_OK, create_result);
    defer abi_agent_destroy(agent);

    var response: AgentResponse = .{};
    const send_result = abi_agent_send(agent, "Hello, agent!", &response);
    try std.testing.expectEqual(ABI_OK, send_result);
    try std.testing.expect(response.text != null);
    try std.testing.expect(response.length > 0);

    // Echo backend should return "Echo: <message>"
    const response_str = std.mem.sliceTo(response.text.?, 0);
    try std.testing.expect(std.mem.indexOf(u8, response_str, "Echo:") != null);
}

test "agent stats" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var agent: ?*AgentHandle = null;
    const create_result = abi_agent_create(null, &agent);
    try std.testing.expectEqual(ABI_OK, create_result);
    defer abi_agent_destroy(agent);

    // Send a message to populate history
    var response: AgentResponse = .{};
    _ = abi_agent_send(agent, "test", &response);

    var stats: AgentStats = .{};
    const stats_result = abi_agent_get_stats(agent, &stats);
    try std.testing.expectEqual(ABI_OK, stats_result);
    try std.testing.expect(stats.history_length >= 2); // user + assistant
    try std.testing.expect(stats.user_messages >= 1);
    try std.testing.expect(stats.assistant_messages >= 1);
}

test "agent clear history" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var agent: ?*AgentHandle = null;
    const create_result = abi_agent_create(null, &agent);
    try std.testing.expectEqual(ABI_OK, create_result);
    defer abi_agent_destroy(agent);

    // Send a message
    var response: AgentResponse = .{};
    _ = abi_agent_send(agent, "test", &response);

    // Clear history
    const clear_result = abi_agent_clear_history(agent);
    try std.testing.expectEqual(ABI_OK, clear_result);

    // Verify history is empty
    var stats: AgentStats = .{};
    _ = abi_agent_get_stats(agent, &stats);
    try std.testing.expectEqual(@as(usize, 0), stats.history_length);
}

test "agent set parameters" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var agent: ?*AgentHandle = null;
    const create_result = abi_agent_create(null, &agent);
    try std.testing.expectEqual(ABI_OK, create_result);
    defer abi_agent_destroy(agent);

    // Valid temperature
    const temp_result = abi_agent_set_temperature(agent, 0.8);
    try std.testing.expectEqual(ABI_OK, temp_result);

    // Invalid temperature
    const invalid_temp_result = abi_agent_set_temperature(agent, 3.0);
    try std.testing.expectEqual(ABI_ERROR_INVALID_ARGUMENT, invalid_temp_result);

    // Valid max tokens
    const tokens_result = abi_agent_set_max_tokens(agent, 2048);
    try std.testing.expectEqual(ABI_OK, tokens_result);

    // Invalid max tokens
    const invalid_tokens_result = abi_agent_set_max_tokens(agent, 0);
    try std.testing.expectEqual(ABI_ERROR_INVALID_ARGUMENT, invalid_tokens_result);
}

test "agent null handle returns error" {
    if (!build_options.enable_ai) {
        return error.SkipZigTest;
    }

    var response: AgentResponse = .{};
    const send_result = abi_agent_send(null, "test", &response);
    try std.testing.expectEqual(ABI_ERROR_NOT_INITIALIZED, send_result);

    var stats: AgentStats = .{};
    const stats_result = abi_agent_get_stats(null, &stats);
    try std.testing.expectEqual(ABI_ERROR_NOT_INITIALIZED, stats_result);

    const status = abi_agent_get_status(null);
    try std.testing.expectEqual(ABI_AGENT_STATUS_ERROR, status);
}
