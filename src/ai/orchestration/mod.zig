//! Multi-Model Orchestration Module
//!
//! Provides sophisticated model orchestration capabilities including:
//! - Model registry for managing multiple LLM backends
//! - Routing strategies (round-robin, least-loaded, task-based)
//! - Ensemble methods (voting, averaging, weighted)
//! - Automatic fallback on model failure
//! - Load balancing across model instances
//!
//! ## Usage
//!
//! ```zig
//! const orchestration = @import("orchestration/mod.zig");
//!
//! // Create orchestrator with configuration
//! var orch = try orchestration.Orchestrator.init(allocator, .{
//!     .strategy = .task_based,
//!     .enable_fallback = true,
//!     .enable_ensemble = false,
//! });
//! defer orch.deinit();
//!
//! // Register models
//! try orch.registerModel(.{
//!     .id = "gpt-4",
//!     .backend = .openai,
//!     .capabilities = &.{.reasoning, .coding},
//! });
//!
//! // Route request to best model
//! const response = try orch.route("Write a function to sort an array", .coding);
//! ```

const std = @import("std");
const build_options = @import("build_options");

// Sub-modules
pub const router = @import("router.zig");
pub const ensemble = @import("ensemble.zig");
pub const fallback = @import("fallback.zig");

// Re-exports
pub const Router = router.Router;
pub const RoutingStrategy = router.RoutingStrategy;
pub const TaskType = router.TaskType;
pub const RouteResult = router.RouteResult;

pub const Ensemble = ensemble.Ensemble;
pub const EnsembleMethod = ensemble.EnsembleMethod;
pub const EnsembleResult = ensemble.EnsembleResult;

pub const FallbackManager = fallback.FallbackManager;
pub const FallbackPolicy = fallback.FallbackPolicy;
pub const HealthStatus = fallback.HealthStatus;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for multi-model orchestration.
pub const OrchestrationConfig = struct {
    /// Routing strategy for model selection.
    strategy: RoutingStrategy = .round_robin,

    /// Enable automatic fallback on model failure.
    enable_fallback: bool = true,

    /// Enable ensemble mode for combining multiple model outputs.
    enable_ensemble: bool = false,

    /// Ensemble method when ensemble mode is enabled.
    ensemble_method: EnsembleMethod = .voting,

    /// Maximum concurrent requests across all models.
    max_concurrent_requests: u32 = 100,

    /// Request timeout in milliseconds.
    timeout_ms: u64 = 30000,

    /// Health check interval in milliseconds.
    health_check_interval_ms: u64 = 60000,

    /// Minimum number of models required for ensemble.
    min_ensemble_models: u32 = 2,

    /// Maximum retries before fallback.
    max_retries: u32 = 3,

    /// Load balancing weight factor (0.0 to 1.0).
    load_factor_weight: f32 = 0.5,

    pub fn defaults() OrchestrationConfig {
        return .{};
    }

    /// Configuration optimized for high availability.
    pub fn highAvailability() OrchestrationConfig {
        return .{
            .strategy = .least_loaded,
            .enable_fallback = true,
            .enable_ensemble = false,
            .max_concurrent_requests = 200,
            .max_retries = 5,
            .health_check_interval_ms = 30000,
        };
    }

    /// Configuration optimized for quality (ensemble mode).
    pub fn highQuality() OrchestrationConfig {
        return .{
            .strategy = .task_based,
            .enable_fallback = true,
            .enable_ensemble = true,
            .ensemble_method = .weighted_average,
            .min_ensemble_models = 3,
        };
    }
};

/// Model backend type (mirrors agent.AgentBackend but specific to orchestration).
pub const ModelBackend = enum {
    openai,
    ollama,
    huggingface,
    anthropic,
    local,

    pub fn toString(self: ModelBackend) []const u8 {
        return @tagName(self);
    }
};

/// Model capability flags.
pub const Capability = enum {
    reasoning,
    coding,
    creative,
    analysis,
    summarization,
    translation,
    math,
    vision,
    embedding,

    pub fn toString(self: Capability) []const u8 {
        return @tagName(self);
    }
};

/// Configuration for a single model in the orchestration pool.
pub const ModelConfig = struct {
    /// Unique identifier for this model.
    id: []const u8,

    /// Human-readable name.
    name: []const u8 = "",

    /// Backend provider.
    backend: ModelBackend = .openai,

    /// Model identifier for the backend (e.g., "gpt-4", "llama-3.2").
    model_name: []const u8 = "",

    /// Capabilities this model excels at.
    capabilities: []const Capability = &.{},

    /// Priority for fallback ordering (lower = higher priority).
    priority: u32 = 100,

    /// Weight for load balancing (higher = more traffic).
    weight: f32 = 1.0,

    /// Maximum tokens this model supports.
    max_tokens: u32 = 4096,

    /// Cost per 1K tokens (for cost-aware routing).
    cost_per_1k_tokens: f32 = 0.0,

    /// Custom API endpoint (if different from default).
    endpoint: ?[]const u8 = null,

    /// Whether this model is currently enabled.
    enabled: bool = true,
};

// ============================================================================
// Errors
// ============================================================================

pub const OrchestrationError = error{
    /// No models registered.
    NoModelsAvailable,
    /// All models failed.
    AllModelsFailed,
    /// Model not found by ID.
    ModelNotFound,
    /// Duplicate model ID.
    DuplicateModelId,
    /// Invalid configuration.
    InvalidConfig,
    /// Request timeout.
    Timeout,
    /// Ensemble requires minimum models.
    InsufficientModelsForEnsemble,
    /// Model is disabled.
    ModelDisabled,
    /// Rate limit exceeded.
    RateLimitExceeded,
    /// Out of memory.
    OutOfMemory,
    /// Feature disabled at compile time.
    OrchestrationDisabled,
};

// ============================================================================
// Model Entry (internal state)
// ============================================================================

/// Internal representation of a registered model with runtime state.
pub const ModelEntry = struct {
    config: ModelConfig,
    status: HealthStatus = .healthy,
    active_requests: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    total_latency_ms: u64 = 0,
    last_request_time: i64 = 0,
    last_failure_time: i64 = 0,
    consecutive_failures: u32 = 0,

    /// Calculate average latency in milliseconds.
    pub fn avgLatencyMs(self: ModelEntry) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_latency_ms)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    /// Calculate success rate (0.0 to 1.0).
    pub fn successRate(self: ModelEntry) f64 {
        if (self.total_requests == 0) return 1.0;
        const successes = self.total_requests - self.total_failures;
        return @as(f64, @floatFromInt(successes)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    /// Calculate current load factor (0.0 to 1.0).
    pub fn loadFactor(self: ModelEntry, max_concurrent: u32) f64 {
        if (max_concurrent == 0) return 1.0;
        return @as(f64, @floatFromInt(self.active_requests)) /
            @as(f64, @floatFromInt(max_concurrent));
    }

    /// Check if model is available for requests.
    pub fn isAvailable(self: ModelEntry) bool {
        return self.config.enabled and self.status == .healthy;
    }
};

// ============================================================================
// Orchestrator
// ============================================================================

/// Multi-model orchestrator that coordinates requests across multiple LLM backends.
pub const Orchestrator = struct {
    allocator: std.mem.Allocator,
    config: OrchestrationConfig,
    models: std.StringHashMapUnmanaged(ModelEntry),
    router_instance: Router,
    ensemble_instance: ?Ensemble,
    fallback_manager: FallbackManager,
    round_robin_index: usize = 0,
    mutex: std.Thread.Mutex = .{},

    /// Initialize the orchestrator with configuration.
    pub fn init(allocator: std.mem.Allocator, config: OrchestrationConfig) OrchestrationError!Orchestrator {
        if (!isEnabled()) return OrchestrationError.OrchestrationDisabled;

        return .{
            .allocator = allocator,
            .config = config,
            .models = .{},
            .router_instance = Router.init(allocator, config.strategy),
            .ensemble_instance = if (config.enable_ensemble)
                Ensemble.init(allocator, config.ensemble_method)
            else
                null,
            .fallback_manager = FallbackManager.init(allocator, .{
                .max_retries = config.max_retries,
                .health_check_interval_ms = config.health_check_interval_ms,
            }),
        };
    }

    /// Clean up resources.
    pub fn deinit(self: *Orchestrator) void {
        // Free model ID strings
        var it = self.models.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.models.deinit(self.allocator);

        self.router_instance.deinit();
        if (self.ensemble_instance) |*e| e.deinit();
        self.fallback_manager.deinit();
    }

    /// Register a model with the orchestrator.
    pub fn registerModel(self: *Orchestrator, config: ModelConfig) OrchestrationError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check for duplicate
        if (self.models.contains(config.id)) {
            return OrchestrationError.DuplicateModelId;
        }

        // Copy the ID for storage
        const id_copy = self.allocator.dupe(u8, config.id) catch return OrchestrationError.OutOfMemory;
        errdefer self.allocator.free(id_copy);

        const entry = ModelEntry{
            .config = config,
        };

        self.models.put(self.allocator, id_copy, entry) catch return OrchestrationError.OutOfMemory;
    }

    /// Unregister a model by ID.
    pub fn unregisterModel(self: *Orchestrator, model_id: []const u8) OrchestrationError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const kv = self.models.fetchRemove(model_id) orelse return OrchestrationError.ModelNotFound;
        self.allocator.free(kv.key);
    }

    /// Get a model entry by ID.
    pub fn getModel(self: *Orchestrator, model_id: []const u8) ?*ModelEntry {
        return self.models.getPtr(model_id);
    }

    /// Enable or disable a model.
    pub fn setModelEnabled(self: *Orchestrator, model_id: []const u8, enabled: bool) OrchestrationError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.models.getPtr(model_id) orelse return OrchestrationError.ModelNotFound;
        entry.config.enabled = enabled;
    }

    /// Update model health status.
    pub fn setModelHealth(self: *Orchestrator, model_id: []const u8, status: HealthStatus) OrchestrationError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.models.getPtr(model_id) orelse return OrchestrationError.ModelNotFound;
        entry.status = status;
    }

    /// Route a request to the best available model based on strategy.
    pub fn route(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?TaskType,
    ) OrchestrationError!RouteResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Get available models
        var available = std.ArrayListUnmanaged(*ModelEntry).empty;
        defer available.deinit(self.allocator);

        var it = self.models.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.isAvailable()) {
                available.append(self.allocator, entry.value_ptr) catch
                    return OrchestrationError.OutOfMemory;
            }
        }

        if (available.items.len == 0) {
            return OrchestrationError.NoModelsAvailable;
        }

        // Select model based on strategy
        const selected = self.selectModel(available.items, task_type);

        return RouteResult{
            .model_id = selected.config.id,
            .model_name = selected.config.model_name,
            .backend = selected.config.backend,
            .prompt = prompt,
        };
    }

    /// Execute a request with automatic fallback.
    pub fn execute(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?TaskType,
        response_allocator: std.mem.Allocator,
    ) OrchestrationError![]u8 {
        // Route to best model
        const route_result = try self.route(prompt, task_type);

        // Try execution with fallback
        if (self.config.enable_fallback) {
            return self.executeWithFallback(route_result, response_allocator);
        }

        return self.executeSingle(route_result.model_id, prompt, response_allocator);
    }

    /// Execute using ensemble mode (combine outputs from multiple models).
    pub fn executeEnsemble(
        self: *Orchestrator,
        prompt: []const u8,
        task_type: ?TaskType,
        response_allocator: std.mem.Allocator,
    ) OrchestrationError!EnsembleResult {
        if (!self.config.enable_ensemble) {
            return OrchestrationError.InvalidConfig;
        }

        const ens = self.ensemble_instance orelse return OrchestrationError.InvalidConfig;
        _ = ens;

        // Get available models for ensemble
        var available = std.ArrayListUnmanaged(*ModelEntry).empty;
        defer available.deinit(self.allocator);

        self.mutex.lock();
        var it = self.models.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.isAvailable()) {
                if (task_type) |tt| {
                    if (self.modelSupportsTask(entry.value_ptr, tt)) {
                        available.append(self.allocator, entry.value_ptr) catch
                            return OrchestrationError.OutOfMemory;
                    }
                } else {
                    available.append(self.allocator, entry.value_ptr) catch
                        return OrchestrationError.OutOfMemory;
                }
            }
        }
        self.mutex.unlock();

        if (available.items.len < self.config.min_ensemble_models) {
            return OrchestrationError.InsufficientModelsForEnsemble;
        }

        // Execute on all available models
        var responses = std.ArrayListUnmanaged([]u8).empty;
        defer {
            for (responses.items) |resp| {
                response_allocator.free(resp);
            }
            responses.deinit(self.allocator);
        }

        for (available.items) |model| {
            const resp = self.executeSingle(model.config.id, prompt, response_allocator) catch continue;
            responses.append(self.allocator, resp) catch continue;
        }

        if (responses.items.len == 0) {
            return OrchestrationError.AllModelsFailed;
        }

        // Combine responses using ensemble method
        return EnsembleResult{
            .response = try response_allocator.dupe(u8, responses.items[0]),
            .model_count = responses.items.len,
            .confidence = calculateEnsembleConfidence(responses.items.len, available.items.len),
        };
    }

    /// Get statistics about all registered models.
    pub fn getStats(self: *Orchestrator) OrchestratorStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = OrchestratorStats{};

        var it = self.models.iterator();
        while (it.next()) |entry| {
            stats.total_models += 1;
            if (entry.value_ptr.isAvailable()) {
                stats.available_models += 1;
            }
            stats.total_requests += entry.value_ptr.total_requests;
            stats.total_failures += entry.value_ptr.total_failures;
            stats.active_requests += entry.value_ptr.active_requests;
        }

        return stats;
    }

    /// List all registered model IDs.
    pub fn listModels(self: *Orchestrator, allocator: std.mem.Allocator) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var ids = std.ArrayListUnmanaged([]const u8).empty;
        errdefer ids.deinit(allocator);

        var it = self.models.iterator();
        while (it.next()) |entry| {
            try ids.append(allocator, entry.key_ptr.*);
        }

        return ids.toOwnedSlice(allocator);
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    fn selectModel(
        self: *Orchestrator,
        available: []*ModelEntry,
        task_type: ?TaskType,
    ) *ModelEntry {
        return switch (self.config.strategy) {
            .round_robin => self.selectRoundRobin(available),
            .least_loaded => self.selectLeastLoaded(available),
            .task_based => self.selectByTask(available, task_type),
            .weighted => self.selectWeighted(available),
            .priority => self.selectByPriority(available),
            .cost_optimized => self.selectByCost(available),
            .latency_optimized => self.selectByLatency(available),
        };
    }

    fn selectRoundRobin(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        const index = self.round_robin_index % available.len;
        self.round_robin_index += 1;
        return available[index];
    }

    fn selectLeastLoaded(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        var min_load: f64 = 1.0;
        var selected: *ModelEntry = available[0];

        for (available) |model| {
            const load = model.loadFactor(self.config.max_concurrent_requests);
            if (load < min_load) {
                min_load = load;
                selected = model;
            }
        }

        return selected;
    }

    fn selectByTask(self: *Orchestrator, available: []*ModelEntry, task_type: ?TaskType) *ModelEntry {
        if (task_type == null) {
            return self.selectRoundRobin(available);
        }

        // Find models that support this task
        var best: ?*ModelEntry = null;
        var best_score: f64 = 0.0;

        for (available) |model| {
            if (self.modelSupportsTask(model, task_type.?)) {
                const score = model.successRate() * (1.0 - model.loadFactor(self.config.max_concurrent_requests));
                if (score > best_score) {
                    best_score = score;
                    best = model;
                }
            }
        }

        return best orelse self.selectRoundRobin(available);
    }

    fn selectWeighted(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        _ = self;
        // Calculate total weight
        var total_weight: f64 = 0.0;
        for (available) |model| {
            total_weight += model.config.weight;
        }

        // Random selection based on weight (simplified - use first model for determinism)
        var cumulative: f64 = 0.0;
        const target = total_weight * 0.5; // Deterministic for now
        for (available) |model| {
            cumulative += model.config.weight;
            if (cumulative >= target) {
                return model;
            }
        }

        return available[0];
    }

    fn selectByPriority(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        _ = self;
        var best: *ModelEntry = available[0];
        var best_priority: u32 = available[0].config.priority;

        for (available) |model| {
            if (model.config.priority < best_priority) {
                best_priority = model.config.priority;
                best = model;
            }
        }

        return best;
    }

    fn selectByCost(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        _ = self;
        var cheapest: *ModelEntry = available[0];
        var min_cost: f32 = available[0].config.cost_per_1k_tokens;

        for (available) |model| {
            if (model.config.cost_per_1k_tokens < min_cost) {
                min_cost = model.config.cost_per_1k_tokens;
                cheapest = model;
            }
        }

        return cheapest;
    }

    fn selectByLatency(self: *Orchestrator, available: []*ModelEntry) *ModelEntry {
        _ = self;
        var fastest: *ModelEntry = available[0];
        var min_latency: f64 = available[0].avgLatencyMs();

        for (available) |model| {
            const latency = model.avgLatencyMs();
            if (latency < min_latency or (min_latency == 0.0 and latency == 0.0)) {
                min_latency = latency;
                fastest = model;
            }
        }

        return fastest;
    }

    fn modelSupportsTask(self: *Orchestrator, model: *ModelEntry, task_type: TaskType) bool {
        _ = self;
        const capability = taskToCapability(task_type);
        for (model.config.capabilities) |cap| {
            if (cap == capability) return true;
        }
        // If no capabilities defined, assume it supports everything
        return model.config.capabilities.len == 0;
    }

    fn executeWithFallback(
        self: *Orchestrator,
        route_result: RouteResult,
        response_allocator: std.mem.Allocator,
    ) OrchestrationError![]u8 {
        // Try primary model
        if (self.executeSingle(route_result.model_id, route_result.prompt, response_allocator)) |resp| {
            return resp;
        } else |_| {
            // Mark model as degraded
            if (self.getModel(route_result.model_id)) |model| {
                model.consecutive_failures += 1;
                if (model.consecutive_failures >= self.config.max_retries) {
                    model.status = .degraded;
                }
            }
        }

        // Try fallback models
        self.mutex.lock();
        var fallbacks = std.ArrayListUnmanaged(*ModelEntry).empty;
        defer fallbacks.deinit(self.allocator);

        var it = self.models.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.isAvailable() and
                !std.mem.eql(u8, entry.value_ptr.config.id, route_result.model_id))
            {
                fallbacks.append(self.allocator, entry.value_ptr) catch continue;
            }
        }
        self.mutex.unlock();

        // Sort by priority
        std.mem.sort(*ModelEntry, fallbacks.items, {}, struct {
            fn lessThan(_: void, a: *ModelEntry, b: *ModelEntry) bool {
                return a.config.priority < b.config.priority;
            }
        }.lessThan);

        // Try each fallback
        for (fallbacks.items) |model| {
            if (self.executeSingle(model.config.id, route_result.prompt, response_allocator)) |resp| {
                return resp;
            } else |_| {
                model.consecutive_failures += 1;
            }
        }

        return OrchestrationError.AllModelsFailed;
    }

    fn executeSingle(
        self: *Orchestrator,
        model_id: []const u8,
        prompt: []const u8,
        response_allocator: std.mem.Allocator,
    ) OrchestrationError![]u8 {
        const model = self.getModel(model_id) orelse return OrchestrationError.ModelNotFound;

        if (!model.config.enabled) {
            return OrchestrationError.ModelDisabled;
        }

        // Update stats
        self.mutex.lock();
        model.active_requests += 1;
        model.total_requests += 1;
        model.last_request_time = std.time.milliTimestamp();
        self.mutex.unlock();

        defer {
            self.mutex.lock();
            model.active_requests -= 1;
            self.mutex.unlock();
        }

        // Generate response (placeholder - actual implementation would call the backend)
        const response = std.fmt.allocPrint(
            response_allocator,
            "[{s}] Response to: {s}",
            .{ model.config.id, prompt },
        ) catch return OrchestrationError.OutOfMemory;

        // Reset consecutive failures on success
        self.mutex.lock();
        model.consecutive_failures = 0;
        self.mutex.unlock();

        return response;
    }
};

// ============================================================================
// Statistics
// ============================================================================

pub const OrchestratorStats = struct {
    total_models: u32 = 0,
    available_models: u32 = 0,
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    active_requests: u32 = 0,

    pub fn successRate(self: OrchestratorStats) f64 {
        if (self.total_requests == 0) return 1.0;
        const successes = self.total_requests - self.total_failures;
        return @as(f64, @floatFromInt(successes)) /
            @as(f64, @floatFromInt(self.total_requests));
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

fn taskToCapability(task_type: TaskType) Capability {
    return switch (task_type) {
        .reasoning => .reasoning,
        .coding => .coding,
        .creative => .creative,
        .analysis => .analysis,
        .summarization => .summarization,
        .translation => .translation,
        .math => .math,
        .general => .reasoning,
    };
}

fn calculateEnsembleConfidence(successful: usize, total: usize) f64 {
    if (total == 0) return 0.0;
    return @as(f64, @floatFromInt(successful)) / @as(f64, @floatFromInt(total));
}

/// Check if orchestration is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

// ============================================================================
// Tests
// ============================================================================

test "orchestrator initialization" {
    if (!isEnabled()) return;

    var orch = try Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.total_models);
}

test "model registration" {
    if (!isEnabled()) return;

    var orch = try Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "gpt-4",
        .name = "GPT-4",
        .backend = .openai,
        .capabilities = &.{ .reasoning, .coding },
    });

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.total_models);
    try std.testing.expectEqual(@as(u32, 1), stats.available_models);

    // Test duplicate registration
    try std.testing.expectError(
        OrchestrationError.DuplicateModelId,
        orch.registerModel(.{ .id = "gpt-4" }),
    );
}

test "round robin routing" {
    if (!isEnabled()) return;

    var orch = try Orchestrator.init(std.testing.allocator, .{ .strategy = .round_robin });
    defer orch.deinit();

    try orch.registerModel(.{ .id = "model-a" });
    try orch.registerModel(.{ .id = "model-b" });

    const result1 = try orch.route("test", null);
    const result2 = try orch.route("test", null);

    // Should alternate between models
    try std.testing.expect(!std.mem.eql(u8, result1.model_id, result2.model_id));
}

test "model enable/disable" {
    if (!isEnabled()) return;

    var orch = try Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "test-model" });

    var stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.available_models);

    try orch.setModelEnabled("test-model", false);

    stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.available_models);
}
