const feature_manager = @import("feature_manager.zig");
const state = @import("state.zig");

const core = @import("../shared/core/core.zig");
const lifecycle = @import("../shared/core/lifecycle.zig");
const core_logging = @import("../shared/core/logging.zig");
const structured_logging = @import("../shared/logging/logging.zig");
const plugins = @import("../shared/mod.zig");
const ai = @import("../features/ai/mod.zig");
const gpu = @import("../features/gpu/mod.zig");
const database = @import("../features/database/mod.zig");
const web = @import("../features/web/mod.zig");
const monitoring = @import("../features/monitoring/mod.zig");
const connectors = @import("../features/connectors/mod.zig");
const utils = @import("../shared/utils/mod.zig");

const Environment = feature_manager.Environment;
const FeatureDescriptor = feature_manager.FeatureDescriptor;
const FeatureCategory = feature_manager.FeatureCategory;
const RuntimeState = state.RuntimeState;

/// Static catalog describing every feature exposed by the framework runtime.
pub const descriptors = [_]FeatureDescriptor{
    .{
        .name = "core.kernel",
        .display_name = "Core Kernel",
        .category = .core,
        .description = "Initializes the minimal ABI core state machine.",
        .init = initCoreKernel,
        .deinit = deinitCoreKernel,
    },
    .{
        .name = "core.lifecycle",
        .display_name = "Lifecycle Manager",
        .category = .core,
        .dependencies = &.{"core.kernel"},
        .description = "Configures the lifecycle coordinator and bootstrap logging.",
        .init = initLifecycle,
        .deinit = deinitLifecycle,
    },
    .{
        .name = "core.logging.bootstrap",
        .display_name = "Bootstrap Logger",
        .category = .logging,
        .dependencies = &.{"core.kernel"},
        .description = "Enables lightweight logging for early initialization stages.",
        .init = initBootstrapLogger,
        .deinit = deinitBootstrapLogger,
    },
    .{
        .name = "core.logging.structured",
        .display_name = "Structured Logger",
        .category = .logging,
        .dependencies = &.{ "core.kernel", "core.logging.bootstrap" },
        .description = "Installs the structured logger with configurable sinks.",
        .init = initStructuredLogger,
        .deinit = deinitStructuredLogger,
    },
    .{
        .name = "shared.plugins",
        .display_name = "Plugin Runtime",
        .category = .plugins,
        .dependencies = &.{"core.logging.structured"},
        .description = "Starts the enhanced plugin loader and registry stack.",
        .init = initPluginRuntime,
        .deinit = deinitPluginRuntime,
    },
    .{
        .name = "feature.ai",
        .display_name = "AI Toolkit",
        .category = .ai,
        .dependencies = &.{"core.logging.structured"},
        .description = "Exposes the neural, transformer, and agent orchestration APIs.",
        .init = initAiFeature,
        .deinit = deinitAiFeature,
    },
    .{
        .name = "feature.gpu",
        .display_name = "GPU Acceleration",
        .category = .gpu,
        .dependencies = &.{"core.logging.structured"},
        .description = "Registers GPU backends, shader pipelines, and compute schedulers.",
        .init = initGpuFeature,
        .deinit = deinitGpuFeature,
    },
    .{
        .name = "feature.database",
        .display_name = "Vector Database",
        .category = .database,
        .dependencies = &.{"core.logging.structured"},
        .description = "Initializes persistence utilities and data access helpers.",
        .init = initDatabaseFeature,
        .deinit = deinitDatabaseFeature,
    },
    .{
        .name = "feature.web",
        .display_name = "Web Services",
        .category = .web,
        .dependencies = &.{"core.logging.structured"},
        .description = "Enables HTTP servers, clients, and protocol integrations.",
        .init = initWebFeature,
        .deinit = deinitWebFeature,
    },
    .{
        .name = "feature.monitoring",
        .display_name = "Observability",
        .category = .monitoring,
        .dependencies = &.{"core.logging.structured"},
        .description = "Makes performance tracing and metrics collectors available.",
        .init = initMonitoringFeature,
        .deinit = deinitMonitoringFeature,
    },
    .{
        .name = "feature.connectors",
        .display_name = "External Connectors",
        .category = .connectors,
        .dependencies = &.{ "core.logging.structured", "shared.plugins" },
        .description = "Registers OpenAI, Ollama, and plugin bridge connectors.",
        .init = initConnectorsFeature,
        .deinit = deinitConnectorsFeature,
    },
    .{
        .name = "shared.utilities",
        .display_name = "Shared Utilities",
        .category = .utilities,
        .dependencies = &.{"core.kernel"},
        .description = "Provides HTTP, JSON, crypto, and filesystem helpers.",
        .init = initUtilitiesFeature,
        .deinit = deinitUtilitiesFeature,
    },
};

fn runtimeState(env: Environment) !*RuntimeState {
    return env.contextAs(RuntimeState) orelse error.MissingRuntimeState;
}

fn initCoreKernel(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    try core.init(state_ptr.allocator);
}

fn deinitCoreKernel(env: Environment) void {
    _ = env;
    core.deinit();
}

fn initLifecycle(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = lifecycle; // ensure module is referenced
    _ = core_logging; // ensure bootstrap logger definitions are linked
    _ = state_ptr;
}

fn deinitLifecycle(env: Environment) void {
    _ = env;
}

fn initBootstrapLogger(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    try core_logging.log.init(env.allocator);
    core_logging.log.setLevel(.info);
    if (!state_ptr.options.logging.enabled) {
        core_logging.log.setLevel(.warn);
    }
}

fn deinitBootstrapLogger(env: Environment) void {
    _ = env;
    core_logging.log.deinit();
}

fn initStructuredLogger(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    if (!state_ptr.options.logging.enabled) return;
    try structured_logging.initGlobalLogger(env.allocator, state_ptr.options.logging.config);
    state_ptr.setLogger(structured_logging.getGlobalLogger());
}

fn deinitStructuredLogger(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    if (!state_ptr.options.logging.enabled) return;
    structured_logging.deinitGlobalLogger();
    state_ptr.clearLogger();
}

fn initPluginRuntime(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    if (!state_ptr.options.ensure_plugin_system) return;
    const registry = try plugins.PluginRegistry.init(env.allocator);
    state_ptr.setPluginRegistry(registry);
}

fn deinitPluginRuntime(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    if (state_ptr.plugin_registry) |*registry| {
        registry.deinit();
        state_ptr.plugin_registry = null;
    }
}

/// Initialize AI feature - placeholder for future implementation
fn initAiFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = ai; // Reference to ensure ai module is available
    logFeatureActivation(state_ptr, "feature.ai", "initialized");
    // TODO: Implement actual AI feature initialization (model loading, etc.)
}

/// Deinitialize AI feature - placeholder for future implementation
fn deinitAiFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.ai", "shutdown");
    // TODO: Implement actual AI feature cleanup (model unloading, etc.)
}

fn initGpuFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = gpu;
    logFeatureActivation(state_ptr, "feature.gpu", "initialized");
}

fn deinitGpuFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.gpu", "shutdown");
}

fn initDatabaseFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = database;
    logFeatureActivation(state_ptr, "feature.database", "initialized");
}

fn deinitDatabaseFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.database", "shutdown");
}

fn initWebFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = web;
    logFeatureActivation(state_ptr, "feature.web", "initialized");
}

fn deinitWebFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.web", "shutdown");
}

fn initMonitoringFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = monitoring;
    logFeatureActivation(state_ptr, "feature.monitoring", "initialized");
}

fn deinitMonitoringFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.monitoring", "shutdown");
}

fn initConnectorsFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = connectors;
    logFeatureActivation(state_ptr, "feature.connectors", "initialized");
}

fn deinitConnectorsFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "feature.connectors", "shutdown");
}

fn initUtilitiesFeature(env: Environment) anyerror!void {
    const state_ptr = try runtimeState(env);
    _ = utils;
    logFeatureActivation(state_ptr, "shared.utilities", "initialized");
}

fn deinitUtilitiesFeature(env: Environment) void {
    const state_ptr = runtimeState(env) catch return;
    logFeatureActivation(state_ptr, "shared.utilities", "shutdown");
}

fn logFeatureActivation(state_ptr: *RuntimeState, feature: []const u8, event: []const u8) void {
    if (state_ptr.logger) |_| {
        structured_logging.info(
            "feature transition",
            .{ .feature = feature, .event = event },
            @src(),
        ) catch {};
    } else {
        core_logging.log.info("{s}: {s}", .{ feature, event });
    }
}

pub fn featureCount() usize {
    return descriptors.len;
}
