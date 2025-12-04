const std = @import("std");
const abi = @import("abi");
const plugins = abi.plugins;
const connectors = abi.connectors;

// Import the specific types we need
const PluginContext = plugins.types.PluginContext;
const PluginVersion = plugins.types.PluginVersion;
const PluginInfo = plugins.types.PluginInfo;
const PluginConfig = plugins.types.PluginConfig;

test "plugin system with connector integration" {
    const allocator = std.testing.allocator;

    // Initialize plugin registry
    var registry = try plugins.init(allocator);
    defer registry.deinit();

    // Test that registry is properly initialized
    try std.testing.expectEqual(@as(usize, 0), registry.getPluginCount());

    // Test plugin version compatibility
    const plugin_version = plugins.VERSION;
    try std.testing.expectEqual(@as(u32, 1), plugin_version.MAJOR);
    try std.testing.expectEqual(@as(u32, 0), plugin_version.MINOR);
    try std.testing.expectEqual(@as(u32, 0), plugin_version.PATCH);

    // Test version string
    try std.testing.expectEqualStrings("1.0.0", plugin_version.string());

    // Test compatibility checking
    try std.testing.expect(plugin_version.isCompatible(1, 0));
    try std.testing.expect(!plugin_version.isCompatible(1, 1)); // 1.0.0 is not compatible with 1.1.0 requirement
    try std.testing.expect(!plugin_version.isCompatible(2, 0));
}

test "connector configuration with plugin context" {
    const allocator = std.testing.allocator;

    // Test Ollama configuration
    const ollama_config = connectors.OllamaConfig{
        .host = "http://localhost:11434",
        .model = "nomic-embed-text",
    };

    _ = ollama_config; // Suppress unused variable warning

    // Test that we can create a plugin context that could use this config
    var config = PluginConfig.init(allocator);
    defer config.deinit();

    config.enabled = true;
    config.auto_load = true;
    config.priority = 10;

    // Test plugin context creation
    const context = PluginContext{
        .allocator = allocator,
        .config = &config,
    };

    // Test that context is properly initialized
    try std.testing.expect(context.config == &config);
}

test "plugin types and connector types compatibility" {
    // Test that plugin types are properly defined
    const plugin_types = [_]plugins.PluginType{
        plugins.PluginType.text_processor,
        plugins.PluginType.neural_network,
        plugins.PluginType.vector_database,
        plugins.PluginType.embedding_generator,
        plugins.PluginType.custom,
    };

    try std.testing.expectEqual(@as(usize, 5), plugin_types.len);

    // Test connector provider types
    const provider_types = [_]connectors.ProviderType{
        connectors.ProviderType.ollama,
        connectors.ProviderType.openai,
    };

    try std.testing.expectEqual(@as(usize, 2), provider_types.len);
}

test "plugin error handling with connector errors" {
    // Test plugin error types
    const plugin_errors = [_]plugins.PluginError{
        plugins.PluginError.LoadFailed,
        plugins.PluginError.InitializationFailed,
        plugins.PluginError.IncompatibleVersion,
        plugins.PluginError.DependencyMissing,
        plugins.PluginError.ExecutionFailed,
        plugins.PluginError.PermissionDenied,
        plugins.PluginError.OutOfMemory,
        plugins.PluginError.InvalidParameters,
        plugins.PluginError.PluginNotFound,
        plugins.PluginError.AlreadyRegistered,
        plugins.PluginError.NotRegistered,
        plugins.PluginError.ConflictingPlugin,
    };

    try std.testing.expectEqual(@as(usize, 12), plugin_errors.len);

    // Test connector error types
    const connector_errors = [_]connectors.ConnectorsError{
        connectors.ConnectorsError.NetworkError,
        connectors.ConnectorsError.InvalidResponse,
        connectors.ConnectorsError.ParseError,
        connectors.ConnectorsError.MissingApiKey,
    };

    try std.testing.expectEqual(@as(usize, 4), connector_errors.len);
}

test "plugin info structure with connector integration" {

    // Create a plugin info that could represent a connector plugin
    const plugin_info = PluginInfo{
        .name = "embedding_connector_plugin",
        .version = PluginVersion.init(1, 0, 0),
        .author = "Abi AI Framework Team",
        .description = "Plugin that provides embedding services via Ollama and OpenAI connectors",
        .plugin_type = plugins.PluginType.embedding_generator,
        .abi_version = PluginVersion.init(1, 0, 0),
        .dependencies = &[_][]const u8{},
        .provides = &[_][]const u8{ "embedding_service", "text_processing" },
        .requires = &[_][]const u8{"network_access"},
        .license = "MIT",
        .homepage = "https://github.com/donaldfilimon/abi",
        .repository = "https://github.com/donaldfilimon/abi.git",
    };

    // Test plugin info properties
    try std.testing.expectEqualStrings("embedding_connector_plugin", plugin_info.name);
    try std.testing.expectEqualStrings("Plugin that provides embedding services via Ollama and OpenAI connectors", plugin_info.description);
    try std.testing.expectEqual(plugins.PluginType.embedding_generator, plugin_info.plugin_type);

    // Test version compatibility
    try std.testing.expect(plugin_info.isCompatible(PluginVersion.init(1, 0, 0)));
    try std.testing.expect(!plugin_info.isCompatible(PluginVersion.init(1, 1, 0))); // 1.0.0 is not compatible with 1.1.0 requirement
    try std.testing.expect(!plugin_info.isCompatible(PluginVersion.init(2, 0, 0)));

    // Test that provides array contains expected services
    try std.testing.expectEqual(@as(usize, 2), plugin_info.provides.len);
    try std.testing.expectEqualStrings("embedding_service", plugin_info.provides[0]);
    try std.testing.expectEqualStrings("text_processing", plugin_info.provides[1]);
}
