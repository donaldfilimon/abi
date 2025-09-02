# Abi AI Framework Plugin System

## 🔌 **Overview**

The Abi AI Framework includes a robust, cross-platform plugin system that enables dynamic loading and management of extensions. The plugin system is designed with the following principles:

- **Cross-Platform Compatibility**: Works on Windows (.dll), Linux (.so), and macOS (.dylib)
- **Type Safety**: Uses C-compatible interfaces with Zig wrapper types
- **Dependency Management**: Handles plugin dependencies and load ordering
- **Lifecycle Management**: Complete plugin lifecycle from loading to unloading
- **Event System**: Plugin communication through events and service registry
- **Resource Management**: Automatic cleanup and memory management

## 🏗️ **Architecture**

### **Core Components**

1. **Plugin Interface** (`src/plugins/interface.zig`)
   - Defines the standard C-compatible plugin interface
   - Provides safe Zig wrapper around C interface
   - Manages plugin lifecycle and state transitions

2. **Plugin Loader** (`src/plugins/loader.zig`)
   - Handles cross-platform dynamic library loading
   - Discovers plugins in specified directories
   - Manages library handles and symbol resolution

3. **Plugin Registry** (`src/plugins/registry.zig`)
   - Centralized plugin management
   - Dependency resolution and load ordering
   - Event broadcasting and service discovery

4. **Plugin Types** (`src/plugins/types.zig`)
   - Common types, errors, and enumerations
   - Plugin metadata and configuration structures
   - Version compatibility checking

## 📋 **Plugin Types Supported**

The framework supports various plugin categories:

### **Database Plugins**
- `vector_database` - Custom vector database implementations
- `indexing_algorithm` - Alternative indexing strategies (LSH, HNSW variants)
- `compression_algorithm` - Data compression methods

### **AI/ML Plugins**
- `neural_network` - Custom neural network architectures
- `embedding_generator` - Text/image embedding generators
- `training_algorithm` - Training optimization algorithms
- `inference_engine` - Inference acceleration engines

### **Processing Plugins**
- `text_processor` - Text analysis and processing
- `image_processor` - Image manipulation and analysis
- `audio_processor` - Audio processing and analysis
- `data_transformer` - Data transformation utilities

### **I/O Plugins**
- `data_loader` - Custom data format loaders
- `data_exporter` - Data export functionality
- `protocol_handler` - Network protocol implementations

### **Utility Plugins**
- `logger` - Custom logging implementations
- `metrics_collector` - Performance metrics collection
- `security_provider` - Authentication and security
- `configuration_provider` - Configuration management

## 🚀 **Creating a Plugin**

### **1. Basic Plugin Structure**

```zig
const std = @import("std");
// Import plugin types from the framework
const PluginInterface = @import("abi").plugins.PluginInterface;
const PluginInfo = @import("abi").plugins.PluginInfo;
const PluginContext = @import("abi").plugins.PluginContext;

// Plugin metadata
const PLUGIN_INFO = PluginInfo{
    .name = "my_awesome_plugin",
    .version = .{ .major = 1, .minor = 0, .patch = 0 },
    .author = "Your Name",
    .description = "Description of your plugin",
    .plugin_type = .text_processor,
    .abi_version = .{ .major = 1, .minor = 0, .patch = 0 },
    .provides = &[_][]const u8{"feature1", "feature2"},
    .dependencies = &[_][]const u8{}, // Dependencies on other plugins
    .license = "MIT",
};

// Plugin state
var initialized = false;
var running = false;

// Required interface implementation
fn getInfo() callconv(.c) *const PluginInfo {
    return &PLUGIN_INFO;
}

fn initPlugin(context: *PluginContext) callconv(.c) c_int {
    if (initialized) return -1;
    
    context.log(1, "Initializing my awesome plugin");
    initialized = true;
    return 0; // Success
}

fn deinitPlugin(context: *PluginContext) callconv(.c) void {
    if (!initialized) return;
    
    context.log(1, "Deinitializing my awesome plugin");
    initialized = false;
    running = false;
}

fn startPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!initialized || running) return -1;
    
    context.log(1, "Starting my awesome plugin");
    running = true;
    return 0;
}

fn stopPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!running) return -1;
    
    context.log(1, "Stopping my awesome plugin");
    running = false;
    return 0;
}

// Process function - implement your plugin logic here
fn processData(context: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int {
    if (!running) return -1;
    
    // Your processing logic here
    _ = context;
    _ = input;
    _ = output;
    
    return 0; // Success
}

// Plugin interface vtable
const PLUGIN_INTERFACE = PluginInterface{
    .get_info = getInfo,
    .init = initPlugin,
    .deinit = deinitPlugin,
    .start = startPlugin,
    .stop = stopPlugin,
    .process = processData,
};

// Plugin entry point - this is called by the framework
export fn abi_plugin_create() ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}
```

### **2. Building the Plugin**

To build your plugin as a shared library:

```bash
# Windows
zig build-lib -dynamic my_plugin.zig -O ReleaseFast

# Linux
zig build-lib -dynamic my_plugin.zig -O ReleaseFast

# macOS  
zig build-lib -dynamic my_plugin.zig -O ReleaseFast
```

This creates:
- Windows: `my_plugin.dll`
- Linux: `libmy_plugin.so`
- macOS: `libmy_plugin.dylib`

### **3. Advanced Plugin Features**

#### **Configuration Support**

```zig
fn configurePlugin(context: *PluginContext, config: *const PluginConfig) callconv(.c) c_int {
    const my_setting = config.getParameter("my_setting") orelse "default";
    context.log(1, std.fmt.bufPrint(buffer, "Configured with: {s}", .{my_setting}));
    return 0;
}
```

#### **Event Handling**

```zig
fn onEvent(context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int {
    switch (event_type) {
        1 => context.log(1, "System startup event received"),
        2 => context.log(1, "System shutdown event received"),
        else => {},
    }
    return 0;
}
```

#### **Custom API Exposure**

```zig
fn getApi(api_name: [*:0]const u8) callconv(.c) ?*anyopaque {
    const name = std.mem.span(api_name);
    
    if (std.mem.eql(u8, name, "my_custom_api")) {
        return @ptrCast(&my_custom_function);
    }
    
    return null;
}
```

## 🔧 **Using Plugins in Your Application**

### **1. Basic Plugin Loading**

```zig
const std = @import("std");
const plugins = @import("abi").plugins;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize plugin registry
    var registry = try plugins.init(allocator);
    defer registry.deinit();
    
    // Add plugin search paths
    try registry.addPluginPath("./plugins");
    try registry.addPluginPath("/usr/local/lib/abi-plugins");
    
    // Discover available plugins
    var discovered = try registry.discoverPlugins();
    defer {
        for (discovered.items) |path| {
            allocator.free(path);
        }
        discovered.deinit(allocator);
    }
    
    std.log.info("Found {} plugins", .{discovered.items.len});
    
    // Load a specific plugin
    try registry.loadPlugin("./plugins/my_plugin.dll");
    
    // Initialize and start the plugin
    try registry.initializePlugin("my_awesome_plugin", null);
    try registry.startPlugin("my_awesome_plugin");
    
    // Use the plugin
    const plugin = registry.getPlugin("my_awesome_plugin");
    if (plugin) |p| {
        // Process data with the plugin
        var input_data = "Hello, Plugin!";
        var output_data: [256]u8 = undefined;
        try p.process(&input_data, &output_data);
    }
    
    // Stop and unload
    try registry.stopPlugin("my_awesome_plugin");
    try registry.unloadPlugin("my_awesome_plugin");
}
```

### **2. Plugin Management**

```zig
// Get all plugins of a specific type
var text_processors = try registry.getPluginsByType(.text_processor);
defer text_processors.deinit(allocator);

for (text_processors.items) |plugin| {
    const info = plugin.getInfo();
    std.log.info("Text processor: {s} v{}", .{ info.name, info.version });
}

// Get plugin information
if (registry.getPluginInfo("my_plugin")) |info| {
    std.log.info("Plugin: {s} by {s}", .{ info.name, info.author });
    std.log.info("Description: {s}", .{info.description});
}

// Configure a plugin
var config = plugins.PluginConfig.init(allocator);
defer config.deinit();

try config.setParameter("input_format", "json");
try config.setParameter("output_format", "xml");
try registry.configurePlugin("my_plugin", &config);
```

### **3. Event System**

```zig
// Broadcast events to all plugins
try registry.broadcastEvent(1, null); // System startup
try registry.broadcastEvent(2, null); // System shutdown

// Register custom event handlers
const MyEventHandler = struct {
    fn handleEvent(event_data: ?*anyopaque) void {
        _ = event_data;
        std.log.info("Custom event received!");
    }
};

try registry.registerEventHandler(100, MyEventHandler.handleEvent);
```

## 🔒 **Security Considerations**

### **Plugin Sandboxing**

```zig
var config = plugins.PluginConfig.init(allocator);
config.sandboxed = true;
config.max_memory_mb = 100;
config.max_cpu_time_ms = 5000;
config.permissions = &[_][]const u8{"read_files", "network_access"};
```

### **Validation and Trust**

- Always validate plugin signatures in production
- Use allowlists for trusted plugin directories
- Implement resource limits to prevent abuse
- Monitor plugin behavior for anomalies

## 🧪 **Testing Plugins**

### **Unit Testing**

```zig
test "plugin functionality" {
    var registry = try plugins.init(std.testing.allocator);
    defer registry.deinit();
    
    // Load test plugin
    try registry.loadPlugin("./test_plugins/test_plugin.dll");
    
    // Test plugin lifecycle
    try registry.initializePlugin("test_plugin", null);
    try std.testing.expectEqual(.initialized, registry.getPlugin("test_plugin").?.getState());
    
    try registry.startPlugin("test_plugin");
    try std.testing.expectEqual(.running, registry.getPlugin("test_plugin").?.getState());
}
```

### **Integration Testing**

```zig
test "plugin integration" {
    var registry = try plugins.init(std.testing.allocator);
    defer registry.deinit();
    
    // Test plugin discovery
    try registry.addPluginPath("./test_plugins");
    var discovered = try registry.discoverPlugins();
    defer discovered.deinit(std.testing.allocator);
    
    try std.testing.expect(discovered.items.len > 0);
}
```

## 📊 **Performance Considerations**

### **Plugin Loading**
- Plugins are loaded on-demand to minimize startup time
- Multiple plugins can be loaded in parallel
- Plugin discovery caches results for faster subsequent loads

### **Runtime Performance**
- Plugin calls use direct function pointers (minimal overhead)
- Memory allocations are tracked per plugin
- Resource limits prevent plugins from consuming excessive resources

### **Best Practices**
- Keep plugin interfaces simple and focused
- Minimize data copying between plugin and host
- Use event-driven communication for loose coupling
- Implement proper error handling and recovery

## 🚀 **Production Deployment**

### **Plugin Distribution**
```bash
# Plugin package structure
my-plugin/
├── plugin.manifest     # Plugin metadata
├── libmy_plugin.so     # Linux binary
├── my_plugin.dll       # Windows binary
├── libmy_plugin.dylib  # macOS binary
├── README.md           # Documentation
└── LICENSE             # License file
```

### **Configuration Management**
```toml
# plugins.toml
[plugins]
search_paths = [
    "./plugins",
    "/usr/local/lib/abi-plugins",
    "~/.abi/plugins"
]

[plugins.text_processor]
enabled = true
auto_load = true
config = { input_format = "json", output_format = "xml" }

[plugins.my_custom_plugin]
enabled = false
priority = 10
max_memory_mb = 200
```

### **Monitoring and Logging**
```zig
// Plugin metrics collection
const metrics = try plugin.getMetrics(buffer);
std.log.info("Plugin metrics: {s}", .{metrics});

// Plugin health checks
const status = plugin.getStatus();
if (status != 2) { // Not running
    std.log.warn("Plugin {} is not healthy (status: {})", .{ plugin.getInfo().name, status });
}
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Plugin Not Loading**
   - Check file permissions
   - Verify plugin path is correct
   - Ensure plugin exports `abi_plugin_create` function

2. **ABI Incompatibility**
   - Verify plugin was built for correct architecture
   - Check plugin ABI version matches framework version
   - Ensure calling convention is `.c`

3. **Missing Dependencies**
   - Check plugin dependency list
   - Ensure dependent plugins are loaded first
   - Verify plugin search paths include all dependencies

### **Debug Mode**
```zig
var config = plugins.PluginConfig.init(allocator);
config.debug_mode = true;  // Enable verbose logging
```

## 📚 **API Reference**

For complete API documentation, see:
- [Plugin Interface API](./api/plugins.md)
- [Plugin Types Reference](./api/plugin_types.md)
- [Plugin Examples](../examples/plugins/)

## 🎯 **Future Enhancements**

- **Hot Reloading**: Reload plugins without restarting the application
- **Plugin Marketplace**: Centralized plugin repository and distribution
- **Visual Plugin Editor**: GUI for creating and configuring plugins
- **Language Bindings**: Support for plugins written in C, C++, and Rust
- **Distributed Plugins**: Load plugins from remote locations
- **Plugin Analytics**: Usage statistics and performance metrics

---

The Abi AI Framework plugin system provides a powerful foundation for extensibility while maintaining safety, performance, and ease of use. Whether you're building custom data processors, AI model integrations, or specialized I/O handlers, the plugin system gives you the flexibility to extend the framework to meet your specific needs.
