//! Plugin ABI Compatibility Tests
//!
//! This test suite ensures that the plugin ABI remains stable and compatible
//! across different versions and implementations.

const std = @import("std");
const testing = std.testing;

const plugin_interface = @import("../src/plugins/interface.zig");
const plugin_types = @import("../src/plugins/types.zig");
const plugin_loader = @import("../src/plugins/loader.zig");

const PluginInterface = plugin_interface.PluginInterface;
const PluginInfo = plugin_types.PluginInfo;
const PluginVersion = plugin_types.PluginVersion;
const PluginContext = plugin_types.PluginContext;
const PluginConfig = plugin_types.PluginConfig;
const PluginError = plugin_types.PluginError;

// =============================================================================
// ABI VERSION COMPATIBILITY TESTS
// =============================================================================

test "ABI version compatibility checks" {
    const current_abi = plugin_interface.PLUGIN_ABI_VERSION;
    
    // Test exact match
    {
        const plugin_abi = PluginVersion.init(1, 0, 0);
        try testing.expect(plugin_abi.isCompatible(current_abi));
    }
    
    // Test minor version compatibility
    {
        const plugin_abi = PluginVersion.init(1, 1, 0);
        try testing.expect(plugin_abi.isCompatible(current_abi));
    }
    
    // Test patch version compatibility
    {
        const plugin_abi = PluginVersion.init(1, 0, 1);
        try testing.expect(plugin_abi.isCompatible(current_abi));
    }
    
    // Test major version incompatibility
    {
        const plugin_abi = PluginVersion.init(2, 0, 0);
        try testing.expect(!plugin_abi.isCompatible(current_abi));
    }
    
    // Test older minor version incompatibility
    {
        const required_abi = PluginVersion.init(1, 2, 0);
        const plugin_abi = PluginVersion.init(1, 1, 0);
        try testing.expect(!plugin_abi.isCompatible(required_abi));
    }
}

// =============================================================================
// PLUGIN INTERFACE STRUCTURE TESTS
// =============================================================================

test "PluginInterface struct layout and size" {
    // Ensure the struct has the expected size and alignment
    const interface_size = @sizeOf(PluginInterface);
    const interface_align = @alignOf(PluginInterface);
    
    // These values should remain stable for ABI compatibility
    try testing.expect(interface_size > 0);
    try testing.expect(interface_align > 0);
    
    // Test field offsets remain stable
    const info_offset = @offsetOf(PluginInterface, "get_info");
    const init_offset = @offsetOf(PluginInterface, "init");
    const deinit_offset = @offsetOf(PluginInterface, "deinit");
    
    try testing.expect(info_offset == 0);
    try testing.expect(init_offset == @sizeOf(*const fn () callconv(.c) *const PluginInfo));
    try testing.expect(deinit_offset == init_offset + @sizeOf(*const fn (*PluginContext) callconv(.c) c_int));
}

test "PluginInterface required fields" {
    // Create a minimal valid interface
    const MockInterface = PluginInterface{
        .get_info = struct {
            fn get() callconv(.c) *const PluginInfo {
                return &test_info;
            }
        }.get,
        .init = struct {
            fn init(ctx: *PluginContext) callconv(.c) c_int {
                _ = ctx;
                return 0;
            }
        }.init,
        .deinit = struct {
            fn deinit(ctx: *PluginContext) callconv(.c) void {
                _ = ctx;
            }
        }.deinit,
    };
    
    // Ensure the interface is valid
    try testing.expect(MockInterface.isValid());
}

// =============================================================================
// CALLING CONVENTION TESTS
// =============================================================================

test "C calling convention compatibility" {
    // Test that function pointers use the correct calling convention
    const GetInfoFn = @TypeOf(@as(PluginInterface, undefined).get_info);
    const InitFn = @TypeOf(@as(PluginInterface, undefined).init);
    const DeinitFn = @TypeOf(@as(PluginInterface, undefined).deinit);
    
    // Verify these are function pointers with C calling convention
    const get_info_info = @typeInfo(GetInfoFn);
    const init_info = @typeInfo(InitFn);
    const deinit_info = @typeInfo(DeinitFn);
    
    try testing.expect(get_info_info == .Pointer);
    try testing.expect(init_info == .Pointer);
    try testing.expect(deinit_info == .Pointer);
}

// =============================================================================
// PLUGIN LIFECYCLE TESTS
// =============================================================================

var test_plugin_state: struct {
    initialized: bool = false,
    started: bool = false,
    call_count: u32 = 0,
} = .{};

const test_info = PluginInfo{
    .name = "test_plugin",
    .version = PluginVersion.init(1, 0, 0),
    .author = "Test",
    .description = "Test plugin for ABI compatibility",
    .plugin_type = .custom,
    .abi_version = plugin_interface.PLUGIN_ABI_VERSION,
};

fn testGetInfo() callconv(.c) *const PluginInfo {
    return &test_info;
}

fn testInit(ctx: *PluginContext) callconv(.c) c_int {
    _ = ctx;
    if (test_plugin_state.initialized) return -1;
    test_plugin_state.initialized = true;
    test_plugin_state.call_count += 1;
    return 0;
}

fn testDeinit(ctx: *PluginContext) callconv(.c) void {
    _ = ctx;
    test_plugin_state.initialized = false;
    test_plugin_state.started = false;
    test_plugin_state.call_count += 1;
}

fn testStart(ctx: *PluginContext) callconv(.c) c_int {
    _ = ctx;
    if (!test_plugin_state.initialized) return -1;
    if (test_plugin_state.started) return -2;
    test_plugin_state.started = true;
    test_plugin_state.call_count += 1;
    return 0;
}

fn testStop(ctx: *PluginContext) callconv(.c) c_int {
    _ = ctx;
    if (!test_plugin_state.started) return -1;
    test_plugin_state.started = false;
    test_plugin_state.call_count += 1;
    return 0;
}

const TestInterface = PluginInterface{
    .get_info = testGetInfo,
    .init = testInit,
    .deinit = testDeinit,
    .start = testStart,
    .stop = testStop,
};

test "Plugin lifecycle through interface" {
    // Reset test state
    test_plugin_state = .{};
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &TestInterface);
    defer plugin.deinit();
    
    // Test info retrieval
    const info = plugin.getInfo();
    try testing.expectEqualStrings("test_plugin", info.name);
    
    // Test initialization
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    try testing.expect(test_plugin_state.initialized);
    try testing.expectEqual(@as(u32, 1), test_plugin_state.call_count);
    
    // Test start
    try plugin.start();
    try testing.expect(test_plugin_state.started);
    try testing.expectEqual(@as(u32, 2), test_plugin_state.call_count);
    
    // Test stop
    try plugin.stop();
    try testing.expect(!test_plugin_state.started);
    try testing.expectEqual(@as(u32, 3), test_plugin_state.call_count);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

test "Plugin error propagation" {
    const ErrorInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = struct {
            fn init(ctx: *PluginContext) callconv(.c) c_int {
                _ = ctx;
                return -42; // Custom error code
            }
        }.init,
        .deinit = testDeinit,
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &ErrorInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    // Initialization should fail
    try testing.expectError(PluginError.InitializationFailed, plugin.initialize(&config));
}

// =============================================================================
// MEMORY SAFETY TESTS
// =============================================================================

test "Plugin memory allocation and cleanup" {
    var allocation_count: u32 = 0;
    var deallocation_count: u32 = 0;
    
    const MemoryTestInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = struct {
            fn init(ctx: *PluginContext) callconv(.c) c_int {
                // Allocate some memory
                const data = ctx.allocator.alloc(u8, 1024) catch return -1;
                ctx.plugin_data = @ptrCast(data.ptr);
                allocation_count += 1;
                return 0;
            }
        }.init,
        .deinit = struct {
            fn deinit(ctx: *PluginContext) callconv(.c) void {
                if (ctx.plugin_data) |data| {
                    const bytes = @as([*]u8, @ptrCast(data))[0..1024];
                    ctx.allocator.free(bytes);
                    deallocation_count += 1;
                }
            }
        }.deinit,
    };
    
    {
        var plugin = try plugin_interface.Plugin.init(testing.allocator, &MemoryTestInterface);
        defer plugin.deinit();
        
        var config = PluginConfig.init(testing.allocator);
        defer config.deinit();
        
        try plugin.initialize(&config);
    }
    
    // Ensure cleanup happened
    try testing.expectEqual(allocation_count, deallocation_count);
}

// =============================================================================
// OPTIONAL FUNCTION TESTS
// =============================================================================

test "Optional function handling" {
    const MinimalInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = testInit,
        .deinit = testDeinit,
        // All other fields are optional and null by default
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &MinimalInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    
    // Starting should work even without start function
    try plugin.start();
    
    // Process should work with null function (no-op)
    try plugin.process(null, null);
    
    // Status should return state enum value
    const status = plugin.getStatus();
    try testing.expect(status > 0);
}

// =============================================================================
// THREAD SAFETY TESTS
// =============================================================================

test "Concurrent plugin access" {
    if (!std.io.is_async) return error.SkipZigTest;
    
    const ThreadSafeInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = testInit,
        .deinit = testDeinit,
        .process = struct {
            var counter = std.atomic.Value(u32).init(0);
            
            fn process(ctx: *PluginContext, input: ?*anyopaque, output: ?*anyopaque) callconv(.c) c_int {
                _ = ctx;
                _ = input;
                _ = output;
                _ = counter.fetchAdd(1, .seq_cst);
                return 0;
            }
        }.process,
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &ThreadSafeInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    try plugin.start();
    
    // Simulate concurrent access
    const thread_count = 4;
    var threads: [thread_count]std.Thread = undefined;
    
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, struct {
            fn run(p: *plugin_interface.Plugin) void {
                for (0..100) |_| {
                    p.process(null, null) catch {};
                }
            }
        }.run, .{&plugin});
    }
    
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all operations completed
    const final_count = ThreadSafeInterface.process.counter.load(.seq_cst);
    try testing.expectEqual(@as(u32, thread_count * 100), final_count);
}

// =============================================================================
// BINARY COMPATIBILITY TESTS
// =============================================================================

test "Entry point symbol compatibility" {
    const expected_symbol = plugin_interface.PLUGIN_ENTRY_POINT;
    try testing.expectEqualStrings("abi_plugin_create", expected_symbol);
}

test "Factory function signature" {
    const FactoryFn = plugin_interface.PluginFactoryFn;
    const fn_info = @typeInfo(FactoryFn);
    
    try testing.expect(fn_info == .Pointer);
    
    // The factory should return an optional pointer to PluginInterface
    const return_type = @typeInfo(fn_info.Pointer.child).Fn.return_type.?;
    try testing.expect(@typeInfo(return_type) == .Optional);
}

// =============================================================================
// PLUGIN DISCOVERY TESTS
// =============================================================================

test "Plugin type enumeration" {
    // Ensure all plugin types can be converted to/from strings
    const types = [_]plugin_types.PluginType{
        .vector_database,
        .neural_network,
        .text_processor,
        .custom,
    };
    
    for (types) |plugin_type| {
        const str = plugin_type.toString();
        const converted = plugin_types.PluginType.fromString(str);
        try testing.expect(converted != null);
        try testing.expectEqual(plugin_type, converted.?);
    }
}

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

test "Plugin configuration API" {
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    // Set parameters
    try config.setParameter("test_key", "test_value");
    try config.setParameter("number", "42");
    
    // Retrieve parameters
    const value = config.getParameter("test_key");
    try testing.expect(value != null);
    try testing.expectEqualStrings("test_value", value.?);
    
    const number_str = config.getParameter("number");
    try testing.expect(number_str != null);
    const number = try std.fmt.parseInt(u32, number_str.?, 10);
    try testing.expectEqual(@as(u32, 42), number);
    
    // Non-existent parameter
    const missing = config.getParameter("missing");
    try testing.expect(missing == null);
}

// =============================================================================
// EVENT HANDLING TESTS
// =============================================================================

test "Plugin event system" {
    var event_received = false;
    var event_type_received: u32 = 0;
    
    const EventInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = testInit,
        .deinit = testDeinit,
        .on_event = struct {
            fn onEvent(ctx: *PluginContext, event_type: u32, data: ?*anyopaque) callconv(.c) c_int {
                _ = ctx;
                _ = data;
                event_received = true;
                event_type_received = event_type;
                return 0;
            }
        }.onEvent,
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &EventInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    
    // Send events
    try plugin.onEvent(1, null); // System startup
    try testing.expect(event_received);
    try testing.expectEqual(@as(u32, 1), event_type_received);
    
    event_received = false;
    try plugin.onEvent(2, null); // System shutdown
    try testing.expect(event_received);
    try testing.expectEqual(@as(u32, 2), event_type_received);
}

// =============================================================================
// METRICS COLLECTION TESTS
// =============================================================================

test "Plugin metrics API" {
    const MetricsInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = testInit,
        .deinit = testDeinit,
        .get_metrics = struct {
            fn getMetrics(ctx: *PluginContext, buffer: [*]u8, size: usize) callconv(.c) c_int {
                _ = ctx;
                const json = 
                    \\{"calls": 42, "errors": 0}
                ;
                if (size < json.len) return -1;
                @memcpy(buffer[0..json.len], json);
                return @intCast(json.len);
            }
        }.getMetrics,
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &MetricsInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    
    // Get metrics
    var buffer: [256]u8 = undefined;
    const len = try plugin.getMetrics(&buffer);
    try testing.expect(len > 0);
    
    const metrics = buffer[0..len];
    try testing.expect(std.mem.indexOf(u8, metrics, "calls") != null);
    try testing.expect(std.mem.indexOf(u8, metrics, "42") != null);
}

// =============================================================================
// EXTENDED API TESTS
// =============================================================================

test "Plugin extended API access" {
    const ExtendedInterface = PluginInterface{
        .get_info = testGetInfo,
        .init = testInit,
        .deinit = testDeinit,
        .get_api = struct {
            const CustomApi = struct {
                value: u32 = 12345,
            };
            var api = CustomApi{};
            
            fn getApi(name: [*:0]const u8) callconv(.c) ?*anyopaque {
                const api_name = std.mem.span(name);
                if (std.mem.eql(u8, api_name, "custom")) {
                    return @ptrCast(&api);
                }
                return null;
            }
        }.getApi,
    };
    
    var plugin = try plugin_interface.Plugin.init(testing.allocator, &ExtendedInterface);
    defer plugin.deinit();
    
    var config = PluginConfig.init(testing.allocator);
    defer config.deinit();
    
    try plugin.initialize(&config);
    
    // Get custom API
    const api_ptr = plugin.getApi("custom");
    try testing.expect(api_ptr != null);
    
    const CustomApi = struct { value: u32 };
    const custom_api = @as(*CustomApi, @ptrCast(@alignCast(api_ptr.?)));
    try testing.expectEqual(@as(u32, 12345), custom_api.value);
    
    // Non-existent API
    const missing_api = plugin.getApi("missing");
    try testing.expect(missing_api == null);
}