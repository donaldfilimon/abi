//! Plugin Validator Tool
//!
//! This tool validates that a plugin conforms to the Abi AI Framework ABI

const std = @import("std");
const plugin_interface = @import("../src/plugins/interface.zig");
const plugin_types = @import("../src/plugins/types.zig");
const plugin_loader = @import("../src/plugins/loader.zig");

const PluginInterface = plugin_interface.PluginInterface;
const PluginInfo = plugin_types.PluginInfo;
const PLUGIN_ABI_VERSION = plugin_interface.PLUGIN_ABI_VERSION;

const ValidationResult = struct {
    passed: bool,
    errors: std.ArrayList([]const u8),
    warnings: std.ArrayList([]const u8),
    info: std.ArrayList([]const u8),
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const plugin_path = args[1];
    const verbose = if (args.len > 2) std.mem.eql(u8, args[2], "--verbose") else false;

    std.debug.print("🔍 Validating plugin: {s}\n\n", .{plugin_path});

    var result = ValidationResult{
        .passed = true,
        .errors = std.ArrayList([]const u8).init(allocator),
        .warnings = std.ArrayList([]const u8).init(allocator),
        .info = std.ArrayList([]const u8).init(allocator),
    };
    defer {
        result.errors.deinit();
        result.warnings.deinit();
        result.info.deinit();
    }

    // Load the plugin
    const lib = std.DynLib.open(plugin_path) catch |err| {
        try result.errors.append(try std.fmt.allocPrint(allocator, "Failed to load plugin: {}", .{err}));
        result.passed = false;
        try printResults(result, verbose);
        return;
    };
    defer lib.close();

    // Check for entry point
    const entry_symbol = plugin_interface.PLUGIN_ENTRY_POINT;
    const factory_fn = lib.lookup(plugin_interface.PluginFactoryFn, entry_symbol) orelse {
        try result.errors.append(try std.fmt.allocPrint(allocator, "Missing entry point: {s}", .{entry_symbol}));
        result.passed = false;
        try printResults(result, verbose);
        return;
    };

    try result.info.append("✓ Found plugin entry point");

    // Get plugin interface
    const interface_ptr = factory_fn();
    if (interface_ptr == null) {
        try result.errors.append("Plugin factory returned null");
        result.passed = false;
        try printResults(result, verbose);
        return;
    }

    const interface = interface_ptr.?;
    try result.info.append("✓ Plugin factory returned valid interface");

    // Validate required functions
    if (!validateRequiredFunctions(interface, &result, allocator)) {
        result.passed = false;
    }

    // Get and validate plugin info
    const info = interface.get_info();
    try validatePluginInfo(info, &result, allocator);

    // Check ABI compatibility
    if (!info.abi_version.isCompatible(PLUGIN_ABI_VERSION)) {
        try result.errors.append(try std.fmt.allocPrint(allocator, 
            "Incompatible ABI version: plugin requires {}, framework provides {}", 
            .{ info.abi_version, PLUGIN_ABI_VERSION }));
        result.passed = false;
    } else {
        try result.info.append(try std.fmt.allocPrint(allocator, 
            "✓ ABI version compatible: {}", .{info.abi_version}));
    }

    // Test basic lifecycle
    if (verbose) {
        try testPluginLifecycle(interface, &result, allocator);
    }

    try printResults(result, verbose);
}

fn printUsage() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print(
        \\Plugin Validator - Validate Abi AI Framework plugins
        \\
        \\Usage: plugin_validator <plugin_path> [--verbose]
        \\
        \\Arguments:
        \\  plugin_path    Path to the plugin shared library (.so/.dll/.dylib)
        \\  --verbose      Run additional tests and show detailed output
        \\
        \\Example:
        \\  plugin_validator ./plugins/example_plugin.so --verbose
        \\
    , .{});
}

fn validateRequiredFunctions(interface: *const PluginInterface, result: *ValidationResult, allocator: std.mem.Allocator) bool {
    var valid = true;

    // Check required function pointers
    if (@intFromPtr(interface.get_info) == 0) {
        result.errors.append(allocator.dupe(u8, "Missing required function: get_info") catch {};
        valid = false;
    } else {
        result.info.append(allocator.dupe(u8, "✓ Found get_info function") catch {};
    }

    if (@intFromPtr(interface.init) == 0) {
        result.errors.append(allocator.dupe(u8, "Missing required function: init") catch {};
        valid = false;
    } else {
        result.info.append(allocator.dupe(u8, "✓ Found init function") catch {};
    }

    if (@intFromPtr(interface.deinit) == 0) {
        result.errors.append(allocator.dupe(u8, "Missing required function: deinit") catch {};
        valid = false;
    } else {
        result.info.append(allocator.dupe(u8, "✓ Found deinit function") catch {};
    }

    // Check optional functions
    var optional_count: u32 = 0;
    if (interface.start != null) optional_count += 1;
    if (interface.stop != null) optional_count += 1;
    if (interface.pause != null) optional_count += 1;
    if (interface.plugin_resume != null) optional_count += 1;
    if (interface.process != null) optional_count += 1;
    if (interface.configure != null) optional_count += 1;
    if (interface.get_config != null) optional_count += 1;
    if (interface.get_status != null) optional_count += 1;
    if (interface.get_metrics != null) optional_count += 1;
    if (interface.on_event != null) optional_count += 1;
    if (interface.get_api != null) optional_count += 1;

    result.info.append(std.fmt.allocPrint(allocator, "ℹ️  Plugin implements {d} optional functions", .{optional_count}) catch {}) catch {};

    return valid;
}

fn validatePluginInfo(info: *const PluginInfo, result: *ValidationResult, allocator: std.mem.Allocator) !void {
    // Validate name
    if (info.name.len == 0) {
        try result.errors.append(try allocator.dupe(u8, "Plugin name is empty"));
    } else if (info.name.len > 128) {
        try result.warnings.append(try std.fmt.allocPrint(allocator, "Plugin name is very long: {d} characters", .{info.name.len}));
    } else {
        try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin name: {s}", .{info.name}));
    }

    // Validate version
    try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin version: {}", .{info.version}));

    // Validate author
    if (info.author.len == 0) {
        try result.warnings.append(try allocator.dupe(u8, "Plugin author is not specified"));
    } else {
        try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin author: {s}", .{info.author}));
    }

    // Validate description
    if (info.description.len == 0) {
        try result.warnings.append(try allocator.dupe(u8, "Plugin description is empty"));
    }

    // Validate plugin type
    try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin type: {s}", .{info.plugin_type.toString()}));

    // Check dependencies
    if (info.dependencies.len > 0) {
        try result.info.append(try std.fmt.allocPrint(allocator, "ℹ️  Plugin has {d} dependencies:", .{info.dependencies.len}));
        for (info.dependencies) |dep| {
            try result.info.append(try std.fmt.allocPrint(allocator, "    - {s}", .{dep}));
        }
    }

    // Check provided capabilities
    if (info.provides.len > 0) {
        try result.info.append(try std.fmt.allocPrint(allocator, "ℹ️  Plugin provides {d} capabilities:", .{info.provides.len}));
        for (info.provides) |capability| {
            try result.info.append(try std.fmt.allocPrint(allocator, "    - {s}", .{capability}));
        }
    }

    // Check requirements
    if (info.requires.len > 0) {
        try result.info.append(try std.fmt.allocPrint(allocator, "ℹ️  Plugin requires {d} services:", .{info.requires.len}));
        for (info.requires) |requirement| {
            try result.info.append(try std.fmt.allocPrint(allocator, "    - {s}", .{requirement}));
        }
    }
}

fn testPluginLifecycle(interface: *const PluginInterface, result: *ValidationResult, allocator: std.mem.Allocator) !void {
    try result.info.append("\n🧪 Testing plugin lifecycle...");

    // Create a test context
    var config = plugin_types.PluginConfig.init(allocator);
    defer config.deinit();

    var context = plugin_types.PluginContext{
        .allocator = allocator,
        .config = &config,
    };

    // Test initialization
    const init_result = interface.init(&context);
    if (init_result != 0) {
        try result.warnings.append(try std.fmt.allocPrint(allocator, "Plugin init returned error code: {d}", .{init_result}));
    } else {
        try result.info.append("✓ Plugin initialized successfully");
    }

    // Test status if available
    if (interface.get_status) |get_status| {
        const status = get_status(&context);
        try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin status: {d}", .{status}));
    }

    // Test metrics if available
    if (interface.get_metrics) |get_metrics| {
        var buffer: [1024]u8 = undefined;
        const metrics_len = get_metrics(&context, &buffer, buffer.len);
        if (metrics_len > 0) {
            const metrics = buffer[0..@intCast(metrics_len)];
            try result.info.append(try std.fmt.allocPrint(allocator, "✓ Plugin metrics: {s}", .{metrics}));
        }
    }

    // Clean up
    interface.deinit(&context);
    try result.info.append("✓ Plugin deinitialized successfully");
}

fn printResults(result: ValidationResult, verbose: bool) !void {
    const stdout = std.io.getStdOut().writer();

    // Print info messages if verbose
    if (verbose) {
        for (result.info.items) |msg| {
            try stdout.print("{s}\n", .{msg});
        }
        if (result.info.items.len > 0) {
            try stdout.print("\n", .{});
        }
    }

    // Print warnings
    if (result.warnings.items.len > 0) {
        try stdout.print("⚠️  Warnings:\n", .{});
        for (result.warnings.items) |warning| {
            try stdout.print("   - {s}\n", .{warning});
        }
        try stdout.print("\n", .{});
    }

    // Print errors
    if (result.errors.items.len > 0) {
        try stdout.print("❌ Errors:\n", .{});
        for (result.errors.items) |err| {
            try stdout.print("   - {s}\n", .{err});
        }
        try stdout.print("\n", .{});
    }

    // Print final result
    if (result.passed) {
        try stdout.print("✅ Plugin validation PASSED\n", .{});
    } else {
        try stdout.print("❌ Plugin validation FAILED\n", .{});
        std.process.exit(1);
    }
}