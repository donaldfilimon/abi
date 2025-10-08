//! Benchmarks
//!
//! Performance benchmarks for the ABI framework

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    std.log.info("Running ABI Framework Benchmarks", .{});
    
    try benchmarkFrameworkInitialization();
    try benchmarkFeatureManagement();
    try benchmarkMemoryAllocation();
    try benchmarkComponentRegistration();
    try benchmarkRuntimeOperations();
    
    std.log.info("Benchmarks completed", .{});
}

fn benchmarkFrameworkInitialization() !void {
    std.log.info("Benchmarking framework initialization...", .{});
    
    const iterations = 1000;
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        var framework = try abi.createDefaultFramework(std.heap.page_allocator);
        defer framework.deinit();
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_time = total_time / iterations;
    
    std.log.info("Framework initialization: {}ns per iteration ({} total)", .{ avg_time, total_time });
}

fn benchmarkFeatureManagement() !void {
    std.log.info("Benchmarking feature management...", .{});
    
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    const iterations = 10000;
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        framework.enableFeature(.gpu);
        framework.disableFeature(.gpu);
        framework.enableFeature(.connectors);
        framework.disableFeature(.connectors);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_time = total_time / iterations;
    
    std.log.info("Feature management: {}ns per iteration ({} total)", .{ avg_time, total_time });
}

fn benchmarkMemoryAllocation() !void {
    std.log.info("Benchmarking memory allocation patterns...", .{});
    
    var tracked = abi.core.allocators.AllocatorFactory.createTracked(std.heap.page_allocator, 1024 * 1024);
    const allocator = tracked.allocator();
    
    const iterations = 1000;
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        var list = abi.core.utils.createArrayList(u32, allocator);
        defer list.deinit();
        
        // Allocate and populate list
        for (0..100) |i| {
            try list.append(@intCast(i));
        }
        
        // Access elements
        for (list.items) |item| {
            _ = item;
        }
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_time = total_time / iterations;
    
    const stats = tracked.getStats();
    std.log.info("Memory allocation: {}ns per iteration ({} total)", .{ avg_time, total_time });
    std.log.info("Memory stats: {} allocated, {} freed, {} peak", .{ 
        stats.bytes_allocated, 
        stats.bytes_freed, 
        stats.peak_usage 
    });
}

fn benchmarkComponentRegistration() !void {
    std.log.info("Benchmarking component registration...", .{});
    
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    const iterations = 1000;
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const component = abi.framework.Component{
            .name = std.fmt.allocPrint(std.testing.allocator, "component_{d}", .{i}) catch unreachable,
            .version = "1.0.0",
        };
        defer std.testing.allocator.free(@constCast(component.name));
        
        try framework.registerComponent(component);
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_time = total_time / iterations;
    
    std.log.info("Component registration: {}ns per iteration ({} total)", .{ avg_time, total_time });
    std.log.info("Registered {} components", .{framework.getStats().total_components});
}

fn benchmarkRuntimeOperations() !void {
    std.log.info("Benchmarking runtime operations...", .{});
    
    var framework = try abi.createDefaultFramework(std.testing.allocator);
    defer framework.deinit();
    
    // Register some components
    for (0..10) |i| {
        const component = abi.framework.Component{
            .name = std.fmt.allocPrint(std.testing.allocator, "bench_component_{d}", .{i}) catch unreachable,
            .version = "1.0.0",
        };
        defer std.testing.allocator.free(@constCast(component.name));
        
        try framework.registerComponent(component);
    }
    
    const iterations = 100;
    
    // Benchmark start/stop cycle
    const start_stop_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        try framework.start();
        framework.stop();
    }
    const start_stop_time = std.time.nanoTimestamp() - start_stop_start;
    const start_stop_avg = start_stop_time / iterations;
    
    std.log.info("Start/stop cycle: {}ns per iteration ({} total)", .{ start_stop_avg, start_stop_time });
    
    // Benchmark runtime with updates
    try framework.start();
    defer framework.stop();
    
    const update_start = std.time.nanoTimestamp();
    for (0..iterations * 10) |_| {
        framework.update(0.016); // ~60 FPS
    }
    const update_time = std.time.nanoTimestamp() - update_start;
    const update_avg = update_time / (iterations * 10);
    
    std.log.info("Runtime updates: {}ns per iteration ({} total)", .{ update_avg, update_time });
    
    // Benchmark stats collection
    const stats_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = framework.getStats();
    }
    const stats_time = std.time.nanoTimestamp() - stats_start;
    const stats_avg = stats_time / iterations;
    
    std.log.info("Stats collection: {}ns per iteration ({} total)", .{ stats_avg, stats_time });
}

fn benchmarkCollectionOperations() !void {
    std.log.info("Benchmarking collection operations...", .{});
    
    const iterations = 10000;
    
    // Benchmark ArrayList operations
    const list_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        var list = abi.core.utils.createArrayList(u32, std.testing.allocator);
        defer list.deinit();
        
        for (0..100) |i| {
            try list.append(@intCast(i));
        }
        
        // Search operations
        for (list.items) |item| {
            if (item == 50) break;
        }
    }
    const list_time = std.time.nanoTimestamp() - list_start;
    const list_avg = list_time / iterations;
    
    std.log.info("ArrayList operations: {}ns per iteration ({} total)", .{ list_avg, list_time });
    
    // Benchmark StringHashMap operations
    const map_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        var map = abi.core.utils.createStringHashMap(u32, std.testing.allocator);
        defer map.deinit();
        
        for (0..100) |i| {
            const key = std.fmt.allocPrint(std.testing.allocator, "key_{d}", .{i}) catch unreachable;
            defer std.testing.allocator.free(@constCast(key));
            
            try map.put(key, @intCast(i));
        }
        
        // Lookup operations
        const lookup_key = "key_50";
        _ = map.get(lookup_key);
    }
    const map_time = std.time.nanoTimestamp() - map_start;
    const map_avg = map_time / iterations;
    
    std.log.info("StringHashMap operations: {}ns per iteration ({} total)", .{ map_avg, map_time });
}

test "benchmark collection operations" {
    try benchmarkCollectionOperations();
}