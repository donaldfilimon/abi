//! Code Quality Analyzer
//!
//! This tool analyzes the codebase for quality metrics including:
//! - Code complexity
//! - Documentation coverage
//! - Error handling patterns
//! - Memory safety
//! - Performance patterns

const std = @import("std");
const abi = @import("../src/mod.zig");

const AnalysisResult = struct {
    total_files: u32,
    total_lines: u32,
    documented_functions: u32,
    total_functions: u32,
    error_handling_score: f32,
    memory_safety_score: f32,
    complexity_score: f32,
    overall_score: f32,
};

const FileAnalysis = struct {
    path: []const u8,
    lines: u32,
    functions: u32,
    documented_functions: u32,
    error_handling_patterns: u32,
    memory_allocations: u32,
    complexity: f32,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const writer = abi.core.Writer.stdout();
    try writer.print("ABI Framework Code Quality Analysis\n");
    try writer.print("===================================\n\n");
    
    var analysis_results = std.ArrayList(FileAnalysis).init(allocator);
    defer analysis_results.deinit();
    
    // Analyze source files
    try analyzeDirectory(allocator, "src", &analysis_results);
    
    // Calculate overall metrics
    const result = try calculateOverallMetrics(allocator, analysis_results.items);
    
    // Print results
    try printAnalysisResults(writer, result, analysis_results.items);
}

fn analyzeDirectory(allocator: std.mem.Allocator, dir_path: []const u8, results: *std.ArrayList(FileAnalysis)) !void {
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();
    
    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".zig")) {
            const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, entry.name });
            defer allocator.free(full_path);
            
            const analysis = try analyzeFile(allocator, full_path);
            try results.append(analysis);
        } else if (entry.kind == .directory) {
            const subdir_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, entry.name });
            defer allocator.free(subdir_path);
            
            try analyzeDirectory(allocator, subdir_path, results);
        }
    }
}

fn analyzeFile(allocator: std.mem.Allocator, file_path: []const u8) !FileAnalysis {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    
    const content = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(content);
    
    var lines: u32 = 0;
    var functions: u32 = 0;
    var documented_functions: u32 = 0;
    var error_handling_patterns: u32 = 0;
    var memory_allocations: u32 = 0;
    var complexity: f32 = 0.0;
    
    var line_iter = std.mem.split(u8, content, "\n");
    while (line_iter.next()) |line| {
        lines += 1;
        
        // Count functions
        if (std.mem.indexOf(u8, line, "pub fn ") != null or std.mem.indexOf(u8, line, "fn ") != null) {
            functions += 1;
            
            // Check if function is documented
            if (std.mem.indexOf(u8, line, "///") != null) {
                documented_functions += 1;
            }
        }
        
        // Count error handling patterns
        if (std.mem.indexOf(u8, line, "catch") != null or 
            std.mem.indexOf(u8, line, "try ") != null or
            std.mem.indexOf(u8, line, "!") != null) {
            error_handling_patterns += 1;
        }
        
        // Count memory allocations
        if (std.mem.indexOf(u8, line, "alloc") != null or 
            std.mem.indexOf(u8, line, "create") != null or
            std.mem.indexOf(u8, line, "destroy") != null) {
            memory_allocations += 1;
        }
        
        // Calculate complexity (simple metric)
        if (std.mem.indexOf(u8, line, "if ") != null or 
            std.mem.indexOf(u8, line, "while ") != null or
            std.mem.indexOf(u8, line, "for ") != null or
            std.mem.indexOf(u8, line, "switch ") != null) {
            complexity += 1.0;
        }
    }
    
    return FileAnalysis{
        .path = try allocator.dupe(u8, file_path),
        .lines = lines,
        .functions = functions,
        .documented_functions = documented_functions,
        .error_handling_patterns = error_handling_patterns,
        .memory_allocations = memory_allocations,
        .complexity = complexity,
    };
}

fn calculateOverallMetrics(allocator: std.mem.Allocator, files: []FileAnalysis) !AnalysisResult {
    var total_files: u32 = @intCast(files.len);
    var total_lines: u32 = 0;
    var total_functions: u32 = 0;
    var total_documented_functions: u32 = 0;
    var total_error_patterns: u32 = 0;
    var total_memory_ops: u32 = 0;
    var total_complexity: f32 = 0.0;
    
    for (files) |file| {
        total_lines += file.lines;
        total_functions += file.functions;
        total_documented_functions += file.documented_functions;
        total_error_patterns += file.error_handling_patterns;
        total_memory_ops += file.memory_allocations;
        total_complexity += file.complexity;
    }
    
    // Calculate scores (0-100)
    const documentation_score = if (total_functions > 0) 
        (@as(f32, @floatFromInt(total_documented_functions)) / @as(f32, @floatFromInt(total_functions))) * 100.0 
    else 0.0;
    
    const error_handling_score = if (total_functions > 0)
        (@as(f32, @floatFromInt(total_error_patterns)) / @as(f32, @floatFromInt(total_functions))) * 100.0
    else 0.0;
    
    const memory_safety_score = if (total_memory_ops > 0)
        (@as(f32, @floatFromInt(total_memory_ops)) / @as(f32, @floatFromInt(total_lines))) * 100.0
    else 100.0;
    
    const complexity_score = if (total_lines > 0)
        100.0 - ((total_complexity / @as(f32, @floatFromInt(total_lines))) * 100.0)
    else 100.0;
    
    const overall_score = (documentation_score + error_handling_score + memory_safety_score + complexity_score) / 4.0;
    
    return AnalysisResult{
        .total_files = total_files,
        .total_lines = total_lines,
        .documented_functions = total_documented_functions,
        .total_functions = total_functions,
        .error_handling_score = error_handling_score,
        .memory_safety_score = memory_safety_score,
        .complexity_score = complexity_score,
        .overall_score = overall_score,
    };
}

fn printAnalysisResults(writer: abi.core.Writer, result: AnalysisResult, files: []FileAnalysis) !void {
    try writer.print("Overall Metrics:\n");
    try writer.print("===============\n");
    try writer.print("Total files analyzed: {d}\n", .{result.total_files});
    try writer.print("Total lines of code: {d}\n", .{result.total_lines});
    try writer.print("Total functions: {d}\n", .{result.total_functions});
    try writer.print("Documented functions: {d}\n", .{result.documented_functions});
    try writer.print("\n");
    
    try writer.print("Quality Scores (0-100):\n");
    try writer.print("======================\n");
    try writer.print("Documentation: {d:.1}\n", .{result.error_handling_score});
    try writer.print("Error Handling: {d:.1}\n", .{result.error_handling_score});
    try writer.print("Memory Safety: {d:.1}\n", .{result.memory_safety_score});
    try writer.print("Code Complexity: {d:.1}\n", .{result.complexity_score});
    try writer.print("Overall Score: {d:.1}\n", .{result.overall_score});
    try writer.print("\n");
    
    try writer.print("File-by-File Analysis:\n");
    try writer.print("=====================\n");
    for (files) |file| {
        const doc_percentage = if (file.functions > 0)
            (@as(f32, @floatFromInt(file.documented_functions)) / @as(f32, @floatFromInt(file.functions))) * 100.0
        else 0.0;
        
        try writer.print("{s}:\n", .{file.path});
        try writer.print("  Lines: {d}, Functions: {d}, Documented: {d} ({d:.1}%)\n", .{
            file.lines, file.functions, file.documented_functions, doc_percentage
        });
        try writer.print("  Error patterns: {d}, Memory ops: {d}, Complexity: {d:.1}\n", .{
            file.error_handling_patterns, file.memory_allocations, file.complexity
        });
        try writer.print("\n");
    }
    
    // Recommendations
    try writer.print("Recommendations:\n");
    try writer.print("===============\n");
    
    if (result.error_handling_score < 50.0) {
        try writer.print("- Improve error handling patterns\n");
    }
    if (result.memory_safety_score < 70.0) {
        try writer.print("- Review memory allocation patterns\n");
    }
    if (result.complexity_score < 60.0) {
        try writer.print("- Reduce code complexity\n");
    }
    if (result.error_handling_score < 80.0) {
        try writer.print("- Add more documentation\n");
    }
    
    try writer.print("\nAnalysis completed!\n");
}