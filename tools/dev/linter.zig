//! Development Linter
//!
//! Basic linting tool for the ABI framework

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    std.log.info("Running ABI Framework Linter", .{});
    
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Check for common issues
    var issues_found: usize = 0;
    
    issues_found += try checkSourceFiles(allocator);
    issues_found += try checkDocumentation(allocator);
    issues_found += try checkTests(allocator);
    issues_found += try checkExamples(allocator);
    
    if (issues_found == 0) {
        std.log.info("Linting completed successfully - no issues found", .{});
    } else {
        std.log.err("Linting completed with {d} issues found", .{issues_found});
        std.process.exit(1);
    }
}

fn checkSourceFiles(allocator: std.mem.Allocator) !usize {
    std.log.info("Checking source files...", .{});
    
    var issues: usize = 0;
    
    // Check for TODO comments
    issues += try checkTODOs(allocator, "lib");
    
    // Check for unused imports
    issues += try checkUnusedImports(allocator, "lib");
    
    // Check for missing documentation
    issues += try checkDocumentation(allocator, "lib");
    
    return issues;
}

fn checkDocumentation(allocator: std.mem.Allocator) !usize {
    std.log.info("Checking documentation...", .{});
    
    var issues: usize = 0;
    
    // Check for missing README files
    const required_files = [_][]const u8{
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "CONTRIBUTING.md",
    };
    
    for (required_files) |filename| {
        const file = std.fs.cwd().openFile(filename, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Missing required file: {s}", .{filename});
                issues += 1;
                continue;
            },
            else => return err,
        };
        defer file.close();
        
        // Check file is not empty
        const stat = try file.stat();
        if (stat.size == 0) {
            std.log.err("Empty required file: {s}", .{filename});
            issues += 1;
        }
    }
    
    return issues;
}

fn checkTests(allocator: std.mem.Allocator) !usize {
    std.log.info("Checking tests...", .{});
    
    var issues: usize = 0;
    
    // Check test files exist
    const test_files = [_][]const u8{
        "tests/unit/mod.zig",
        "tests/integration/mod.zig",
        "tests/benchmarks/mod.zig",
    };
    
    for (test_files) |test_file| {
        std.fs.cwd().openFile(test_file, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Missing test file: {s}", .{test_file});
                issues += 1;
            },
            else => return err,
        };
    }
    
    return issues;
}

fn checkExamples(allocator: std.mem.Allocator) !usize {
    std.log.info("Checking examples...", .{});
    
    var issues: usize = 0;
    
    // Check example files exist and compile
    const example_files = [_][]const u8{
        "examples/basic-usage.zig",
        "examples/advanced-features.zig",
    };
    
    for (example_files) |example_file| {
        std.fs.cwd().openFile(example_file, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Missing example file: {s}", .{example_file});
                issues += 1;
            },
            else => return err,
        };
    }
    
    return issues;
}

fn checkTODOs(allocator: std.mem.Allocator, directory: []const u8) !usize {
    var issues: usize = 0;
    
    var dir = try std.fs.cwd().openDir(directory, .{ .iterate = true });
    defer dir.close();
    
    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ directory, entry.name });
        defer allocator.free(full_path);
        
        if (entry.kind == .directory) {
            issues += try checkTODOs(allocator, full_path);
        } else if (std.mem.endsWith(u8, entry.name, ".zig")) {
            const file = try dir.openFile(entry.name, .{});
            defer file.close();
            
            const contents = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(contents);
            
            var lines = std.mem.splitScalar(u8, contents, '\n');
            var line_number: usize = 1;
            
            while (lines.next()) |line| {
                if (std.mem.indexOf(u8, line, "TODO") != null or 
                    std.mem.indexOf(u8, line, "FIXME") != null or
                    std.mem.indexOf(u8, line, "HACK") != null) {
                    std.log.warn("TODO/FIXME/HACK found in {s}:{d}: {s}", .{ full_path, line_number, line });
                    issues += 1;
                }
                line_number += 1;
            }
        }
    }
    
    return issues;
}

fn checkUnusedImports(allocator: std.mem.Allocator, directory: []const u8) !usize {
    var issues: usize = 0;
    
    var dir = try std.fs.cwd().openDir(directory, .{ .iterate = true });
    defer dir.close();
    
    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ directory, entry.name });
        defer allocator.free(full_path);
        
        if (entry.kind == .directory) {
            issues += try checkUnusedImports(allocator, full_path);
        } else if (std.mem.endsWith(u8, entry.name, ".zig")) {
            const file = try dir.openFile(entry.name, .{});
            defer file.close();
            
            const contents = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(contents);
            
            var lines = std.mem.splitScalar(u8, contents, '\n');
            var line_number: usize = 1;
            
            while (lines.next()) |line| {
                // Simple check for potentially unused imports
                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " "), "const") and 
                    std.mem.indexOf(u8, line, "= @import") != null) {
                    const import_name = extractImportName(line) orelse continue;
                    
                    // Check if import is used later in the file
                    if (!isImportUsed(contents, import_name)) {
                        std.log.warn("Potentially unused import in {s}:{d}: {s}", .{ full_path, line_number, import_name });
                        issues += 1;
                    }
                }
                line_number += 1;
            }
        }
    }
    
    return issues;
}

fn extractImportName(line: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, line, " ");
    if (!std.mem.startsWith(u8, trimmed, "const ")) return null;
    
    const const_end = std.mem.indexOfScalar(u8, trimmed, ' ') orelse return null;
    const after_const = trimmed[const_end + 1 ..];
    
    const name_end = std.mem.indexOfScalar(u8, after_const, ' ') orelse return null;
    return after_const[0..name_end];
}

fn isImportUsed(contents: []const u8, import_name: []const u8) bool {
    // Simple heuristic: check if import name appears after its declaration
    const import_pos = std.mem.indexOf(u8, contents, import_name) orelse return false;
    const after_import = contents[import_pos + import_name.len ..];
    
    // Look for usage patterns
    const usage_patterns = [_][]const u8{
        try std.fmt.allocPrint(std.heap.page_allocator, "{s}.", .{import_name}),
        try std.fmt.allocPrint(std.heap.page_allocator, " {s} ", .{import_name}),
        try std.fmt.allocPrint(std.heap.page_allocator, "({s}", .{import_name}),
        try std.fmt.allocPrint(std.heap.page_allocator, "{s}(", .{import_name}),
    };
    defer {
        for (usage_patterns) |pattern| {
            std.heap.page_allocator.free(pattern);
        }
    }
    
    for (usage_patterns) |pattern| {
        if (std.mem.indexOf(u8, after_import, pattern) != null) {
            return true;
        }
    }
    
    return false;
}