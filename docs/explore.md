# ABI Explore Module

A powerful codebase exploration and search tool with natural language query understanding, AST parsing, and multiple output formats.

## Overview

The explore module provides intelligent search capabilities for codebases with:

- **Natural Language Queries**: Understand queries like "find all HTTP handlers" or "show me test functions"
- **Multiple Search Levels**: From quick filename search to deep semantic analysis
- **Pattern Matching**: Literal, glob, regex, and fuzzy matching
- **AST Parsing**: Extract functions, types, imports, and more from source files
- **Multiple Output Formats**: Human-readable, JSON, YAML, or compact output

## Quick Start

### CLI Usage

```bash
# Basic search
abi explore "HTTP handler"

# Search with specific level
abi explore -l thorough "TODO"

# Search with output format
abi explore -f json "pub fn"

# Include specific file types
abi explore -i "*.zig" "pub const"

# Exclude patterns
abi explore -e "test" "handler"

# Use regex patterns
abi explore -r "fn\s+\w+"
```

### Library Usage

```zig
const explore = abi.ai.explore;

var agent = explore.ExploreAgent.init(allocator, explore.ExploreConfig.defaultForLevel(.medium));
defer agent.deinit();

const result = try agent.explore(".", "my search query");
defer result.deinit();

// Print results
try result.formatHuman(std.debug);
```

## Explore Levels

| Level | Max Files | Max Depth | Timeout | Use Case |
|-------|-----------|-----------|---------|----------|
| `quick` | 1,000 | 3 | 10s | Fast filename search |
| `medium` | 5,000 | 10 | 30s | Balanced search |
| `thorough` | 10,000 | 20 | 60s | Full content analysis |
| `deep` | 50,000 | 50 | 5min | Complete codebase scan |

## Configuration

```zig
const config = explore.ExploreConfig{
    .level = .medium,
    .max_files = 5000,
    .max_depth = 10,
    .timeout_ms = 30000,
    .case_sensitive = false,
    .use_regex = false,
    .parallel_io = true,
    .include_patterns = &.{ "*.zig", "*.md" },
    .exclude_patterns = &.{ "*.git", "node_modules" },
    .file_size_limit_bytes = 1024 * 1024,
};
```

## Natural Language Query Understanding

The explore module can understand natural language queries:

```zig
var understander = explore.QueryUnderstanding.init(allocator);
defer understander.deinit();

const parsed = try understander.parse("find all HTTP handlers in src/api/");
// parsed.intent = .find_functions
// parsed.patterns = &.{"handler"}
// parsed.target_paths = &.{"src/api"}
// parsed.file_extensions = &.{".zig"}
```

### Supported Intent Types

- `find_functions` - Search for function definitions
- `find_types` - Search for type definitions (structs, enums, etc.)
- `find_tests` - Search for test cases
- `find_imports` - Search for import statements
- `find_comments` - Search for comments and TODOs
- `find_configs` - Search for configuration
- `find_docs` - Search for documentation
- `list_files` - List files in paths
- `count_occurrences` - Count pattern occurrences
- `analyze_structure` - Analyze code structure

## Pattern Matching

### Literal Search

```zig
const pattern = try compiler.compile("my_function", .literal, false);
```

### Glob Patterns

```zig
const pattern = try compiler.compile("*.zig", .glob, false);
const pattern = try compiler.compile("src/**/*.test.zig", .glob, false);
```

### Regex Patterns

```zig
const pattern = try compiler.compile("fn\\s+\\w+", .regex, false);
```

### Fuzzy Matching

```zig
const pattern = try compiler.compile("hndlr", .fuzzy, false);
// Matches: "handler", "handlr", "hnldr"
```

## AST Parsing

Parse source code to extract code elements:

```zig
var parser = explore.AstParser.init(allocator);
defer parser.deinit();

const content = try std.fs.cwd().readFileAlloc(allocator, "src/main.zig", 1024 * 1024);
defer allocator.free(content);

const file_stat = try fs.getFileStats("src/main.zig");
const parsed = try parser.parseFile(&file_stat, content);
defer parsed.deinit();

// Access parsed elements
for (parsed.functions.items) |fn_name| {
    std.debug.print("Found function: {s}\n", .{fn_name});
}

for (parsed.imports.items) |imp| {
    std.debug.print("Found import: {s}\n", .{imp});
}

for (parsed.types.items) |type_name| {
    std.debug.print("Found type: {s}\n", .{type_name});
}
```

### Supported Languages

- **Zig**: Functions, structs, enums, const, imports, tests
- **Rust**: Functions, structs, enums, use imports, tests
- **TypeScript/JavaScript**: Functions, classes, interfaces, imports
- **Generic**: Functions, comments (works for any language)

## Output Formats

### Human Readable (Default)

```
Exploration Results for: "handler"
Level: medium
Files Scanned: 45
Matches Found: 12
Duration: 234ms

Top Matches:
-------------
1. src/http/server.zig:42
   pub fn handleRequest(
   Score: 0.85
```

### JSON

```json
{
  "query": "handler",
  "level": "medium",
  "files_scanned": 45,
  "matches_found": 12,
  "duration_ms": 234,
  "matches": [
    {
      "file": "src/http/server.zig",
      "line": 42,
      "type": "function_definition",
      "text": "pub fn handleRequest(",
      "score": 0.85
    }
  ]
}
```

### YAML

```yaml
query: "handler"
level: medium
matches_found: 12
duration_ms: 234
```

### Compact

```
Query: "handler" | Found: 12 matches in 234ms
```

## CLI Options

```
Usage: abi explore [options] <query>

Arguments:
  <query>              Search pattern or natural language query

Options:
  -l, --level <level>  Exploration depth: quick, medium, thorough, deep
  -f, --format <fmt>   Output format: human, json, compact, yaml
  -i, --include <pat>  Include files matching pattern
  -e, --exclude <pat>  Exclude files matching pattern
  -c, --case-sensitive Match case sensitively
  -r, --regex          Treat query as regex pattern
  --path <path>        Root directory to search (default: .)
  --max-files <n>      Maximum files to scan
  --max-depth <n>      Maximum directory depth
  --timeout <ms>       Timeout in milliseconds
  -h, --help           Show this help message
```

## Examples

### Find all HTTP handlers

```bash
abi explore "HTTP handler"
abi explore -l thorough "handleRequest"
abi explore -i "*.zig" "pub fn"
```

### Find test functions

```bash
abi explore "test"
abi explore -l thorough "test case"
abi explore -i "_test.zig" ""
```

### Find configuration

```bash
abi explore "const CONFIG"
abi explore "pub const"
abi explore -e "test" "config"
```

### Find TODO comments

```bash
abi explore "TODO"
abi explore -l thorough "TODO FIXME"
abi explore -f json "fixme"
```

### Find imports

```bash
abi explore "import"
abi explore -l quick "use @import"
```

### Analyze code structure

```bash
abi explore -l thorough "analyze structure"
abi explore -l deep "list all functions"
```

## API Reference

### Main Types

- `ExploreAgent` - Main exploration agent
- `ExploreConfig` - Configuration for exploration
- `ExploreResult` - Results from exploration
- `ExploreLevel` - Exploration depth levels
- `OutputFormat` - Output format options
- `QueryUnderstanding` - Natural language query parser
- `AstParser` - Source code AST parser
- `ParallelExplorer` - Multi-threaded exploration
- `WorkItem` - Work item for parallel processing
- `SearchPattern` - Compiled search pattern
- `PatternType` - Pattern matching type (literal, glob, regex, fuzzy)
- `PatternCompiler` - Compiler for search patterns
- `AstNode` - AST node with type, name, location
- `ParsedQuery` - Parsed natural language query
- `QueryIntent` - Query classification (find_functions, find_types, etc.)

### Key Functions

```zig
// Create agents with default configurations
pub fn createDefaultAgent(allocator: std.mem.Allocator) ExploreAgent
pub fn createQuickAgent(allocator: std.mem.Allocator) ExploreAgent
pub fn createThoroughAgent(allocator: std.mem.Allocator) ExploreAgent

// Create parallel exploration agents
pub fn parallelExplore(
    allocator: std.mem.Allocator,
    root_path: []const u8,
    config: ExploreConfig,
    search_query: []const u8,
) !ExploreResult

// Explore methods
pub fn explore(self: *ExploreAgent, root_path: []const u8, query: []const u8) !ExploreResult
pub fn exploreWithPatterns(self: *ExploreAgent, root_path: []const u8, patterns: []const []const u8) !ExploreResult
pub fn exploreNaturalLanguage(self: *ExploreAgent, root_path: []const u8, nl_query: []const u8) !ExploreResult
```

### Parallel Exploration

The `parallelExplore` function provides high-performance multi-threaded exploration:

```zig
const allocator = std.testing.allocator;

const config = explore.ExploreConfig.defaultForLevel(.thorough);
const result = try explore.parallelExplore(allocator, ".", config, "my query");
defer result.deinit();

try result.formatHuman(std.debug);
```

Parallel exploration features:
- Automatic CPU detection and thread pool sizing
- Work chunking for balanced load distribution
- Thread-safe result aggregation with mutex locks
- Graceful fallback on thread spawn failure
- Cancellation support with `cancel()` method

## Performance Tips

1. **Use `quick` level for filename-only searches**
2. **Use `--max-files` and `--max-depth` to limit scope**
3. **Use `-i` to include only relevant file types**
4. **Use `-e` to exclude build artifacts and dependencies**
5. **Use `--timeout` to prevent long-running searches**
6. **Use `compact` format for scripting and parsing**

## Error Handling

```zig
const result = agent.explore(".", "query) catch |err| {
    switch (err) {
        error.PathNotFound => std.debug.print("Path not found\n", .{}),
        error.Timeout => std.debug.print("Search timed out\n", .{}),
        error.TooManyFiles => std.debug.print("Too many files\n", .{}),
        else => std.debug.print("Error: {}\n", .{err}),
    }
    return;
};
```
