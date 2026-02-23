//! AI explore stub — disabled at compile time.

const std = @import("std");
const stub_root = @This();

// --- Submodule Namespace Re-exports ---

pub const config = struct {
    pub const ExploreConfig = stub_root.ExploreConfig;
    pub const ExploreLevel = stub_root.ExploreLevel;
    pub const OutputFormat = stub_root.OutputFormat;
    pub const FileType = stub_root.FileType;
    pub const FileFilter = stub_root.FileFilter;
    pub const SearchScope = stub_root.SearchScope;
    pub const SearchOptions = stub_root.SearchOptions;
};
pub const results = struct {
    pub const ExploreResult = stub_root.ExploreResult;
    pub const Match = stub_root.Match;
    pub const MatchType = stub_root.MatchType;
    pub const ExploreError = stub_root.ExploreError;
    pub const ExplorationStats = stub_root.ExplorationStats;
};
pub const fs = struct {
    pub const FileVisitor = stub_root.FileVisitor;
    pub const FileStats = stub_root.FileStats;
};
pub const search = struct {
    pub const SearchPattern = stub_root.SearchPattern;
    pub const PatternType = stub_root.PatternType;
    pub const PatternCompiler = stub_root.PatternCompiler;
};
pub const agent = struct {
    pub const ExploreAgent = stub_root.ExploreAgent;
    pub const createDefaultAgent = stub_root.createDefaultAgent;
    pub const createQuickAgent = stub_root.createQuickAgent;
    pub const createThoroughAgent = stub_root.createThoroughAgent;
};
pub const query = struct {
    pub const QueryIntent = stub_root.QueryIntent;
    pub const ParsedQuery = stub_root.ParsedQuery;
    pub const QueryUnderstanding = stub_root.QueryUnderstanding;
};
pub const ast = struct {
    pub const AstNode = stub_root.AstNode;
    pub const AstNodeType = stub_root.AstNodeType;
    pub const ParsedFile = stub_root.ParsedFile;
    pub const AstParser = stub_root.AstParser;
};
pub const parallel = struct {
    pub const ParallelExplorer = stub_root.ParallelExplorer;
    pub const WorkItem = stub_root.WorkItem;
    pub const parallelExplore = stub_root.parallelExplore;
};
pub const callgraph = struct {
    pub const Function = stub_root.Function;
    pub const CallEdge = stub_root.CallEdge;
    pub const CallGraph = stub_root.CallGraph;
    pub const CallGraphBuilder = stub_root.CallGraphBuilder;
    pub const buildCallGraph = stub_root.buildCallGraph;
};
pub const dependency = struct {
    pub const Module = stub_root.Module;
    pub const DependencyEdge = stub_root.DependencyEdge;
    pub const ImportType = stub_root.ImportType;
    pub const DependencyGraph = stub_root.DependencyGraph;
    pub const ModuleDependency = stub_root.DependencyGraph.ModuleDependency;
    pub const DependencyAnalyzer = stub_root.DependencyAnalyzer;
    pub const buildDependencyGraph = stub_root.buildDependencyGraph;
};

// --- Error & Enums ---

pub const ExploreError = error{FeatureDisabled};
pub const ExploreLevel = enum(u2) { quick = 0, medium = 1, thorough = 2, deep = 3 };
pub const OutputFormat = enum { human, json, compact, yaml };
pub const FileType = enum { source, header, test_file, documentation, config, data, binary, other };

// --- Config Types ---

pub const ExploreConfig = struct {
    level: ExploreLevel = .medium,
    max_files: usize = 10000,
    max_depth: usize = 20,
    timeout_ms: u64 = 60000,
    include_patterns: []const []const u8 = &.{},
    exclude_patterns: []const []const u8 = &.{},
    case_sensitive: bool = false,
    use_regex: bool = false,
    parallel_io: bool = true,
    worker_count: ?usize = null,
    follow_symlinks: bool = false,
    include_hidden: bool = false,
    file_size_limit_bytes: ?u64 = null,
    output_format: OutputFormat = .human,
    pub fn defaultForLevel(level: ExploreLevel) ExploreConfig {
        return .{ .level = level };
    }
};

pub const FileFilter = struct {
    extensions: ?[]const []const u8 = null,
    exclude_extensions: ?[]const []const u8 = null,
    min_size_bytes: ?u64 = null,
    max_size_bytes: ?u64 = null,
    modified_after: ?i128 = null,
    modified_before: ?i128 = null,
    pub fn matches(_: *const FileFilter, _: anytype) bool {
        return false;
    }
};

pub const SearchScope = struct {
    paths: []const []const u8 = &.{"."},
    recursive: bool = true,
    filter: ?FileFilter = null,
};

pub const SearchOptions = struct {
    scope: SearchScope = .{},
    patterns: []const []const u8 = &.{},
    config: ExploreConfig = .{},
};

// --- Result Types ---

pub const Match = struct {
    path: []const u8 = "",
    line: u32 = 0,
    content: []const u8 = "",
    context_before: []const u8 = "",
    context_after: []const u8 = "",
};
pub const MatchType = enum { exact, fuzzy, regex, semantic };
pub const ExplorationStats = struct { files_searched: u32 = 0, matches_found: u32 = 0, search_time_ms: i64 = 0 };

pub const ExploreResult = struct {
    allocator: std.mem.Allocator = undefined,
    matches: []Match = &.{},
    stats: ExplorationStats = .{},
    query: []const u8 = "",
    matches_found: usize = 0,
    files_scanned: usize = 0,
    duration_ms: i64 = 0,
    cancelled: bool = false,
    explore_error: ?ExploreError = null,
    error_message: ?[]const u8 = null,
    pub fn init(allocator: std.mem.Allocator, _: []const u8, _: ExploreLevel) ExploreResult {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *ExploreResult) void {}
    pub fn formatHuman(_: *ExploreResult, _: anytype) !void {}
    pub fn formatJSON(_: *ExploreResult, _: anytype) !void {}
};

// --- Query Types ---

pub const QueryIntent = enum { search, definition, usage, structure };
pub const ParsedQuery = struct { original: []const u8, intent: QueryIntent, terms: []const []const u8 };
pub const QueryUnderstanding = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) QueryUnderstanding {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *QueryUnderstanding) void {}

    pub fn parse(_: *QueryUnderstanding, query_str: []const u8) ParsedQuery {
        return .{ .original = query_str, .intent = .search, .terms = &.{} };
    }

    pub fn freeParsedQuery(_: *QueryUnderstanding, _: *ParsedQuery) void {}
};

// --- Stub Impl Types ---

pub const FileVisitor = struct {
    pub fn init(_: std.mem.Allocator, _: ExploreConfig) FileVisitor {
        return .{};
    }
    pub fn deinit(_: *FileVisitor) void {}
};

// Fix 1: FileStats with all 7 fields matching fs.zig
pub const FileStats = struct {
    path: []const u8 = "",
    size_bytes: u64 = 0,
    mtime: i128 = 0,
    ctime: i128 = 0,
    is_directory: bool = false,
    is_symlink: bool = false,
    mode: u16 = 0,
};

pub const SearchPattern = struct { pattern: []const u8 = "", pattern_type: PatternType = .literal };
pub const PatternType = enum { literal, glob, regex, fuzzy, compound };
pub const PatternCompiler = struct {
    pub fn init(_: std.mem.Allocator) PatternCompiler {
        return .{};
    }
    pub fn deinit(_: *PatternCompiler) void {}
};

pub const ExploreAgent = struct {
    allocator: std.mem.Allocator = undefined,
    config: ExploreConfig = .{},
    pub fn init(_: std.mem.Allocator, _: ExploreConfig) ExploreError!ExploreAgent {
        return ExploreError.FeatureDisabled;
    }
    pub fn deinit(_: *ExploreAgent) void {}
    pub fn explore(_: *ExploreAgent, _: []const u8, _: []const u8) ExploreError!ExploreResult {
        return ExploreError.FeatureDisabled;
    }
    pub fn exploreWithPatterns(_: *ExploreAgent, _: []const u8, _: []const []const u8) ExploreError!ExploreResult {
        return ExploreError.FeatureDisabled;
    }
    pub fn exploreNaturalLanguage(_: *ExploreAgent, _: []const u8, _: []const u8) ExploreError!ExploreResult {
        return ExploreError.FeatureDisabled;
    }
    pub fn cancel(_: *ExploreAgent) void {}
    pub fn isCancelled(_: *ExploreAgent) bool {
        return false;
    }
    pub fn getStats(_: *ExploreAgent) ExplorationStats {
        return .{};
    }
};

pub const AstNode = struct { node_type: AstNodeType = .other, name: []const u8 = "" };

// Fix 2: AstNodeType with all 25 variants matching ast.zig
pub const AstNodeType = enum {
    function,
    struct_type,
    enum_type,
    union_type,
    interface_type,
    class_type,
    const_decl,
    var_decl,
    type_alias,
    import_decl,
    test_decl,
    comment,
    doc_comment,
    error_decl,
    fn_param,
    fn_return_type,
    block,
    if_stmt,
    while_stmt,
    for_stmt,
    switch_stmt,
    field,
    method,
    property,
    other,
};

pub const ParsedFile = struct { path: []const u8 = "", nodes: []AstNode = &.{} };

// Fix 3: AstParser with nodeToMatchType
pub const AstParser = struct {
    pub fn init(_: std.mem.Allocator) AstParser {
        return .{};
    }
    pub fn deinit(_: *AstParser) void {}
    pub fn nodeToMatchType(_: AstNodeType) MatchType {
        return .exact;
    }
};

// Fix 5: ParallelExplorer.init matching real parallel.zig (5 params including io)
pub const ParallelExplorer = struct {
    pub fn init(_: std.mem.Allocator, _: ExploreConfig, _: *ExploreResult, _: []const SearchPattern, _: std.Io) ParallelExplorer {
        return .{};
    }
    pub fn deinit(_: *ParallelExplorer) void {}
};
pub const WorkItem = struct { path: []const u8 = "" };

pub const Function = struct { name: []const u8 = "", file_path: []const u8 = "", line: usize = 0 };
pub const CallEdge = struct { caller: Function = .{}, callee: Function = .{} };

// Fix 6: CallGraph with all methods from callgraph.zig
pub const CallGraph = struct {
    functions: []Function = &.{},
    edges: []CallEdge = &.{},

    pub fn init(_: std.mem.Allocator) CallGraph {
        return .{};
    }

    pub fn deinit(_: *CallGraph) void {}

    pub fn addFunction(_: *CallGraph, _: Function) !void {
        return error.FeatureDisabled;
    }

    pub fn addCall(_: *CallGraph, _: Function, _: Function) !void {
        return error.FeatureDisabled;
    }

    pub fn getCallees(_: *const CallGraph, _: []const u8) ?[]const Function {
        return null;
    }

    pub fn getCallers(_: *const CallGraph, _: []const u8) ?[]const Function {
        return null;
    }

    pub fn hasPathTo(_: *const CallGraph, _: []const u8, _: []const u8) bool {
        return false;
    }

    pub fn toDot(_: *const CallGraph, _: anytype) !void {}
};

pub const CallGraphBuilder = struct {
    pub fn init(_: std.mem.Allocator) CallGraphBuilder {
        return .{};
    }
    pub fn deinit(_: *CallGraphBuilder) void {}
};

pub const Module = struct { name: []const u8 = "", file_path: []const u8 = "", file_type: []const u8 = "" };
pub const DependencyEdge = struct { from: Module = .{}, to: Module = .{}, import_type: ImportType = .local };
pub const ImportType = enum { std, external, local, relative };

// Fix 7: DependencyGraph with all methods from dependency.zig
pub const DependencyGraph = struct {
    modules: []Module = &.{},
    edges: []DependencyEdge = &.{},

    pub const ModuleDependency = struct {
        module: Module,
        import_type: ImportType,
    };

    pub fn init(_: std.mem.Allocator) DependencyGraph {
        return .{};
    }

    pub fn deinit(_: *DependencyGraph) void {}

    pub fn addModule(_: *DependencyGraph, _: Module) !void {
        return error.FeatureDisabled;
    }

    pub fn addDependency(_: *DependencyGraph, _: Module, _: Module, _: ImportType) !void {
        return error.FeatureDisabled;
    }

    pub fn getDependencies(_: *const DependencyGraph, _: []const u8) ?[]const ModuleDependency {
        return null;
    }

    pub fn getDependents(_: *const DependencyGraph, _: []const u8) ?[]const ModuleDependency {
        return null;
    }

    pub fn findCircularDependencies(_: *const DependencyGraph) !std.ArrayListUnmanaged([]const []const u8) {
        return .empty;
    }

    pub fn topologicalSort(_: *const DependencyGraph) !std.ArrayListUnmanaged([]const u8) {
        return .empty;
    }

    pub fn toDot(_: *const DependencyGraph, _: anytype) !void {}
};

pub const DependencyAnalyzer = struct {
    pub fn init(_: std.mem.Allocator) DependencyAnalyzer {
        return .{};
    }
    pub fn deinit(_: *DependencyAnalyzer) void {}
};

// --- Free Functions ---

// Fix 8: createDefault/Quick/ThoroughAgent return ExploreError!ExploreAgent
pub fn createDefaultAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}
pub fn createQuickAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}
pub fn createThoroughAgent(_: std.mem.Allocator) ExploreError!ExploreAgent {
    return error.FeatureDisabled;
}

pub fn parallelExplore(_: std.mem.Allocator, _: []const u8, _: ExploreConfig, _: []const u8) ExploreError!ExploreResult {
    return ExploreError.FeatureDisabled;
}
pub fn buildCallGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!CallGraph {
    return ExploreError.FeatureDisabled;
}
pub fn buildDependencyGraph(_: std.mem.Allocator, _: []const []const u8) ExploreError!DependencyGraph {
    return ExploreError.FeatureDisabled;
}

// --- Discovery types (merged from stubs/discovery.zig) ---

pub const DiscoveryConfig = struct {
    custom_paths: []const []const u8 = &.{},
    recursive: bool = true,
    max_depth: u32 = 5,
    extensions: []const []const u8 = &.{ ".gguf", ".mlx", ".bin", ".safetensors" },
    validate_files: bool = true,
    validation_timeout_ms: u32 = 5000,
};

pub const ModelFormat = enum {
    gguf,
    mlx,
    safetensors,
    pytorch_bin,
    onnx,
    unknown,

    pub fn fromExtension(_: []const u8) ModelFormat {
        return .unknown;
    }
};

pub const QuantizationType = enum {
    f32,
    f16,
    q8_0,
    q8_1,
    q5_0,
    q5_1,
    q4_0,
    q4_1,
    q4_k_m,
    q4_k_s,
    q5_k_m,
    q5_k_s,
    q6_k,
    q2_k,
    q3_k_m,
    q3_k_s,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    unknown,

    pub fn bitsPerWeight(self: QuantizationType) f32 {
        return switch (self) {
            .f32 => 32.0,
            .f16 => 16.0,
            .q8_0, .q8_1 => 8.0,
            .q6_k => 6.0,
            .q5_0, .q5_1, .q5_k_m, .q5_k_s => 5.0,
            .q4_0, .q4_1, .q4_k_m, .q4_k_s => 4.0,
            .q3_k_m, .q3_k_s => 3.0,
            .q2_k => 2.0,
            .iq2_xxs, .iq2_xs => 2.5,
            .iq3_xxs => 3.0,
            .unknown => 4.0,
        };
    }
};

pub const DiscoveredModel = struct {
    path: []const u8 = "",
    name: []const u8 = "",
    size_bytes: u64 = 0,
    format: ModelFormat = .unknown,
    estimated_params: ?u64 = null,
    quantization: ?QuantizationType = null,
    validated: bool = false,
    modified_time: i128 = 0,

    pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
};

pub const SystemCapabilities = struct {
    cpu_cores: u32 = 1,
    total_ram_bytes: u64 = 0,
    available_ram_bytes: u64 = 0,
    gpu_available: bool = false,
    gpu_memory_bytes: u64 = 0,
    gpu_compute_capability: ?f32 = null,
    avx2_available: bool = false,
    avx512_available: bool = false,
    neon_available: bool = false,
    os: std.Target.Os.Tag = .linux,
    arch: std.Target.Cpu.Arch = .x86_64,

    pub fn maxModelSize(self: @This()) u64 {
        return self.available_ram_bytes * 80 / 100;
    }

    pub fn recommendedThreads(self: @This()) u32 {
        if (self.cpu_cores > 2) return self.cpu_cores - 1;
        return self.cpu_cores;
    }

    pub fn recommendedBatchSize(_: @This(), _: u64) u32 {
        return 1;
    }
};

pub const AdaptiveConfig = struct {
    num_threads: u32 = 4,
    batch_size: u32 = 1,
    context_length: u32 = 2048,
    use_gpu: bool = true,
    use_mmap: bool = true,
    mlock: bool = false,
    kv_cache_type: KvCacheType = .standard,
    flash_attention: bool = false,
    tensor_parallel: u32 = 1,
    prefill_chunk_size: u32 = 512,

    pub const KvCacheType = enum {
        standard,
        sliding_window,
        paged,
    };
};

pub const ModelRequirements = struct {
    min_ram_bytes: u64 = 0,
    min_gpu_memory_bytes: u64 = 0,
    min_compute_capability: f32 = 0,
    requires_avx2: bool = false,
    requires_avx512: bool = false,
    recommended_context: u32 = 2048,
};

pub const WarmupResult = struct {
    load_time_ms: u64 = 0,
    first_inference_ms: u64 = 0,
    tokens_per_second: f32 = 0,
    memory_usage_bytes: u64 = 0,
    success: bool = false,
    error_message: ?[]const u8 = null,
    recommended_config: ?AdaptiveConfig = null,
};

pub const ModelDiscovery = struct {
    allocator: std.mem.Allocator,
    config: DiscoveryConfig,
    discovered_models: std.ArrayListUnmanaged(DiscoveredModel) = .empty,
    capabilities: SystemCapabilities = .{},

    pub fn init(allocator: std.mem.Allocator, disc_config: DiscoveryConfig) @This() {
        return .{
            .allocator = allocator,
            .config = disc_config,
            .discovered_models = .empty,
            .capabilities = detectCapabilities(),
        };
    }

    pub fn deinit(self: *@This()) void {
        for (self.discovered_models.items) |*model| {
            model.deinit(self.allocator);
        }
        self.discovered_models.deinit(self.allocator);
    }

    pub fn scanAll(_: *@This()) !void {
        return error.FeatureDisabled;
    }

    pub fn scanPath(_: *@This(), _: []const u8) !void {
        return error.FeatureDisabled;
    }

    pub fn addModelPath(_: *@This(), _: []const u8) !void {
        return error.FeatureDisabled;
    }

    pub fn addModelWithSize(_: *@This(), _: []const u8, _: u64) !void {
        return error.FeatureDisabled;
    }

    pub fn getModels(self: *@This()) []DiscoveredModel {
        return self.discovered_models.items;
    }

    pub fn modelCount(self: *@This()) usize {
        return self.discovered_models.items.len;
    }

    pub fn findBestModel(_: *@This(), _: ModelRequirements) ?*DiscoveredModel {
        return null;
    }

    pub fn generateConfig(_: *@This(), _: *const DiscoveredModel) AdaptiveConfig {
        return .{};
    }
};

pub fn detectCapabilities() SystemCapabilities {
    return .{};
}

pub fn runWarmup(_: []const u8, _: AdaptiveConfig) WarmupResult {
    return .{};
}

// --- Codebase Index types (merged from stubs/codebase_index.zig) ---

pub const FileMetadata = struct {
    path: []const u8 = "",
    size_bytes: usize = 0,
    line_count: usize = 0,
    chunk_count: usize = 0,
    last_indexed: i64 = 0,
};

pub const CodeSnippet = struct {
    file_path: []const u8 = "",
    start_line: usize = 0,
    end_line: usize = 0,
    content: []const u8 = "",
    relevance_score: f32 = 0,
};

// Fix 9: CodebaseAnswer.snippets uses .empty (Zig 0.16 pattern)
pub const CodebaseAnswer = struct {
    allocator: std.mem.Allocator,
    snippets: std.ArrayListUnmanaged(CodeSnippet) = .empty,
    summary: []const u8 = "",

    pub fn deinit(_: *CodebaseAnswer) void {}
};

pub const IndexResult = struct {
    files_indexed: usize = 0,
    total_chunks: usize = 0,
    total_lines: usize = 0,
    duration_ms: u64 = 0,
};

pub const IndexStats = struct {
    file_count: usize = 0,
    total_chunks: usize = 0,
    total_size_bytes: usize = 0,
    index_path: []const u8 = "",
};

pub const CodebaseIndex = struct {
    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: []const u8) error{FeatureDisabled}!Self {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn indexFile(_: *Self, _: []const u8, _: []const u8) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }

    pub fn query(_: *Self, _: []const u8) error{FeatureDisabled}!CodebaseAnswer {
        return error.FeatureDisabled;
    }

    pub fn getStats(_: *const Self) IndexStats {
        return .{};
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
