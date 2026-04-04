//! Aviva Profile stub — active when AI sub-feature is disabled.

const std = @import("std");

// ── Base types (defined once, shared by sub-modules and top-level) ─────────

const _QueryType = enum {
    code_request,
    debugging,
    factual_query,
    explanation,
    documentation,
    comparison,
    optimization,
    general,
    pub fn recommendsCodeBlock(self: @This()) bool {
        return switch (self) {
            .code_request, .debugging, .optimization => true,
            else => false,
        };
    }
    pub fn recommendsBrevity(self: @This()) bool {
        return switch (self) {
            .factual_query, .documentation => true,
            else => false,
        };
    }
};
const _Language = enum {
    unknown,
    zig,
    rust,
    python,
    javascript,
    typescript,
    go,
    c,
    cpp,
    java,
    csharp,
    ruby,
    sql,
    bash,
    html,
    css,
};
const _Domain = enum {
    general,
    web_development,
    systems_programming,
    data_science,
    devops,
    databases,
    networking,
    security,
    mobile,
    ai_ml,
    algorithms,
    testing,
};
const _ClassifierConfig = struct {
    min_confidence: f32 = 0.5,
    detect_language: bool = true,
    detect_domain: bool = true,
    estimate_complexity: bool = true,
};
const _ClassificationResult = struct {
    query_type: _QueryType = .general,
    confidence: f32 = 0,
    language: _Language = .unknown,
    domain: _Domain = .general,
};
const _QueryClassifier = struct {
    pub fn init() @This() {
        return .{};
    }
    pub fn initWithConfig(_: _ClassifierConfig) @This() {
        return .{};
    }
    pub fn classify(_: *const @This(), _: []const u8) _ClassificationResult {
        return .{};
    }
};

const _Source = struct { name: []const u8 = "", url: ?[]const u8 = null };
const _KnowledgeFragment = struct {
    content: []const u8 = "",
    source: _Source = .{},
    relevance: f32 = 0,
    confidence: f32 = 0,
};
const _RetrieverConfig = struct {
    max_fragments: usize = 5,
    min_relevance: f32 = 0.5,
    min_confidence: f32 = 0.6,
    enable_cache: bool = true,
    include_sources: bool = true,
};
const _RetrievalResult = struct {
    allocator: std.mem.Allocator,
    fragments: std.ArrayListUnmanaged(_KnowledgeFragment) = .empty,
    pub fn init(a: std.mem.Allocator, _: []const u8) _RetrievalResult {
        return .{ .allocator = a };
    }
    pub fn deinit(self: *_RetrievalResult) void {
        self.fragments.deinit(self.allocator);
    }
};
const _KnowledgeRetriever = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    pub fn init(allocator: std.mem.Allocator) @This() {
        return initWithConfig(allocator, .{});
    }
    pub fn initWithConfig(allocator: std.mem.Allocator, _: _RetrieverConfig) @This() {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *@This()) void {}
    pub fn retrieve(self: *@This(), _: []const u8, _: ?_Domain) !_RetrievalResult {
        return _RetrievalResult.init(self.allocator, "");
    }
};

const _CodeBlock = struct {
    language: _Language = .unknown,
    code: []const u8 = "",
};
const _GenerationOptions = struct {
    include_comments: bool = true,
    include_docs: bool = true,
};
const _GeneratorConfig = struct {
    validate_syntax: bool = true,
    add_language_markers: bool = true,
    default_options: _GenerationOptions = .{},
};
const _CodeGenerator = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    pub fn init(allocator: std.mem.Allocator) @This() {
        return initWithConfig(allocator, .{});
    }
    pub fn initWithConfig(allocator: std.mem.Allocator, _: _GeneratorConfig) @This() {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *@This()) void {}
    pub fn getLanguageName(_: *const @This(), _: _Language) []const u8 {
        return "";
    }
    pub fn generateFunctionTemplate(_: *@This(), _: []const u8, _: []const []const u8, _: ?[]const u8, _: _Language, _: ?_GenerationOptions) !_CodeBlock {
        return error.FeatureDisabled;
    }
};

const _Claim = struct { text: []const u8 = "", confidence: f32 = 0 };
const _FactCheckerConfig = struct {
    min_unqualified_confidence: f32 = 0.8,
    auto_qualify: bool = true,
    extract_claims: bool = true,
};
const _FactCheckResult = struct {
    allocator: std.mem.Allocator,
    qualifications: std.ArrayListUnmanaged([]const u8) = .empty,
    overall_confidence: f32 = 1.0,
    pub fn init(a: std.mem.Allocator) _FactCheckResult {
        return .{ .allocator = a };
    }
    pub fn deinit(self: *_FactCheckResult) void {
        self.qualifications.deinit(self.allocator);
    }
};
const _FactChecker = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    pub fn init(allocator: std.mem.Allocator) @This() {
        return initWithConfig(allocator, .{});
    }
    pub fn initWithConfig(allocator: std.mem.Allocator, _: _FactCheckerConfig) @This() {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *@This()) void {}
    pub fn check(self: *@This(), _: []const u8) !_FactCheckResult {
        return _FactCheckResult.init(self.allocator);
    }
    pub fn scoreStatement(_: *const @This(), _: []const u8) f32 {
        return 0;
    }
};

// ── Sub-module stubs (re-export base types) ────────────────────────────────

pub const classifier_mod = struct {
    pub const QueryType = _QueryType;
    pub const Language = _Language;
    pub const Domain = _Domain;
    pub const ClassifierConfig = _ClassifierConfig;
    pub const ClassificationResult = _ClassificationResult;
    pub const QueryClassifier = _QueryClassifier;
};

pub const knowledge_mod = struct {
    pub const KnowledgeFragment = _KnowledgeFragment;
    pub const RetrieverConfig = _RetrieverConfig;
    pub const KnowledgeRetriever = _KnowledgeRetriever;
};

pub const code_mod = struct {
    pub const CodeBlock = _CodeBlock;
    pub const GenerationOptions = _GenerationOptions;
    pub const GeneratorConfig = _GeneratorConfig;
    pub const CodeGenerator = _CodeGenerator;
    pub fn extractCodeBlocks(a: std.mem.Allocator, _: []const u8) ![]_CodeBlock {
        return a.alloc(_CodeBlock, 0);
    }
};

pub const facts_mod = struct {
    pub const Claim = _Claim;
    pub const FactCheckerConfig = _FactCheckerConfig;
    pub const FactChecker = _FactChecker;
    pub fn applyQualifications(a: std.mem.Allocator, content: []const u8, _: []const []const u8) ![]const u8 {
        return a.dupe(u8, content);
    }
};

// ── Top-level type re-exports (match mod.zig) ──────────────────────────────

pub const QueryClassifier = _QueryClassifier;
pub const QueryType = _QueryType;
pub const ClassificationResult = _ClassificationResult;
pub const Language = _Language;
pub const Domain = _Domain;
pub const KnowledgeRetriever = _KnowledgeRetriever;
pub const KnowledgeFragment = _KnowledgeFragment;
pub const CodeGenerator = _CodeGenerator;
pub const CodeBlock = _CodeBlock;
pub const FactChecker = _FactChecker;
pub const Claim = _Claim;

// ── AvivaProfile stub ──────────────────────────────────────────────────────

const _ProfileType = enum { assistant, coder, writer, analyst, companion, docs, reviewer, minimal, abbey, aviva, abi, ralph, ava };
const _ProfileRequest = struct { content: []const u8 = "", temperature: ?f32 = null, max_tokens: ?u32 = null, system_instruction: ?[]const u8 = null };
const _ProfileResponse = struct { content: []const u8 = "", profile: _ProfileType = .aviva, confidence: f32 = 0, generation_time_ms: u64 = 0 };
const _ProfileInterface = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    const VTable = struct {
        process: *const fn (ctx: *anyopaque, request: _ProfileRequest) anyerror!_ProfileResponse,
        getName: *const fn (ctx: *anyopaque) []const u8,
        getType: *const fn (ctx: *anyopaque) _ProfileType,
    };
};
const _AvivaConfig = struct {
    directness_level: f32 = 0.9,
    include_disclaimers: bool = false,
    include_code_comments: bool = true,
    verify_facts: bool = true,
    max_response_length: u32 = 4096,
    cite_sources: bool = false,
    skip_preamble: bool = false,
};

pub const AvivaProfile = struct {
    allocator: std.mem.Allocator,
    config: _AvivaConfig,
    agent: *anyopaque,
    classifier: QueryClassifier,
    knowledge_retriever: KnowledgeRetriever,
    code_generator: CodeGenerator,
    fact_checker: FactChecker,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: _AvivaConfig) !*Self {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Self) void {}
    pub fn getName(_: *const Self) []const u8 {
        return "Aviva";
    }
    pub fn getType(_: *const Self) _ProfileType {
        return .aviva;
    }
    pub fn process(_: *Self, _: _ProfileRequest) anyerror!_ProfileResponse {
        return error.FeatureDisabled;
    }
    pub fn classifyQuery(_: *const Self, _: []const u8) ClassificationResult {
        return .{};
    }
    pub fn generateFunctionTemplate(_: *Self, _: []const u8, _: []const []const u8, _: ?[]const u8, _: Language) !CodeBlock {
        return error.FeatureDisabled;
    }
    pub fn scoreFactualConfidence(_: *const Self, _: []const u8) f32 {
        return 0;
    }
    pub fn interface(_: *Self) _ProfileInterface {
        return .{ .ptr = undefined, .vtable = &.{ .process = &stubProcess, .getName = &stubGetName, .getType = &stubGetType } };
    }
    fn stubProcess(_: *anyopaque, _: _ProfileRequest) anyerror!_ProfileResponse {
        return error.FeatureDisabled;
    }
    fn stubGetName(_: *anyopaque) []const u8 {
        return "Aviva";
    }
    fn stubGetType(_: *anyopaque) _ProfileType {
        return .aviva;
    }
};

// ── Agent override helpers ─────────────────────────────────────────────────

pub const Overrides = struct { prev_temperature: f32 = 0, prev_max_tokens: u32 = 0, system_index: ?usize = null };
pub fn apply(_: std.mem.Allocator, _: *anyopaque, _: _ProfileRequest) !Overrides {
    return error.FeatureDisabled;
}
pub fn restore(_: std.mem.Allocator, _: *anyopaque, _: Overrides) void {}

test {
    std.testing.refAllDecls(@This());
}
