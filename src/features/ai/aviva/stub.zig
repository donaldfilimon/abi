//! Aviva Profile stub — active when AI sub-feature is disabled.

const std = @import("std");

// ── Sub-module stubs ───────────────────────────────────────────────────────

pub const classifier_mod = struct {
    pub const QueryClassifier = @This().QueryClassifier__;
    pub const QueryType = @This().QueryType__;
    pub const ClassificationResult = @This().ClassificationResult__;
    pub const Language = @This().Language__;
    pub const Domain = @This().Domain__;
    pub const ClassifierConfig = @This().ClassifierConfig__;

    const QueryType__ = enum {
        code_request,
        debugging,
        factual_query,
        explanation,
        documentation,
        comparison,
        optimization,
        general,

        pub fn getDescription(self: @This()) []const u8 {
            _ = self;
            return "";
        }

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

    const Language__ = enum {
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

        pub fn getFileExtension(self: @This()) []const u8 {
            _ = self;
            return "";
        }
    };

    const Domain__ = enum {
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

        pub fn getDescription(self: @This()) []const u8 {
            _ = self;
            return "";
        }
    };

    const Complexity = enum {
        trivial,
        simple,
        moderate,
        complex,
        expert,

        pub fn getEstimatedTokens(self: @This()) usize {
            _ = self;
            return 0;
        }
    };

    const ClassificationResult__ = struct {
        query_type: QueryType__ = .general,
        confidence: f32 = 0,
        language: Language__ = .unknown,
        domain: Domain__ = .general,
        complexity: Complexity = .moderate,
        secondary_types: [2]?QueryType__ = [_]?QueryType__{null} ** 2,
        key_indicators: [5]?[]const u8 = [_]?[]const u8{null} ** 5,
    };

    const ClassifierConfig__ = struct {
        min_confidence: f32 = 0.5,
        detect_language: bool = true,
        detect_domain: bool = true,
        estimate_complexity: bool = true,
    };

    const QueryClassifier__ = struct {
        config: ClassifierConfig__,

        pub fn init() @This() {
            return initWithConfig(.{});
        }

        pub fn initWithConfig(config: ClassifierConfig__) @This() {
            return .{ .config = config };
        }

        pub fn classify(_: *const @This(), _: []const u8) ClassificationResult__ {
            return .{};
        }
    };
};

pub const knowledge_mod = struct {
    pub const KnowledgeRetriever = @This().KnowledgeRetriever__;
    pub const KnowledgeFragment = @This().KnowledgeFragment__;
    pub const Source = @This().Source__;
    pub const SourceType = @This().SourceType__;
    pub const RetrievalResult = @This().RetrievalResult__;
    pub const RetrieverConfig = @This().RetrieverConfig__;
    pub const KnowledgeBase = @This().KnowledgeBase__;

    pub fn formatKnowledgeForResponse(_: std.mem.Allocator, _: []const KnowledgeFragment__, _: bool) ![]const u8 {
        return error.FeatureDisabled;
    }

    const SourceType__ = enum {
        documentation,
        research,
        code_example,
        community,
        internal,
        training_data,
        web_search,
        user_provided,

        pub fn getReliabilityDefault(self: @This()) f32 {
            _ = self;
            return 0;
        }
    };

    const Source__ = struct {
        name: []const u8 = "",
        source_type: SourceType__ = .documentation,
        url: ?[]const u8 = null,
        reference: ?[]const u8 = null,
        reliability: f32 = 0.8,
    };

    const KnowledgeFragment__ = struct {
        content: []const u8 = "",
        source: Source__ = .{},
        relevance: f32 = 0,
        confidence: f32 = 0,
        domain: classifier_mod.Domain = .general,
        last_verified: i64 = 0,
    };

    const RetrieverConfig__ = struct {
        max_fragments: usize = 5,
        min_relevance: f32 = 0.5,
        min_confidence: f32 = 0.6,
        enable_cache: bool = true,
        cache_ttl_seconds: u64 = 3600,
        include_sources: bool = true,
    };

    const RetrievalResult__ = struct {
        allocator: std.mem.Allocator,
        fragments: std.ArrayListUnmanaged(KnowledgeFragment__) = .empty,
        query: []const u8 = "",
        total_found: usize = 0,
        retrieval_time_ms: u64 = 0,
        from_cache: bool = false,

        pub fn init(allocator: std.mem.Allocator, query: []const u8) @This() {
            return .{ .allocator = allocator, .query = query };
        }

        pub fn deinit(self: *@This()) void {
            self.fragments.deinit(self.allocator);
        }

        pub fn getBest(_: *const @This()) ?KnowledgeFragment__ {
            return null;
        }

        pub fn getAboveThreshold(self: *const @This(), _: f32) []const KnowledgeFragment__ {
            return self.fragments.items[0..0];
        }
    };

    const KnowledgeBase__ = struct {
        allocator: std.mem.Allocator,
        fragments: std.ArrayListUnmanaged(KnowledgeFragment__) = .empty,
        db_context: ?*anyopaque = null,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{ .allocator = allocator };
        }

        pub fn initWithWdbx(allocator: std.mem.Allocator, _: *anyopaque) @This() {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *@This()) void {
            self.fragments.deinit(self.allocator);
        }

        pub fn search(self: *@This(), allocator: std.mem.Allocator, _: []const u8, _: ?classifier_mod.Domain, _: usize) ![]KnowledgeFragment__ {
            _ = self;
            return allocator.alloc(KnowledgeFragment__, 0);
        }

        pub fn getByDomain(self: *@This(), allocator: std.mem.Allocator, _: classifier_mod.Domain, _: usize) ![]KnowledgeFragment__ {
            _ = self;
            return allocator.alloc(KnowledgeFragment__, 0);
        }

        pub fn add(self: *@This(), _: KnowledgeFragment__) !void {
            _ = self;
            return error.FeatureDisabled;
        }
    };

    const KnowledgeRetriever__ = struct {
        allocator: std.mem.Allocator,
        config: RetrieverConfig__,
        cache: std.StringHashMapUnmanaged(void) = .empty,
        knowledge_base: KnowledgeBase__,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return initWithConfig(allocator, .{});
        }

        pub fn initWithConfig(allocator: std.mem.Allocator, config: RetrieverConfig__) @This() {
            return .{
                .allocator = allocator,
                .config = config,
                .knowledge_base = KnowledgeBase__.init(allocator),
            };
        }

        pub fn deinit(self: *@This()) void {
            self.cache.deinit(self.allocator);
            self.knowledge_base.deinit();
        }

        pub fn retrieve(self: *@This(), _: []const u8, _: ?classifier_mod.Domain) !RetrievalResult__ {
            return RetrievalResult__.init(self.allocator, "");
        }

        pub fn retrieveForDomain(_: *@This(), _: classifier_mod.Domain, _: usize) ![]KnowledgeFragment__ {
            return error.FeatureDisabled;
        }

        pub fn addKnowledge(_: *@This(), _: KnowledgeFragment__) !void {
            return error.FeatureDisabled;
        }

        pub fn clearCache(self: *@This()) void {
            self.cache.clearRetainingCapacity();
        }
    };
};

pub const code_mod = struct {
    pub const CodeGenerator = @This().CodeGenerator__;
    pub const CodeBlock = @This().CodeBlock__;
    pub const GenerationOptions = @This().GenerationOptions__;
    pub const IndentStyle = @This().IndentStyle__;
    pub const GeneratorConfig = @This().GeneratorConfig__;
    pub const ValidationResult = @This().ValidationResult__;

    pub fn extractCodeBlocks(allocator: std.mem.Allocator, _: []const u8) ![]CodeBlock__ {
        return allocator.alloc(CodeBlock__, 0);
    }

    const IndentStyle__ = enum {
        tabs,
        spaces_2,
        spaces_4,

        pub fn getString(self: @This()) []const u8 {
            _ = self;
            return "    ";
        }
    };

    const GenerationOptions__ = struct {
        include_comments: bool = true,
        include_docs: bool = true,
        include_types: bool = true,
        include_error_handling: bool = true,
        verbose_names: bool = false,
        max_line_length: usize = 100,
        indent_style: IndentStyle__ = .spaces_4,
    };

    const GeneratorConfig__ = struct {
        default_options: GenerationOptions__ = .{},
        validate_syntax: bool = true,
        add_language_markers: bool = true,
    };

    const ValidationResult__ = struct {
        is_valid: bool = true,
        error_message: ?[]const u8 = null,
        line_number: ?usize = null,
    };

    const CodeBlock__ = struct {
        language: classifier_mod.Language = .unknown,
        code: []const u8 = "",
        explanation: ?[]const u8 = null,
        filename: ?[]const u8 = null,
        is_complete_file: bool = false,
        loc: usize = 0,
    };

    const CodeGenerator__ = struct {
        allocator: std.mem.Allocator,
        config: GeneratorConfig__,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return initWithConfig(allocator, .{});
        }

        pub fn initWithConfig(allocator: std.mem.Allocator, config: GeneratorConfig__) @This() {
            return .{ .allocator = allocator, .config = config };
        }

        pub fn formatCodeBlock(_: *@This(), _: []const u8, _: classifier_mod.Language, _: ?GenerationOptions__) !CodeBlock__ {
            return error.FeatureDisabled;
        }

        pub fn generateFunctionTemplate(_: *@This(), _: []const u8, _: []const []const u8, _: ?[]const u8, _: classifier_mod.Language, _: ?GenerationOptions__) !CodeBlock__ {
            return error.FeatureDisabled;
        }

        pub fn wrapInMarkdown(_: *@This(), _: []const u8, _: classifier_mod.Language) ![]const u8 {
            return error.FeatureDisabled;
        }

        pub fn validateStructure(_: *const @This(), _: []const u8, _: classifier_mod.Language) ValidationResult__ {
            return .{};
        }

        pub fn getLanguageName(_: *const @This(), _: classifier_mod.Language) []const u8 {
            return "";
        }
    };
};

pub const facts_mod = struct {
    pub const FactChecker = @This().FactChecker__;
    pub const Claim = @This().Claim__;
    pub const ClaimType = @This().ClaimType__;
    pub const FactCheckResult = @This().FactCheckResult__;
    pub const FactCheckerConfig = @This().FactCheckerConfig__;

    pub fn applyQualifications(allocator: std.mem.Allocator, content: []const u8, _: []const []const u8) ![]const u8 {
        return allocator.dupe(u8, content);
    }

    const ClaimType__ = enum {
        definition,
        numerical,
        temporal,
        comparison,
        causal,
        recommendation,
        procedural,
        attribution,
        general,

        pub fn getDefaultConfidence(self: @This()) f32 {
            _ = self;
            return 0;
        }
    };

    const Claim__ = struct {
        text: []const u8 = "",
        claim_type: ClaimType__ = .general,
        confidence: f32 = 0,
        needs_verification: bool = false,
        evidence: ?[]const u8 = null,
        qualification: ?[]const u8 = null,
    };

    const FactCheckerConfig__ = struct {
        min_unqualified_confidence: f32 = 0.8,
        auto_qualify: bool = true,
        extract_claims: bool = true,
        max_claims: usize = 10,
    };

    const FactCheckResult__ = struct {
        allocator: std.mem.Allocator,
        claims: std.ArrayListUnmanaged(Claim__) = .empty,
        overall_confidence: f32 = 1.0,
        verification_needed_count: usize = 0,
        qualifications: std.ArrayListUnmanaged([]const u8) = .empty,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *@This()) void {
            self.claims.deinit(self.allocator);
            self.qualifications.deinit(self.allocator);
        }

        pub fn getHighConfidence(self: *const @This(), _: f32) []const Claim__ {
            return self.claims.items[0..0];
        }
    };

    const FactChecker__ = struct {
        allocator: std.mem.Allocator,
        config: FactCheckerConfig__,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return initWithConfig(allocator, .{});
        }

        pub fn initWithConfig(allocator: std.mem.Allocator, config: FactCheckerConfig__) @This() {
            return .{ .allocator = allocator, .config = config };
        }

        pub fn check(self: *@This(), _: []const u8) !FactCheckResult__ {
            return FactCheckResult__.init(self.allocator);
        }

        pub fn scoreStatement(_: *const @This(), _: []const u8) f32 {
            return 0;
        }
    };
};

// ── Top-level type re-exports ──────────────────────────────────────────────

pub const QueryClassifier = classifier_mod.QueryClassifier;
pub const QueryType = classifier_mod.QueryType;
pub const ClassificationResult = classifier_mod.ClassificationResult;
pub const Language = classifier_mod.Language;
pub const Domain = classifier_mod.Domain;
pub const KnowledgeRetriever = knowledge_mod.KnowledgeRetriever;
pub const KnowledgeFragment = knowledge_mod.KnowledgeFragment;
pub const CodeGenerator = code_mod.CodeGenerator;
pub const CodeBlock = code_mod.CodeBlock;
pub const FactChecker = facts_mod.FactChecker;
pub const Claim = facts_mod.Claim;

// ── AvivaProfile stub ──────────────────────────────────────────────────────

/// Aviva profile stub — returns FeatureDisabled for all operations.
pub const AvivaProfile = struct {
    allocator: std.mem.Allocator,
    config: AvivaConfig,
    agent: *anyopaque,
    classifier: QueryClassifier,
    knowledge_retriever: KnowledgeRetriever,
    code_generator: CodeGenerator,
    fact_checker: FactChecker,

    const Self = @This();

    pub fn init(_: std.mem.Allocator, _: AvivaConfig) !*Self {
        return error.FeatureDisabled;
    }

    pub fn deinit(_: *Self) void {}

    pub fn getName(_: *const Self) []const u8 {
        return "Aviva";
    }

    pub fn getType(_: *const Self) ProfileType {
        return .aviva;
    }

    pub fn process(_: *Self, _: ProfileRequest) anyerror!ProfileResponse {
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

    pub fn interface(_: *Self) ProfileInterface {
        return .{
            .ptr = undefined,
            .vtable = &.{
                .process = &stubProcess,
                .getName = &stubGetName,
                .getType = &stubGetType,
            },
        };
    }

    fn stubProcess(_: *anyopaque, _: ProfileRequest) anyerror!ProfileResponse {
        return error.FeatureDisabled;
    }

    fn stubGetName(_: *anyopaque) []const u8 {
        return "Aviva";
    }

    fn stubGetType(_: *anyopaque) ProfileType {
        return .aviva;
    }
};

// ── Agent override helpers ─────────────────────────────────────────────────

pub const Overrides = struct {
    prev_temperature: f32 = 0,
    prev_max_tokens: u32 = 0,
    system_index: ?usize = null,
};

pub fn apply(_: std.mem.Allocator, _: *anyopaque, _: ProfileRequest) !Overrides {
    return error.FeatureDisabled;
}

pub fn restore(_: std.mem.Allocator, _: *anyopaque, _: Overrides) void {}

// ── Stub types from parent AI module ───────────────────────────────────────
// These mirror the types used by mod.zig from the parent ai types/config.

const ProfileType = enum {
    assistant,
    coder,
    writer,
    analyst,
    companion,
    docs,
    reviewer,
    minimal,
    abbey,
    aviva,
    abi,
    ralph,
    ava,
};

const ProfileRequest = struct {
    content: []const u8 = "",
    session_id: ?[]const u8 = null,
    user_id: ?[]const u8 = null,
    preferred_profile: ?ProfileType = null,
    system_instruction: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
};

const ProfileResponse = struct {
    content: []const u8 = "",
    profile: ProfileType = .aviva,
    confidence: f32 = 0,
    emotional_tone: ?u8 = null,
    reasoning_chain: ?[]const void = null,
    code_blocks: ?[]const void = null,
    references: ?[]const void = null,
    generation_time_ms: u64 = 0,
};

const ProfileInterface = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        process: *const fn (ctx: *anyopaque, request: ProfileRequest) anyerror!ProfileResponse,
        getName: *const fn (ctx: *anyopaque) []const u8,
        getType: *const fn (ctx: *anyopaque) ProfileType,
    };
};

const AvivaConfig = struct {
    directness_level: f32 = 0.9,
    include_disclaimers: bool = false,
    include_code_comments: bool = true,
    verify_facts: bool = true,
    max_response_length: u32 = 4096,
    cite_sources: bool = false,
    skip_preamble: bool = false,
};

test {
    std.testing.refAllDecls(@This());
}
