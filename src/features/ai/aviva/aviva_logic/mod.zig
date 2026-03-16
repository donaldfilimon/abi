//! Aviva Persona - Direct Expert
//!
//! Aviva provides concise, factual, and technically rigorous responses.
//! This persona minimizes hedging and emotional overhead in favor of
//! density and accuracy.
//!
//! Enhanced Features:
//! - Query classification for optimized response strategy
//! - Knowledge retrieval for factual grounding
//! - Code generation with proper formatting
//! - Fact checking with confidence scoring

const std = @import("std");
const time = @import("../../../services/shared/mod.zig").time;
const types = @import("types");
const config = @import("../config.zig");
const agent_mod = @import("agents");

pub const classifier_mod = @import("classifier.zig");
pub const knowledge_mod = @import("knowledge.zig");
pub const code_mod = @import("code.zig");
pub const facts_mod = @import("facts.zig");

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

/// Aviva persona implementation with enhanced capabilities.
pub const AvivaPersona = struct {
    allocator: std.mem.Allocator,
    config: config.AvivaConfig,
    /// The underlying agent instance used for generation.
    agent: *agent_mod.Agent,
    /// Query classifier for response strategy.
    classifier: QueryClassifier,
    /// Knowledge retriever for factual grounding.
    knowledge_retriever: KnowledgeRetriever,
    /// Code generator for code responses.
    code_generator: CodeGenerator,
    /// Fact checker for accuracy.
    fact_checker: FactChecker,

    const Self = @This();

    /// Initialize the Aviva persona with configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: config.AvivaConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const agent_cfg = agent_mod.AgentConfig{
            .name = "Aviva",
            .temperature = 0.2, // Low temperature for factual consistency
            .top_p = 0.95,
            .max_tokens = cfg.max_response_length,
            .system_prompt = "You are Aviva, a direct expert. Your responses are concise, technically dense, and strictly factual. Avoid conversational fillers, excessive apologies, or emotional hedging. Prioritize accuracy and efficiency. If code is requested, provide it cleanly without unnecessary explanation unless the explanation is vital for technical correctness.",
        };

        const agent_ptr = try allocator.create(agent_mod.Agent);
        errdefer allocator.destroy(agent_ptr);
        agent_ptr.* = try agent_mod.Agent.init(allocator, agent_cfg);

        // Initialize classifier with configured options
        const classifier_config = classifier_mod.ClassifierConfig{
            .min_confidence = 0.5,
            .detect_language = true,
            .detect_domain = true,
            .estimate_complexity = true,
        };

        // Initialize knowledge retriever
        const retriever_config = knowledge_mod.RetrieverConfig{
            .max_fragments = 5,
            .min_relevance = 0.5,
            .min_confidence = 0.6,
            .enable_cache = true,
            .include_sources = cfg.cite_sources,
        };

        // Initialize code generator
        const code_config = code_mod.GeneratorConfig{
            .validate_syntax = true,
            .add_language_markers = true,
            .default_options = .{
                .include_comments = !cfg.skip_preamble,
                .include_docs = !cfg.skip_preamble,
            },
        };

        // Initialize fact checker
        const fact_config = facts_mod.FactCheckerConfig{
            .min_unqualified_confidence = if (cfg.cite_sources) 0.9 else 0.8,
            .auto_qualify = true,
            .extract_claims = true,
        };

        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .agent = agent_ptr,
            .classifier = QueryClassifier.initWithConfig(classifier_config),
            .knowledge_retriever = KnowledgeRetriever.initWithConfig(allocator, retriever_config),
            .code_generator = CodeGenerator.initWithConfig(allocator, code_config),
            .fact_checker = FactChecker.initWithConfig(allocator, fact_config),
        };

        return self;
    }

    /// Shutdown the persona and free resources.
    pub fn deinit(self: *Self) void {
        self.knowledge_retriever.deinit();
        self.agent.deinit();
        self.allocator.destroy(self.agent);
        self.allocator.destroy(self);
    }

    pub fn getName(_: *const Self) []const u8 {
        return "Aviva";
    }

    pub fn getType(_: *const Self) types.PersonaType {
        return .aviva;
    }

    /// Process a request using Aviva's direct and expert logic.
    /// Note: returns anyerror to match PersonaInterface.VTable.process signature.
    /// Actual errors: TimerFailed, OutOfMemory, and errors from agent.process().
    pub fn process(self: *Self, request: types.PersonaRequest) anyerror!types.PersonaResponse {
        var timer = time.Timer.start() catch {
            return error.TimerFailed;
        };

        // Step 1: Classify the query
        const classification = self.classifier.classify(request.content);

        // Step 2: Retrieve relevant knowledge if factual query
        var knowledge_context: ?[]knowledge_mod.KnowledgeFragment = null;
        if (classification.query_type == .factual_query or
            classification.query_type == .documentation or
            classification.query_type == .explanation)
        {
            const retrieval = try self.knowledge_retriever.retrieve(
                request.content,
                classification.domain,
            );
            if (retrieval.fragments.items.len > 0) {
                knowledge_context = retrieval.fragments.items;
            }
        }

        // Step 3: Generate response via agent (apply per-request overrides)
        const applied = try apply(self.allocator, self.agent, request);
        defer restore(self.allocator, self.agent, applied);

        const raw_response = try self.agent.process(request.content, self.allocator);

        // Step 4: Post-process based on query type
        var final_response = raw_response;

        // For code requests, ensure proper formatting
        if (classification.query_type.recommendsCodeBlock()) {
            // Check if response contains code and format it
            const validation = self.code_generator.validateStructure(raw_response, classification.language);
            _ = validation;
        }

        // Step 5: Fact check the response
        var fact_result = try self.fact_checker.check(final_response);
        defer fact_result.deinit();

        // Add qualifications if needed and sources enabled
        if (self.config.cite_sources and fact_result.qualifications.items.len > 0) {
            const qualified = try facts_mod.applyQualifications(
                self.allocator,
                final_response,
                fact_result.qualifications.items,
            );
            if (final_response.ptr != raw_response.ptr) {
                self.allocator.free(final_response);
            }
            final_response = qualified;
        }

        const elapsed_ms = timer.read() / std.time.ns_per_ms;

        // Build response with classification metadata
        var response = types.PersonaResponse{
            .content = final_response,
            .persona = .aviva,
            .confidence = @min(0.95, fact_result.overall_confidence),
            .generation_time_ms = elapsed_ms,
        };

        // Add code blocks if detected
        if (classification.query_type.recommendsCodeBlock()) {
            const blocks = try code_mod.extractCodeBlocks(self.allocator, final_response);
            if (blocks.len > 0) {
                var code_blocks = try self.allocator.alloc(types.CodeBlock, blocks.len);
                for (blocks, 0..) |block, i| {
                    code_blocks[i] = .{
                        .language = self.code_generator.getLanguageName(block.language),
                        .code = block.code,
                    };
                }
                response.code_blocks = code_blocks;
            }
            self.allocator.free(blocks);
        }

        // Add sources if knowledge was retrieved
        if (knowledge_context) |sources| {
            if (self.config.cite_sources and sources.len > 0) {
                var refs = try self.allocator.alloc(types.Source, sources.len);
                for (sources, 0..) |frag, i| {
                    refs[i] = .{
                        .title = frag.source.name,
                        .url = frag.source.url,
                        .confidence = frag.confidence,
                    };
                }
                response.references = refs;
            }
        }

        return response;
    }

    /// Classify a query without full processing.
    pub fn classifyQuery(self: *const Self, query: []const u8) ClassificationResult {
        return self.classifier.classify(query);
    }

    /// Generate a code template for a function.
    pub fn generateFunctionTemplate(
        self: *Self,
        name: []const u8,
        params: []const []const u8,
        return_type: ?[]const u8,
        language: Language,
    ) !CodeBlock {
        return self.code_generator.generateFunctionTemplate(
            name,
            params,
            return_type,
            language,
            null,
        );
    }

    /// Score a statement's factual confidence.
    pub fn scoreFactualConfidence(self: *const Self, statement: []const u8) f32 {
        return self.fact_checker.scoreStatement(statement);
    }

    /// Create the interface wrapper for this persona.
    pub fn interface(self: *Self) types.PersonaInterface {
        return .{
            .ptr = self,
            .vtable = &.{
                .process = @ptrCast(&process),
                .getName = @ptrCast(&getName),
                .getType = @ptrCast(&getType),
            },
        };
    }
};

// --- Agent override helpers (formerly personas/agent_overrides.zig) ---

pub const Overrides = struct {
    prev_temperature: f32,
    prev_max_tokens: u32,
    system_index: ?usize = null,
};

pub fn apply(allocator: std.mem.Allocator, agent: *agent_mod.Agent, request: types.PersonaRequest) !Overrides {
    var ov = Overrides{
        .prev_temperature = agent.config.temperature,
        .prev_max_tokens = agent.config.max_tokens,
    };
    errdefer restore(allocator, agent, ov);

    if (request.temperature) |temp| {
        try agent.setTemperature(temp);
    }
    if (request.max_tokens) |tokens| {
        try agent.setMaxTokens(tokens);
    }
    if (request.system_instruction) |instruction| {
        const content_copy = try allocator.dupe(u8, instruction);
        errdefer allocator.free(content_copy);
        ov.system_index = agent.history.items.len;
        try agent.history.append(allocator, .{
            .role = .system,
            .content = content_copy,
        });
    }

    return ov;
}

pub fn restore(allocator: std.mem.Allocator, agent: *agent_mod.Agent, ov: Overrides) void {
    agent.config.temperature = ov.prev_temperature;
    agent.config.max_tokens = ov.prev_max_tokens;

    if (ov.system_index) |idx| {
        if (idx < agent.history.items.len) {
            const removed = agent.history.orderedRemove(idx);
            allocator.free(removed.content);
        }
    }
}

// Tests

test "aviva module re-exports" {
    // Verify re-exports work
    _ = QueryClassifier;
    _ = QueryType;
    _ = KnowledgeRetriever;
    _ = CodeGenerator;
    _ = FactChecker;
}

test "query type helpers" {
    try std.testing.expect(QueryType.code_request.recommendsCodeBlock());
    try std.testing.expect(QueryType.debugging.recommendsCodeBlock());
    try std.testing.expect(!QueryType.factual_query.recommendsCodeBlock());
    try std.testing.expect(QueryType.factual_query.recommendsBrevity());
}

test {
    std.testing.refAllDecls(@This());
}
