---
title: "multi-persona-ai-assistant"
tags: []
---
# Multi-Persona AI Assistant Architecture

## Overview

This document describes the architecture for implementing a multi-layer, multi-persona AI assistant system based on the research paper "Extended Multi-Layer, Multi-Persona AI Assistant with WDBX." The architecture leverages existing ABI framework components while introducing new modules for persona routing, content moderation, and distributed inference.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Request                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API Gateway Layer                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Rate Limiter│  │Auth/RBAC   │  │ Request     │  │ Session Manager     │ │
│  │             │  │             │  │ Validator   │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Abi: Content Moderation & Routing Layer                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐   │
│  │ Sentiment Analysis│  │ Policy Checker    │  │ Persona Router        │   │
│  │ - Emotion detect  │  │ - Content filter  │  │ - Intent classifier   │   │
│  │ - Urgency score   │  │ - Safety rules    │  │ - Persona scoring     │   │
│  │ - Context capture │  │ - Compliance      │  │ - Load balancing      │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────┐
│   Abbey: Empathetic     │ │  Aviva: Direct Expert   │ │ Custom Personas     │
│   Polymath              │ │                         │ │                     │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │ ┌─────────────────┐ │
│ │ Emotion Processing  │ │ │ │ Fact Retrieval      │ │ │ │ Domain-Specific │ │
│ │ - Empathy injection │ │ │ │ - Knowledge base    │ │ │ │ - Healthcare    │ │
│ │ - Tone adaptation   │ │ │ │ - Code expertise    │ │ │ │ - Legal         │ │
│ │ - Support patterns  │ │ │ │ - Minimal tone      │ │ │ │ - Creative      │ │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │ └─────────────────┘ │
│ ┌─────────────────────┐ │ │ ┌─────────────────────┐ │ │                     │
│ │ Reasoning Chain     │ │ │ │ Direct Response     │ │ │                     │
│ │ - Step-by-step      │ │ │ │ - No disclaimers    │ │ │                     │
│ │ - Confidence scores │ │ │ │ - Code examples     │ │ │                     │
│ └─────────────────────┘ │ │ └─────────────────────┘ │ │                     │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Response Aggregation & Validation                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Response Merger │  │ Quality Check   │  │ Final Policy Compliance     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Response                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Abi: Content Moderation & Routing Layer

**Location**: `src/ai/personas/abi/`

Abi serves as the gatekeeper, handling:
- **Sentiment Analysis**: Detect user emotional state to route appropriately
- **Policy Checking**: Ensure content complies with safety guidelines
- **Persona Routing**: Select optimal persona based on query analysis

```zig
// src/ai/personas/abi/mod.zig
pub const AbiRouter = struct {
    allocator: Allocator,
    sentiment_analyzer: SentimentAnalyzer,
    policy_checker: PolicyChecker,
    persona_scorer: PersonaScorer,
    load_balancer: PersonaLoadBalancer,

    pub fn init(allocator: Allocator, config: AbiConfig) !AbiRouter;
    pub fn route(self: *AbiRouter, request: UserRequest) !RoutingDecision;
    pub fn validateResponse(self: *AbiRouter, response: Response) !ValidationResult;
};

pub const RoutingDecision = struct {
    selected_persona: PersonaType,
    confidence: f32,
    emotional_context: EmotionalState,
    policy_flags: PolicyFlags,
    routing_reason: []const u8,
};
```

#### 1.1 Sentiment Analysis Component

```zig
// src/ai/personas/abi/sentiment.zig
pub const SentimentAnalyzer = struct {
    emotion_classifier: *EmotionClassifier,
    urgency_detector: *UrgencyDetector,

    pub fn analyze(self: *SentimentAnalyzer, text: []const u8) !SentimentResult {
        const emotion = try self.emotion_classifier.classify(text);
        const urgency = try self.urgency_detector.detect(text);

        return SentimentResult{
            .primary_emotion = emotion.primary,
            .secondary_emotions = emotion.secondary,
            .urgency_score = urgency.score,
            .requires_empathy = emotion.requires_support,
            .is_technical = try self.detectTechnicalContent(text),
        };
    }
};

pub const SentimentResult = struct {
    primary_emotion: EmotionType,
    secondary_emotions: []const EmotionType,
    urgency_score: f32, // 0.0 - 1.0
    requires_empathy: bool,
    is_technical: bool,
};
```

#### 1.2 Policy Checker Component

```zig
// src/ai/personas/abi/policy.zig
pub const PolicyChecker = struct {
    content_filter: *ContentFilter,
    safety_rules: []const SafetyRule,
    compliance_checker: *ComplianceChecker,

    pub fn check(self: *PolicyChecker, content: []const u8) !PolicyResult {
        // Check against content filters
        const filter_result = try self.content_filter.scan(content);

        // Check safety rules
        const safety_result = try self.checkSafetyRules(content);

        // Check regulatory compliance (GDPR, CCPA, etc.)
        const compliance_result = try self.compliance_checker.verify(content);

        return PolicyResult{
            .is_allowed = filter_result.is_safe and safety_result.passed and compliance_result.compliant,
            .violations = filter_result.violations ++ safety_result.violations,
            .requires_moderation = filter_result.needs_review,
            .suggested_action = self.determineSuggestedAction(filter_result, safety_result),
        };
    }
};

pub const SafetyRule = struct {
    name: []const u8,
    pattern: ?[]const u8,
    action: SafetyAction,
    severity: Severity,
};

pub const SafetyAction = enum {
    allow,
    warn,
    block,
    redirect_to_support,
    require_human_review,
};
```

#### 1.3 Persona Router Component

```zig
// src/ai/personas/abi/router.zig
pub const PersonaRouter = struct {
    persona_registry: *PersonaRegistry,
    embedding_index: *EmbeddingIndex, // WDBX-backed
    attention_scorer: *MultiHeadAttention,
    load_balancer: *PersonaLoadBalancer,

    pub fn selectPersona(
        self: *PersonaRouter,
        request: UserRequest,
        sentiment: SentimentResult,
        context: ConversationContext,
    ) !RoutingDecision {
        // Generate query embedding
        const query_embedding = try self.generateQueryEmbedding(request.content);

        // Score each persona using attention mechanism
        var persona_scores = std.AutoHashMap(PersonaType, f32).init(self.allocator);
        defer persona_scores.deinit();

        for (self.persona_registry.getActivePersonas()) |persona| {
            const score = try self.scorePersona(persona, query_embedding, sentiment, context);
            try persona_scores.put(persona.persona_type, score);
        }

        // Apply routing rules
        const routing_rules_score = try self.applyRoutingRules(request, sentiment);

        // Combine scores with weighted average
        const final_scores = try self.combineScores(persona_scores, routing_rules_score);

        // Select best persona with load balancing consideration
        const selected = try self.load_balancer.selectWithScores(final_scores);

        return RoutingDecision{
            .selected_persona = selected.persona_type,
            .confidence = selected.score,
            .emotional_context = sentiment.toEmotionalState(),
            .policy_flags = .{},
            .routing_reason = try self.generateRoutingReason(selected),
        };
    }

    fn applyRoutingRules(self: *PersonaRouter, request: UserRequest, sentiment: SentimentResult) !RoutingRulesScore {
        var rules_score = RoutingRulesScore{};

        // Rule 1: High urgency + negative emotion -> Abbey
        if (sentiment.urgency_score > 0.7 and sentiment.requires_empathy) {
            rules_score.abbey_boost = 0.3;
        }

        // Rule 2: Technical query without emotional content -> Aviva
        if (sentiment.is_technical and !sentiment.requires_empathy) {
            rules_score.aviva_boost = 0.25;
        }

        // Rule 3: Sensitive topics -> Policy redirect
        if (try self.detectSensitiveTopic(request.content)) {
            rules_score.requires_moderation = true;
        }

        return rules_score;
    }
};
```

### 2. Abbey: Empathetic Polymath Persona

**Location**: `src/ai/personas/abbey/` (extends existing `src/ai/implementation/abbey/`)

Abbey combines emotional intelligence with technical expertise.

```zig
// src/ai/personas/abbey/mod.zig
pub const AbbeyPersona = struct {
    config: AbbeyConfig,
    emotion_processor: *EmotionProcessor,
    reasoning_engine: *ReasoningEngine,
    memory_system: *ThreeTierMemory,
    llm_client: *LLMClient,

    pub fn process(self: *AbbeyPersona, request: PersonaRequest) !PersonaResponse {
        // Step 1: Process emotional context
        const emotional_response = try self.emotion_processor.process(
            request.content,
            request.emotional_context,
        );

        // Step 2: Retrieve relevant context from memory
        const memory_context = try self.memory_system.retrieveContext(
            request.content,
            request.session_id,
        );

        // Step 3: Build reasoning chain
        const reasoning = try self.reasoning_engine.reason(
            request.content,
            memory_context,
            emotional_response,
        );

        // Step 4: Generate empathetic response
        const response = try self.generateResponse(
            request,
            reasoning,
            emotional_response,
        );

        // Step 5: Update memory
        try self.memory_system.store(request, response);

        return response;
    }

    fn generateResponse(
        self: *AbbeyPersona,
        request: PersonaRequest,
        reasoning: ReasoningChain,
        emotional_response: EmotionalResponse,
    ) !PersonaResponse {
        // Build empathetic prompt
        const prompt = try self.buildEmpathyPrompt(
            request,
            reasoning,
            emotional_response,
        );

        // Generate with appropriate temperature for empathy
        const llm_response = try self.llm_client.generate(.{
            .prompt = prompt,
            .temperature = 0.7, // Slightly higher for more natural empathy
            .max_tokens = 2048,
        });

        return PersonaResponse{
            .content = llm_response.text,
            .persona = .abbey,
            .confidence = reasoning.confidence,
            .emotional_tone = emotional_response.suggested_tone,
            .reasoning_chain = reasoning.steps,
        };
    }
};

pub const AbbeyConfig = struct {
    empathy_level: f32 = 0.8, // 0.0 - 1.0
    technical_depth: f32 = 0.7,
    include_reasoning: bool = true,
    max_reasoning_steps: u32 = 5,
    emotion_adaptation: bool = true,
};
```

### 3. Aviva: Direct Expert Persona

**Location**: `src/ai/personas/aviva/`

Aviva provides concise, factual responses without emotional overhead.

```zig
// src/ai/personas/aviva/mod.zig
pub const AvivaPersona = struct {
    config: AvivaConfig,
    knowledge_retriever: *KnowledgeRetriever,
    code_generator: *CodeGenerator,
    fact_checker: *FactChecker,
    llm_client: *LLMClient,

    pub fn process(self: *AvivaPersona, request: PersonaRequest) !PersonaResponse {
        // Step 1: Classify query type
        const query_type = try self.classifyQuery(request.content);

        // Step 2: Retrieve relevant knowledge
        const knowledge = try self.knowledge_retriever.retrieve(
            request.content,
            query_type,
        );

        // Step 3: Generate response based on query type
        const response = switch (query_type) {
            .code_request => try self.generateCodeResponse(request, knowledge),
            .factual_query => try self.generateFactualResponse(request, knowledge),
            .explanation => try self.generateExplanation(request, knowledge),
            else => try self.generateGenericResponse(request, knowledge),
        };

        // Step 4: Verify facts if applicable
        if (self.config.verify_facts) {
            try self.fact_checker.verify(response);
        }

        return response;
    }

    fn generateCodeResponse(
        self: *AvivaPersona,
        request: PersonaRequest,
        knowledge: Knowledge,
    ) !PersonaResponse {
        // Generate code with minimal commentary
        const code = try self.code_generator.generate(.{
            .request = request.content,
            .context = knowledge,
            .include_comments = self.config.include_code_comments,
            .language = try self.detectLanguage(request.content),
        });

        return PersonaResponse{
            .content = code.formatted,
            .persona = .aviva,
            .confidence = code.confidence,
            .code_blocks = code.blocks,
            .references = knowledge.sources,
        };
    }
};

pub const AvivaConfig = struct {
    directness_level: f32 = 0.9, // 0.0 - 1.0
    include_disclaimers: bool = false,
    include_code_comments: bool = true,
    verify_facts: bool = true,
    max_response_length: u32 = 4096,
};
```

### 4. WDBX Integration for Persona Embeddings

**Location**: `src/ai/personas/embeddings/`

```zig
// src/ai/personas/embeddings/persona_index.zig
pub const PersonaEmbeddingIndex = struct {
    database: *wdbx.Database,
    embedding_model: *EmbeddingModel,
    persona_vectors: std.AutoHashMap(PersonaType, []const f32),

    pub fn init(allocator: Allocator, config: EmbeddingConfig) !PersonaEmbeddingIndex {
        const db = try wdbx.Database.init(allocator, .{
            .name = "persona_embeddings",
            .enable_hnsw = true,
            .hnsw_m = 16,
            .hnsw_ef_construction = 200,
        });

        return PersonaEmbeddingIndex{
            .database = db,
            .embedding_model = try EmbeddingModel.init(config.model_path),
            .persona_vectors = std.AutoHashMap(PersonaType, []const f32).init(allocator),
        };
    }

    /// Store persona characteristic embedding
    pub fn storePersonaEmbedding(
        self: *PersonaEmbeddingIndex,
        persona: PersonaType,
        characteristics: []const u8,
    ) !void {
        const embedding = try self.embedding_model.embed(characteristics);
        try self.persona_vectors.put(persona, embedding);

        try self.database.upsert(.{
            .id = @intFromEnum(persona),
            .vector = embedding,
            .metadata = try std.json.stringifyAlloc(self.allocator, .{
                .persona = @tagName(persona),
                .characteristics = characteristics,
            }),
        });
    }

    /// Find best matching persona for a query
    pub fn findBestPersona(
        self: *PersonaEmbeddingIndex,
        query: []const u8,
        top_k: usize,
    ) ![]PersonaMatch {
        const query_embedding = try self.embedding_model.embed(query);

        const results = try self.database.search(.{
            .query_vector = query_embedding,
            .k = top_k,
            .include_metadata = true,
        });

        var matches = std.ArrayList(PersonaMatch).init(self.allocator);
        for (results) |result| {
            try matches.append(.{
                .persona = try self.parsePersonaFromMetadata(result.metadata),
                .similarity = 1.0 - result.distance, // Convert distance to similarity
                .metadata = result.metadata,
            });
        }

        return matches.toOwnedSlice();
    }

    /// Store conversation embedding for persona learning
    pub fn storeConversationEmbedding(
        self: *PersonaEmbeddingIndex,
        conversation_id: u64,
        content: []const u8,
        persona_used: PersonaType,
        success_score: f32,
    ) !void {
        const embedding = try self.embedding_model.embed(content);

        try self.database.upsert(.{
            .id = conversation_id,
            .vector = embedding,
            .metadata = try std.json.stringifyAlloc(self.allocator, .{
                .persona = @tagName(persona_used),
                .success_score = success_score,
                .timestamp = std.time.timestamp(),
            }),
        });
    }
};

pub const PersonaMatch = struct {
    persona: PersonaType,
    similarity: f32,
    metadata: ?[]const u8,
};
```

### 5. Persona Registry & Configuration

**Location**: `src/ai/personas/registry.zig`

```zig
// src/ai/personas/registry.zig
pub const PersonaRegistry = struct {
    allocator: Allocator,
    personas: std.AutoHashMap(PersonaType, *Persona),
    configs: std.AutoHashMap(PersonaType, PersonaConfig),
    metrics: *PersonaMetrics,

    pub fn init(allocator: Allocator) !PersonaRegistry {
        var registry = PersonaRegistry{
            .allocator = allocator,
            .personas = std.AutoHashMap(PersonaType, *Persona).init(allocator),
            .configs = std.AutoHashMap(PersonaType, PersonaConfig).init(allocator),
            .metrics = try PersonaMetrics.init(allocator),
        };

        // Register default personas
        try registry.registerPersona(.abbey, AbbeyPersona.default());
        try registry.registerPersona(.aviva, AvivaPersona.default());
        try registry.registerPersona(.abi, AbiPersona.default());

        return registry;
    }

    pub fn registerPersona(
        self: *PersonaRegistry,
        persona_type: PersonaType,
        persona: *Persona,
    ) !void {
        try self.personas.put(persona_type, persona);
        try self.metrics.registerPersona(persona_type);
    }

    pub fn getPersona(self: *PersonaRegistry, persona_type: PersonaType) ?*Persona {
        return self.personas.get(persona_type);
    }

    pub fn getActivePersonas(self: *PersonaRegistry) []const *Persona {
        var active = std.ArrayList(*Persona).init(self.allocator);
        var iter = self.personas.iterator();
        while (iter.next()) |entry| {
            if (self.isPersonaActive(entry.key_ptr.*)) {
                active.append(entry.value_ptr.*) catch continue;
            }
        }
        return active.toOwnedSlice() catch &[_]*Persona{};
    }
};

pub const PersonaConfig = struct {
    enabled: bool = true,
    max_concurrent_requests: u32 = 100,
    timeout_ms: u64 = 30000,
    priority: u8 = 5, // 1-10
    routing_weight: f32 = 1.0,
    specialized_domains: []const []const u8 = &[_][]const u8{},
};
```

### 6. Metrics & Observability

**Location**: `src/ai/personas/metrics.zig`

```zig
// src/ai/personas/metrics.zig
pub const PersonaMetrics = struct {
    allocator: Allocator,
    counters: std.AutoHashMap(MetricKey, u64),
    histograms: std.AutoHashMap(MetricKey, *Histogram),
    gauges: std.AutoHashMap(MetricKey, f64),

    pub fn recordRequest(self: *PersonaMetrics, persona: PersonaType) !void {
        const key = MetricKey{ .persona = persona, .metric = "requests_total" };
        const current = self.counters.get(key) orelse 0;
        try self.counters.put(key, current + 1);
    }

    pub fn recordLatency(self: *PersonaMetrics, persona: PersonaType, latency_ms: u64) !void {
        const key = MetricKey{ .persona = persona, .metric = "latency_ms" };
        if (self.histograms.get(key)) |histogram| {
            try histogram.observe(@floatFromInt(latency_ms));
        }
    }

    pub fn recordSuccess(self: *PersonaMetrics, persona: PersonaType, success: bool) !void {
        const suffix = if (success) "success" else "failure";
        const key = MetricKey{ .persona = persona, .metric = suffix };
        const current = self.counters.get(key) orelse 0;
        try self.counters.put(key, current + 1);
    }

    pub fn recordUserSatisfaction(self: *PersonaMetrics, persona: PersonaType, score: f32) !void {
        const key = MetricKey{ .persona = persona, .metric = "satisfaction_score" };
        if (self.histograms.get(key)) |histogram| {
            try histogram.observe(score);
        }
    }

    pub fn getPersonaStats(self: *PersonaMetrics, persona: PersonaType) PersonaStats {
        return PersonaStats{
            .total_requests = self.getCounter(persona, "requests_total"),
            .success_rate = self.calculateSuccessRate(persona),
            .avg_latency_ms = self.getHistogramMean(persona, "latency_ms"),
            .p99_latency_ms = self.getHistogramP99(persona, "latency_ms"),
            .avg_satisfaction = self.getHistogramMean(persona, "satisfaction_score"),
        };
    }
};

pub const PersonaStats = struct {
    total_requests: u64,
    success_rate: f32,
    avg_latency_ms: f64,
    p99_latency_ms: f64,
    avg_satisfaction: f32,
};
```

## Data Flow

### Request Processing Flow

```
1. User Request
   │
   ▼
2. API Gateway
   ├── Rate limiting check
   ├── Authentication/RBAC
   └── Request validation
   │
   ▼
3. Abi Layer (Content Moderation & Routing)
   ├── Sentiment analysis
   │   ├── Emotion detection
   │   ├── Urgency scoring
   │   └── Technical content detection
   ├── Policy checking
   │   ├── Content filtering
   │   ├── Safety rule evaluation
   │   └── Compliance verification
   └── Persona routing
       ├── Query embedding generation
       ├── Persona scoring via attention
       ├── Rule-based adjustments
       └── Load-balanced selection
   │
   ▼
4. Selected Persona Processing
   │
   ├── Abbey (if selected)
   │   ├── Emotion processing
   │   ├── Memory retrieval
   │   ├── Reasoning chain construction
   │   └── Empathetic response generation
   │
   └── Aviva (if selected)
       ├── Query classification
       ├── Knowledge retrieval
       ├── Fact verification
       └── Direct response generation
   │
   ▼
5. Response Validation
   ├── Quality check
   ├── Policy compliance
   └── Confidence threshold
   │
   ▼
6. User Response
```

### Memory Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Three-Tier Memory System                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Working Memory  │  │ Episodic Memory │  │ Semantic    │  │
│  │ (Short-term)    │  │ (Events)        │  │ Memory      │  │
│  │                 │  │                 │  │ (Knowledge) │  │
│  │ • Current ctx   │  │ • Conversations │  │ • Facts     │  │
│  │ • Recent turns  │  │ • User prefs    │  │ • Concepts  │  │
│  │ • Emotional st. │  │ • Outcomes      │  │ • Relations │  │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
│           │                    │                   │         │
│           └────────────────────┼───────────────────┘         │
│                                │                             │
│                    ┌───────────▼───────────┐                │
│                    │    WDBX Vector DB     │                │
│                    │  • Embedding storage  │                │
│                    │  • Hybrid search      │                │
│                    │  • Persona learning   │                │
│                    └───────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Schema

```zig
// src/ai/personas/config.zig
pub const MultiPersonaConfig = struct {
    /// Global settings
    default_persona: PersonaType = .abbey,
    enable_dynamic_routing: bool = true,
    routing_confidence_threshold: f32 = 0.6,

    /// Abi (Router) settings
    abi: AbiConfig = .{},

    /// Abbey settings
    abbey: AbbeyConfig = .{},

    /// Aviva settings
    aviva: AvivaConfig = .{},

    /// WDBX integration
    embeddings: EmbeddingConfig = .{},

    /// Load balancing
    load_balancing: LoadBalancingConfig = .{},

    /// Metrics
    metrics: MetricsConfig = .{},
};

pub const AbiConfig = struct {
    enable_sentiment_analysis: bool = true,
    enable_policy_checking: bool = true,
    sensitive_topic_detection: bool = true,
    content_filter_level: FilterLevel = .moderate,
    max_routing_latency_ms: u64 = 50,
};

pub const LoadBalancingConfig = struct {
    strategy: LoadBalancerStrategy = .health_weighted,
    enable_circuit_breaker: bool = true,
    circuit_breaker_threshold: u32 = 5,
    circuit_breaker_timeout_ms: u64 = 30000,
};
```

## File Structure

```
src/ai/personas/
├── mod.zig                    # Module entry point
├── config.zig                 # Configuration definitions
├── registry.zig               # Persona registry
├── metrics.zig                # Observability/metrics
├── types.zig                  # Shared types
│
├── abi/                       # Abi: Moderation & Routing
│   ├── mod.zig
│   ├── sentiment.zig          # Sentiment analysis
│   ├── policy.zig             # Policy checking
│   ├── router.zig             # Persona routing
│   └── rules.zig              # Routing rules
│
├── abbey/                     # Abbey: Empathetic Polymath
│   ├── mod.zig
│   ├── emotion.zig            # Emotion processing
│   ├── empathy.zig            # Empathy injection
│   └── reasoning.zig          # Enhanced reasoning
│
├── aviva/                     # Aviva: Direct Expert
│   ├── mod.zig
│   ├── knowledge.zig          # Knowledge retrieval
│   ├── code.zig               # Code generation
│   └── facts.zig              # Fact checking
│
├── embeddings/                # WDBX Integration
│   ├── mod.zig
│   ├── persona_index.zig      # Persona embeddings
│   └── learning.zig           # Adaptive learning
│
└── tests/                     # Unit tests
    ├── abi_test.zig
    ├── abbey_test.zig
    ├── aviva_test.zig
    └── integration_test.zig
```

## Integration with Existing Codebase

### 1. Extend Existing Persona System

The existing `src/ai/implementation/prompts/personas.zig` will be extended:

```zig
// Add to PersonaType enum
pub const PersonaType = enum {
    assistant,
    coder,
    writer,
    analyst,
    companion,
    docs,
    reviewer,
    minimal,
    abbey,    // Enhanced
    aviva,    // New
    abi,      // New (routing/moderation)
    ralph,
};
```

### 2. Integrate with Abbey Engine

Leverage existing Abbey engine at `src/ai/implementation/abbey/engine.zig`:

```zig
// Extend ProcessingPipeline to support persona-aware processing
pub const PersonaAwarePipeline = struct {
    base_pipeline: *ProcessingPipeline,
    persona_router: *PersonaRouter,

    pub fn process(self: *PersonaAwarePipeline, request: Request) !Response {
        // Route through Abi
        const routing = try self.persona_router.route(request);

        // Process with selected persona
        return switch (routing.selected_persona) {
            .abbey => try self.processWithAbbey(request, routing),
            .aviva => try self.processWithAviva(request, routing),
            else => try self.base_pipeline.process(request),
        };
    }
};
```

### 3. Integrate with WDBX Database

Leverage existing database at `src/database/database.zig`:

```zig
// Create persona-specific database instance
const persona_db = try wdbx.Database.init(allocator, .{
    .name = "personas",
    .enable_hnsw = true,
    .enable_hybrid_search = true,
});
```

### 4. Integrate with Load Balancer

Leverage existing load balancer at `src/network/loadbalancer.zig`:

```zig
// Create persona-specific load balancer
const persona_lb = try LoadBalancer.init(allocator, .{
    .strategy = .health_weighted,
    .health_check_interval_ms = 5000,
});

// Register personas as nodes
try persona_lb.registerNode(.{
    .id = "abbey",
    .weight = 10,
});
try persona_lb.registerNode(.{
    .id = "aviva",
    .weight = 10,
});
```

## Testing Strategy

### Unit Tests

```zig
// src/ai/personas/tests/abi_test.zig
test "sentiment analysis detects frustration" {
    const analyzer = try SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = try analyzer.analyze("This is so frustrating! Nothing works!");

    try testing.expect(result.primary_emotion == .frustrated);
    try testing.expect(result.requires_empathy);
    try testing.expect(result.urgency_score > 0.5);
}

test "policy checker blocks harmful content" {
    const checker = try PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    const result = try checker.check("[harmful content]");

    try testing.expect(!result.is_allowed);
    try testing.expect(result.violations.len > 0);
}

test "persona router selects Abbey for emotional queries" {
    const router = try PersonaRouter.init(testing.allocator);
    defer router.deinit();

    const decision = try router.selectPersona(.{
        .content = "I'm feeling really overwhelmed with work",
        .emotional_context = .{ .detected = .stressed },
    });

    try testing.expect(decision.selected_persona == .abbey);
    try testing.expect(decision.confidence > 0.7);
}
```

### Integration Tests

```zig
// src/ai/personas/tests/integration_test.zig
test "full request flow through persona system" {
    // Initialize system
    var system = try MultiPersonaSystem.init(testing.allocator);
    defer system.deinit();

    // Test empathetic query -> Abbey
    {
        const response = try system.process(.{
            .content = "I'm struggling with this bug and feeling stuck",
            .session_id = 1,
        });

        try testing.expect(response.persona == .abbey);
        try testing.expect(std.mem.indexOf(u8, response.content, "understand") != null);
    }

    // Test technical query -> Aviva
    {
        const response = try system.process(.{
            .content = "How do I implement a binary search tree in Zig?",
            .session_id = 2,
        });

        try testing.expect(response.persona == .aviva);
        try testing.expect(response.code_blocks.len > 0);
    }
}
```

## Performance Considerations

### Latency Budget

| Component | Target Latency |
|-----------|----------------|
| API Gateway | < 5ms |
| Sentiment Analysis | < 10ms |
| Policy Check | < 5ms |
| Persona Routing | < 20ms |
| Embedding Lookup | < 15ms |
| LLM Generation | < 2000ms |
| Response Validation | < 10ms |
| **Total** | **< 2100ms** |

### Optimization Strategies

1. **Parallel Processing**: Run sentiment analysis and policy checking concurrently
2. **Embedding Caching**: Cache frequently used query embeddings
3. **Warm Persona Pools**: Pre-initialize persona instances
4. **Streaming Responses**: Stream LLM output for perceived lower latency
5. **Circuit Breakers**: Prevent cascade failures with per-persona circuit breakers

## Security Considerations

1. **Input Sanitization**: All user input sanitized before processing
2. **Content Filtering**: Multi-layer content filtering in Abi
3. **Rate Limiting**: Per-user and per-IP rate limiting
4. **RBAC Integration**: Persona access controlled via existing RBAC system
5. **Audit Logging**: All persona decisions logged for audit
6. **PII Detection**: Automatic PII detection and redaction
7. **Encryption**: All data encrypted at rest and in transit

## Future Extensions

1. **Additional Personas**: Healthcare, Legal, Creative Arts, Financial
2. **Multimodal Support**: Voice, image, and video processing
3. **Federated Learning**: Cross-session persona improvement
4. **A/B Testing**: Built-in experimentation framework
5. **User Personalization**: Per-user persona preferences
6. **Multi-turn Memory**: Enhanced conversation continuity

