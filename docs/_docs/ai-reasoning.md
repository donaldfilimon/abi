---
title: "Reasoning"
description: "Abbey reasoning engine, RAG, evaluation, prompt templates, exploration, orchestration, and document understanding in the ABI Reasoning module."
section: "AI"
order: 5
---

# Reasoning

The Reasoning module provides advanced AI reasoning capabilities: the Abbey
engine with meta-learning, self-reflection, and theory of mind; retrieval-
augmented generation (RAG); evaluation frameworks; prompt templates; codebase
exploration; multi-model orchestration; and document understanding.

- **Build flag:** `-Denable-reasoning=true` (default: enabled)
- **Namespace:** `abi.reasoning`
- **Source:** `src/features/ai_reasoning/`

## Overview

The Reasoning module sits at the top of the AI stack. While
[Inference](ai-inference.html) runs models and [AI Core](ai-core.html)
manages agents, Reasoning provides the higher-order cognitive capabilities
that make AI outputs more accurate, grounded, and context-aware.

Key capabilities:

- **Abbey Engine** -- Advanced reasoning with neural attention, three-tier memory, confidence calibration, and emotional intelligence
- **RAG** -- Retrieval-Augmented Generation with document chunking, retrieval, and context building
- **Eval** -- Evaluation framework for measuring AI output quality
- **Templates** -- Prompt template system for reusable prompt patterns
- **Explore** -- Codebase exploration agent for understanding codebases
- **Orchestration** -- Multi-model routing, ensemble methods, and fallback policies
- **Documents** -- Document understanding pipeline with layout analysis and entity extraction

## Quick Start

```zig
const abi = @import("abi");

// Initialize reasoning context
var ctx = try abi.reasoning.Context.init(allocator, ai_config);
defer ctx.deinit();

// Create an Abbey engine for advanced reasoning
var engine = try abi.reasoning.abbey.createEngine(allocator);
defer engine.deinit();

// Build a RAG pipeline
var rag = abi.reasoning.rag.createPipeline(allocator);
defer rag.deinit();
```

## API Reference

### Core Types

**Abbey Engine:**

| Type | Description |
|------|-------------|
| `AbbeyEngine` | Main reasoning engine with neural attention and memory |
| `Abbey` | High-level Abbey interface |
| `AbbeyStats` | Engine performance and usage statistics |
| `ReasoningChain` | A chain of reasoning steps |
| `ReasoningStep` | Individual step in a reasoning chain |
| `ConversationContext` | Conversation tracking with topic and context windows |

**Explore:**

| Type | Description |
|------|-------------|
| `ExploreAgent` | Agent for codebase exploration and understanding |
| `ExploreConfig` | Exploration configuration |
| `ExploreLevel` | Depth of exploration (shallow, medium, deep) |
| `ExploreResult` | Result of an exploration query |
| `Match` | A matched item from exploration |
| `ExplorationStats` | Statistics about an exploration session |
| `QueryIntent` | Classified intent of a user query |
| `ParsedQuery` | Parsed and analyzed query |
| `QueryUnderstanding` | Deep understanding of what the user is asking |

**Orchestration:**

| Type | Description |
|------|-------------|
| `Orchestrator` | Multi-model orchestrator |
| `OrchestrationConfig` | Orchestration settings (strategy, model list) |
| `OrchestrationError` | Error set for orchestration operations |
| `RoutingStrategy` | How to route requests across models |
| `TaskType` | Classification of the task for routing |
| `RouteResult` | Result of a routing decision |
| `EnsembleMethod` | Method for combining multi-model outputs |
| `EnsembleResult` | Combined result from an ensemble |
| `FallbackPolicy` | Policy for handling model failures |
| `HealthStatus` | Health status of a model backend |
| `ModelBackend` | Backend type (OpenAI, Anthropic, Ollama, etc.) |
| `ModelCapability` | Capability flags for a model |
| `OrchestrationModelConfig` | Per-model configuration within orchestration |

**Documents:**

| Type | Description |
|------|-------------|
| `DocumentPipeline` | End-to-end document processing pipeline |
| `Document` | A parsed document |
| `DocumentFormat` | Supported document formats |
| `DocumentElement` | Structural element within a document |
| `ElementType` | Type of document element (heading, paragraph, table, etc.) |
| `TextSegment` | A segment of text extracted from a document |
| `TextSegmenter` | Splits text into meaningful segments |
| `NamedEntity` | A named entity extracted from text |
| `EntityType` | Type of named entity (person, org, location, etc.) |
| `EntityExtractor` | Extracts named entities from text |
| `LayoutAnalyzer` | Analyzes document layout and structure |
| `PipelineConfig` | Document pipeline configuration |
| `SegmentationConfig` | Text segmentation configuration |

### Key Functions

| Function | Description |
|----------|-------------|
| `isEnabled() bool` | Returns `true` if reasoning is compiled in |
| `abbey.createEngine(allocator) !AbbeyEngine` | Create a default Abbey engine |
| `abbey.createEngineWithConfig(allocator, config) !AbbeyEngine` | Create Abbey with custom config |
| `rag.createPipeline(allocator) RagPipeline` | Create a default RAG pipeline |
| `rag.createPipelineWithConfig(allocator, config) RagPipeline` | Create RAG with custom config |

## Abbey Reasoning Engine

Abbey is a comprehensive, opinionated, emotionally intelligent AI reasoning
framework. It combines several advanced cognitive capabilities:

### Neural Attention

Abbey uses a multi-head attention mechanism with adaptive attention for
focusing on relevant context:

```zig
const abbey = abi.reasoning.abbey;

// Types available through the neural module
const attention = abbey.neural.MultiHeadAttention;
const adaptive = abbey.neural.AdaptiveAttention;
const self_attn = abbey.neural.SelfAttention;
const cross_attn = abbey.neural.CrossAttention;
```

### Three-Tier Memory

Abbey maintains three distinct memory systems:

| Memory Tier | Type | Purpose |
|-------------|------|---------|
| Episodic | `EpisodicMemory` | Stores past interactions as episodes with temporal context |
| Semantic | `SemanticMemory` | Long-term knowledge store for facts and relationships |
| Working | `WorkingMemory` | Short-term buffer for active reasoning context |

```zig
const memory = abbey.memory;

// Access memory types
const episode = memory.Episode;
const episodic = memory.EpisodicMemory;
const semantic = memory.SemanticMemory;
const working = memory.WorkingMemory;
const manager = memory.MemoryManager;
```

### Confidence Calibration

Bayesian confidence calibration ensures Abbey reports accurate confidence
levels:

```zig
const calibration = abbey.calibration;

var calibrator = calibration.ConfidenceCalibrator.init(allocator);
defer calibrator.deinit();

// Analyze a query for difficulty and expected confidence
var analyzer = calibration.QueryAnalyzer.init(allocator);
```

### Meta-Learning

The meta-learning subsystem enables Abbey to learn from its own performance
patterns and adapt its reasoning strategies over time.

### Self-Reflection

Abbey can reflect on its own outputs, evaluate quality, and iteratively
improve responses through self-reflection loops.

### Theory of Mind

The theory of mind module helps Abbey model user intent, knowledge level,
and emotional state to produce more appropriate responses.

### Emotional Intelligence

Abbey tracks and responds to emotional context:

```zig
const EmotionType = abbey.EmotionType;
const EmotionalState = abbey.EmotionalState;
```

### Creating an Abbey Engine

```zig
var engine = try abi.reasoning.abbey.createEngine(allocator);
defer engine.deinit();

// Or with custom configuration
var custom_engine = try abi.reasoning.abbey.createEngineWithConfig(allocator, .{
    .behavior = .{ .temperature = 0.8 },
    .memory = .{ .max_episodes = 1000 },
    .reasoning = .{ .max_depth = 10 },
});
defer custom_engine.deinit();
```

## RAG (Retrieval-Augmented Generation)

The RAG subsystem grounds LLM responses in retrieved documents:

### Pipeline

```zig
const rag = abi.reasoning.rag;

// Create a RAG pipeline
var pipeline = rag.createPipeline(allocator);
defer pipeline.deinit();

// Add documents
try pipeline.addText("Zig is a systems programming language...", "Zig Overview");

// Query the pipeline
const response = try pipeline.query(allocator, "What is Zig?", .{});
defer response.deinit(allocator);
```

### Components

The RAG pipeline consists of three stages:

| Stage | Type | Purpose |
|-------|------|---------|
| Chunking | `Chunker` | Split documents into chunks with configurable strategy |
| Retrieval | `Retriever` | Find relevant chunks for a query |
| Context Building | `ContextBuilder` | Assemble retrieved chunks into a prompt context |

```zig
// Configure chunking strategy
const chunker_config = rag.ChunkerConfig{
    .strategy = .sliding_window,
    .chunk_size = 512,
    .overlap = 64,
};

// Configure retrieval
const retriever_config = rag.RetrieverConfig{
    .top_k = 5,
};
```

## Eval Framework

The evaluation module provides tools for measuring AI output quality:

```zig
const eval = abi.reasoning.eval;
// Set up evaluation benchmarks and metrics
```

## Prompt Templates

Reusable prompt templates for common patterns:

```zig
const templates = abi.reasoning.templates;
// Use or create prompt templates
```

## Explore (Codebase Exploration)

The Explore agent navigates and understands codebases. It is gated by
`-Denable-explore=true`:

```zig
const explore = abi.reasoning.explore;

var agent = try explore.ExploreAgent.init(allocator, .{
    .level = .deep,
});
defer agent.deinit();
```

### Query Understanding

The explore agent parses and classifies user queries to determine intent:

```zig
const intent = explore.QueryIntent;   // What the user wants to do
const parsed = explore.ParsedQuery;    // Structured query representation
const understanding = explore.QueryUnderstanding; // Deep query analysis
```

## Orchestration

The orchestration subsystem routes requests across multiple model backends
for reliability, quality, and cost optimization:

```zig
const orch = abi.reasoning.orchestration;

// Create an orchestrator
var orchestrator = try orch.Orchestrator.init(allocator, .defaults());
defer orchestrator.deinit();
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| Round-robin | Distribute evenly across backends |
| Least-latency | Route to the fastest backend |
| Cost-optimized | Minimize API costs |
| Quality-first | Route to the highest-quality backend |

### Ensemble Methods

Combine outputs from multiple models for higher quality:

| Method | Description |
|--------|-------------|
| Best-of-N | Run N models, pick the best output |
| Majority vote | Use consensus across models |
| Weighted average | Combine outputs with quality weights |

### Fallback Policies

Handle model failures gracefully:

```zig
const fallback_policy = orch.FallbackPolicy{
    // Configure retry, skip, or cascade behavior
};
```

### Preset Configurations

```zig
// High availability: maximize uptime with aggressive fallbacks
const ha = orch.OrchestrationConfig.highAvailability();

// High quality: prioritize output quality over latency
const hq = orch.OrchestrationConfig.highQuality();

// Default: balanced configuration
const defaults = orch.OrchestrationConfig.defaults();
```

### Model Backends

| Backend | Description |
|---------|-------------|
| `openai` | OpenAI API |
| `anthropic` | Anthropic Claude API |
| `ollama` | Local Ollama server |
| `huggingface` | HuggingFace Inference API |
| `mistral` | Mistral AI API |
| `cohere` | Cohere API |
| `local` | Local GGUF model via inference module |

## Document Understanding

The documents module processes and analyzes documents:

```zig
const docs = abi.reasoning.documents;

var pipeline = docs.DocumentPipeline.init(allocator, .{});
defer pipeline.deinit();
```

### Pipeline Stages

| Stage | Component | Purpose |
|-------|-----------|---------|
| Parsing | `DocumentPipeline` | Parse documents into structured elements |
| Layout | `LayoutAnalyzer` | Analyze page layout and structure |
| Segmentation | `TextSegmenter` | Split text into meaningful segments |
| Entity Extraction | `EntityExtractor` | Extract named entities |

### Supported Entity Types

The entity extractor recognizes persons, organizations, locations, and other
standard named entity categories.

## Configuration

The reasoning module context is lightweight since most submodules are
stateless collections of types and functions:

```zig
var ctx = try abi.reasoning.Context.init(allocator, ai_config);
defer ctx.deinit();
```

## Disabling at Build Time

```bash
# Compile without reasoning (disables Abbey, RAG, eval, orchestration)
zig build -Denable-reasoning=false

# Compile without explore only (keeps other reasoning features)
zig build -Denable-explore=false
```

When disabled, `Context.init()` returns `error.ReasoningDisabled` and
`isEnabled()` returns `false`. Individual submodules like `explore` can be
disabled independently via their own flags.

## Related

- [AI Overview](ai-overview.html) -- Architecture of all five AI modules
- [AI Core](ai-core.html) -- Agents, tools, prompts, personas
- [Inference](ai-inference.html) -- LLM execution, embeddings, streaming
- [Training](ai-training.html) -- Training pipelines, federated learning

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
