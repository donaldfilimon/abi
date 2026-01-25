# AI Module Architecture

> **Last Updated:** 2026-01-25
> **Module Location:** `src/ai/`
> **Primary Entry Point:** `src/ai/mod.zig`

This document provides a comprehensive technical overview of the AI module architecture, including all submodules, their interactions, and design patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [Core Subsystems](#core-subsystems)
4. [LLM Engine](#llm-engine)
5. [Agent System](#agent-system)
6. [Memory Architecture](#memory-architecture)
7. [Training Pipeline](#training-pipeline)
8. [Orchestration Layer](#orchestration-layer)
9. [RAG Pipeline](#rag-pipeline)
10. [Abbey Cognitive Framework](#abbey-cognitive-framework)
11. [Persona System](#persona-system)
12. [Explore Module](#explore-module)
13. [Feature Flags & Configuration](#feature-flags--configuration)
14. [Integration Patterns](#integration-patterns)

---

## Overview

The AI module (`abi.ai`) is a comprehensive, modular AI system built in pure Zig. It provides:

- **Local LLM inference** with GGUF model support
- **Multi-agent coordination** with tool integration
- **Hierarchical memory systems** (short-term, sliding window, summarizing, long-term)
- **Training pipelines** with gradient accumulation, checkpointing, and LoRA
- **Multi-model orchestration** with routing, fallback, and ensemble methods
- **Retrieval-Augmented Generation (RAG)** pipeline
- **Advanced cognitive framework** (Abbey) with emotional intelligence
- **Persona-based routing** for multi-persona AI assistants
- **Codebase exploration** with parallel search and AST analysis

### Design Principles

1. **Modularity**: Each subsystem is independently compilable with stub fallbacks
2. **Feature Flags**: All major components can be enabled/disabled at compile time
3. **Zero Dependencies**: Pure Zig implementation without external AI libraries
4. **GPU Acceleration**: Optional CUDA/Vulkan/Metal support with CPU fallback
5. **Type Safety**: Comptime configuration and generic patterns throughout

---

## Module Structure

```
src/ai/
├── mod.zig                 # Public API entry point
├── stub.zig                # Disabled feature stub
├── agent.zig               # Main Agent implementation
├── gpu_agent.zig           # GPU-aware agent with RL scheduling
├── model_registry.zig      # Model registration and discovery
│
├── core/                   # Core types and configuration
│   ├── mod.zig             # Public exports
│   ├── types.zig           # InstanceId, SessionId, Confidence, EmotionalState
│   └── config.zig          # AbbeyConfig, BehaviorConfig, MemoryConfig
│
├── llm/                    # Local LLM inference engine
│   ├── mod.zig             # LLM entry point
│   ├── stub.zig            # Disabled stub
│   ├── parallel.zig        # Tensor/pipeline parallelism
│   ├── io/                 # Model I/O
│   │   ├── gguf.zig        # GGUF format reader
│   │   ├── gguf_writer.zig # GGUF export
│   │   ├── mmap.zig        # Memory-mapped file access
│   │   └── tensor_loader.zig
│   ├── tensor/             # Tensor operations
│   │   ├── tensor.zig      # Core Tensor type
│   │   ├── quantized.zig   # Q4_0, Q8_0 quantization
│   │   └── view.zig        # Tensor views
│   ├── tokenizer/          # Tokenization
│   │   ├── bpe.zig         # BPE tokenizer
│   │   ├── sentencepiece.zig # SentencePiece (Viterbi)
│   │   ├── vocab.zig       # Vocabulary management
│   │   └── special_tokens.zig
│   ├── ops/                # Neural network operations
│   │   ├── matmul.zig      # Matrix multiplication
│   │   ├── matmul_quant.zig # Quantized matmul
│   │   ├── attention.zig   # Multi-head attention
│   │   ├── rope.zig        # Rotary position embeddings
│   │   ├── rmsnorm.zig     # RMS normalization
│   │   ├── activations.zig # SiLU, GELU, etc.
│   │   ├── ffn.zig         # Feed-forward networks
│   │   ├── gpu.zig         # GPU kernel dispatch
│   │   ├── gpu_memory_pool.zig
│   │   └── backward/       # Backward pass implementations
│   ├── cache/              # KV cache implementations
│   │   ├── kv_cache.zig    # Standard KV cache
│   │   ├── paged_kv_cache.zig # vLLM-style paged attention
│   │   └── ring_buffer.zig # Sliding window cache
│   ├── model/              # Model architectures
│   │   ├── llama.zig       # LLaMA implementation
│   │   ├── layer.zig       # Transformer layer
│   │   ├── weights.zig     # Weight loading
│   │   └── config.zig      # Model configuration
│   └── generation/         # Text generation
│       ├── generator.zig   # Token generation loop
│       ├── sampler.zig     # Sampling strategies
│       ├── streaming.zig   # Streaming response
│       └── batch.zig       # Batch generation
│
├── agents/                 # Agent runtime
│   ├── mod.zig             # Agent context and management
│   └── stub.zig            # Disabled stub
│
├── tools/                  # Agent tools
│   ├── mod.zig             # Tool registry
│   ├── subagent.zig        # Sub-agent spawning
│   ├── discord.zig         # Discord tools
│   └── os.zig              # OS/shell tools
│
├── memory/                 # Conversation memory
│   ├── mod.zig             # Memory types and manager
│   ├── short_term.zig      # Fixed-capacity message buffer
│   ├── window.zig          # Token-based sliding window
│   ├── summary.zig         # Compression to summaries
│   ├── long_term.zig       # Vector-based retrieval
│   ├── manager.zig         # Multi-strategy manager
│   └── persistence.zig     # Session persistence
│
├── training/               # Training pipelines
│   ├── mod.zig             # Training orchestration
│   ├── stub.zig            # Disabled stub
│   ├── checkpoint.zig      # Checkpoint management
│   ├── llm_checkpoint.zig  # LLM-specific checkpoints
│   ├── gradient.zig        # Gradient accumulation
│   ├── loss.zig            # Loss functions
│   ├── trainable_model.zig # Trainable model wrapper
│   ├── llm_trainer.zig     # LLM training loop
│   ├── data_loader.zig     # Batch data loading
│   ├── lora.zig            # LoRA adapters
│   ├── mixed_precision.zig # FP16/FP32 mixed precision
│   ├── logging.zig         # TensorBoard/W&B logging
│   └── self_learning.zig   # Self-improvement system
│
├── orchestration/          # Multi-model coordination
│   ├── mod.zig             # Orchestrator
│   ├── router.zig          # Routing strategies
│   ├── ensemble.zig        # Ensemble methods
│   └── fallback.zig        # Fallback management
│
├── rag/                    # Retrieval-Augmented Generation
│   ├── mod.zig             # RAG pipeline
│   ├── document.zig        # Document types
│   ├── chunker.zig         # Document chunking
│   ├── retriever.zig       # Vector retrieval
│   └── context.zig         # Context building
│
├── abbey/                  # Abbey cognitive framework
│   ├── mod.zig             # Abbey entry point
│   ├── engine.zig          # Conversation engine
│   ├── reasoning.zig       # Reasoning chains
│   ├── emotions.zig        # Emotional intelligence
│   ├── context.zig         # Conversation context
│   ├── calibration.zig     # Confidence calibration
│   ├── client.zig          # LLM client abstraction
│   ├── server.zig          # HTTP server
│   ├── discord.zig         # Discord bot
│   ├── custom_framework.zig # Custom AI builder
│   ├── neural/             # Neural components
│   │   ├── tensor.zig      # Tensor operations
│   │   ├── layers.zig      # Linear, Embedding, LayerNorm
│   │   ├── attention.zig   # Multi-head attention
│   │   └── learning.zig    # Online learning
│   ├── memory/             # Three-tier memory
│   │   ├── episodic.zig    # Episode storage
│   │   ├── semantic.zig    # Knowledge graph
│   │   └── working.zig     # Working memory
│   └── advanced/           # Advanced cognition
│       ├── meta_learning.zig    # Task adaptation
│       ├── theory_of_mind.zig   # User modeling
│       ├── compositional.zig    # Problem decomposition
│       └── self_reflection.zig  # Self-evaluation
│
├── personas/               # Multi-persona system
│   ├── mod.zig             # Persona orchestrator
│   ├── stub.zig            # Disabled stub
│   ├── types.zig           # PersonaRequest, PersonaResponse
│   ├── config.zig          # Persona configuration
│   ├── registry.zig        # Persona registry
│   ├── metrics.zig         # Latency percentiles
│   ├── loadbalancer.zig    # Health-weighted routing
│   ├── health.zig          # Health checking
│   ├── alerts.zig          # Alert rules
│   ├── abi/                # Router persona
│   │   ├── sentiment.zig   # Sentiment analysis
│   │   ├── policy.zig      # Content moderation
│   │   └── rules.zig       # Routing rules
│   ├── abbey/              # Empathetic persona
│   │   ├── emotion.zig     # Emotion detection
│   │   ├── empathy.zig     # Empathy injection
│   │   └── reasoning.zig   # Reasoning chains
│   ├── aviva/              # Expert persona
│   │   ├── classifier.zig  # Query classification
│   │   ├── knowledge.zig   # Knowledge retrieval
│   │   ├── code.zig        # Code generation
│   │   └── facts.zig       # Fact checking
│   └── embeddings/         # Persona embeddings
│       ├── persona_index.zig
│       ├── learning.zig
│       └── seed_data.zig
│
├── explore/                # Codebase exploration
│   ├── mod.zig             # Explore entry point
│   ├── stub.zig            # Disabled stub
│   ├── agent.zig           # Explore agent
│   ├── config.zig          # ExploreConfig, ExploreLevel
│   ├── results.zig         # Match, ExploreResult
│   ├── fs.zig              # File system traversal
│   ├── search.zig          # Pattern matching
│   ├── query.zig           # Query understanding
│   ├── ast.zig             # AST parsing
│   ├── parallel.zig        # Parallel exploration
│   ├── callgraph.zig       # Function call graphs
│   └── dependency.zig      # Module dependencies
│
├── multi_agent/            # Multi-agent coordination
│   ├── mod.zig             # Coordinator
│   └── stub.zig            # Disabled stub
│
├── embeddings/             # Vector embeddings
│   ├── mod.zig             # Embedding generation
│   └── stub.zig            # Disabled stub
│
├── streaming/              # Streaming utilities
│   └── mod.zig             # StreamingGenerator
│
├── transformer/            # Transformer architecture
│   └── mod.zig             # TransformerModel
│
├── prompts/                # Prompt management
│   └── mod.zig             # PromptBuilder, Persona
│
├── templates/              # Template system
│   └── mod.zig             # Template rendering
│
├── eval/                   # Model evaluation
│   ├── mod.zig             # Evaluation framework
│   └── stub.zig            # Disabled stub
│
├── federated/              # Federated learning
│   └── mod.zig             # Coordinator, aggregation
│
├── vision/                 # Vision processing
│   ├── mod.zig             # Vision pipeline
│   ├── stub.zig            # Disabled stub
│   ├── image.zig           # Image loading
│   ├── preprocessing.zig   # Normalization, augmentation
│   ├── conv.zig            # Convolution layers
│   ├── pooling.zig         # Max/avg pooling
│   └── batchnorm.zig       # Batch normalization
│
└── database/               # AI-specific database
    ├── mod.zig             # WDBX integration
    └── wdbx.zig            # Token dataset format
```

---

## Core Subsystems

### Core Module (`src/ai/core/`)

Provides fundamental types shared across all AI components:

| Type | Description |
|------|-------------|
| `InstanceId` | Unique AI instance identifier |
| `SessionId` | Conversation session identifier |
| `Confidence` | Confidence level with score and reasoning |
| `EmotionalState` | Detected emotional context |
| `Message` | Conversation message with role and content |
| `Response` | AI response with metadata |

### Configuration

```zig
const config = abi.ai.core.AbbeyConfig{
    .behavior = .{
        .base_temperature = 0.7,
        .research_first = true,
        .enable_emotions = true,
    },
    .memory = .{
        .max_messages = 100,
        .max_tokens = 8000,
    },
    .reasoning = .{
        .max_steps = 10,
        .confidence_threshold = 0.7,
    },
};
```

---

## LLM Engine

The LLM module (`src/ai/llm/`) provides local inference with llama.cpp-compatible GGUF models.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LLM Engine                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   GGUF I/O  │  │  Tokenizer  │  │  Generator  │     │
│  │  (mmap)     │  │  (BPE/SP)   │  │  (Sampler)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         v                v                v             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              LLaMA Model                         │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  Transformer Layers                      │    │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │    │   │
│  │  │  │Attention│ │   FFN   │ │ RMSNorm │    │    │   │
│  │  │  │  (RoPE) │ │  (SiLU) │ │         │    │    │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘    │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              KV Cache                            │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│  │  │Standard │ │ Sliding │ │  Paged  │            │   │
│  │  │         │ │ Window  │ │ (vLLM)  │            │   │
│  │  └─────────┘ └─────────┘ └─────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| GGUF Reader | `io/gguf.zig` | Memory-mapped GGUF model loading |
| Tokenizer | `tokenizer/bpe.zig` | BPE and SentencePiece tokenization |
| Quantization | `tensor/quantized.zig` | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 support |
| Attention | `ops/attention.zig` | Multi-head attention with RoPE |
| Sampler | `generation/sampler.zig` | Temperature, top-k, top-p, mirostat |
| Streaming | `generation/streaming.zig` | Token-by-token streaming with SSE |
| Parallelism | `parallel.zig` | Tensor and pipeline parallelism |

### Usage

```zig
const llm = abi.ai.llm;

// Initialize engine
var engine = llm.Engine.init(allocator, .{
    .max_context_length = 2048,
    .temperature = 0.7,
    .top_p = 0.9,
});
defer engine.deinit();

// Load model
try engine.loadModel("models/llama-7b.gguf");

// Generate text
const output = try engine.generate(allocator, "Hello, ");

// Streaming generation
var response = try engine.createStreamingResponse("Write a story", .{
    .max_tokens = 200,
    .on_token = myCallback,
});
while (try response.next()) |event| {
    if (event.text) |text| try stdout.writeAll(text);
}
```

---

## Agent System

The agent system provides conversational AI with tool integration.

### Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Agent                              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Config    │  │   History   │  │   Backend   │     │
│  │ (name,temp) │  │  (messages) │  │(echo,openai)│     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         v                v                v             │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Tool Registry                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│  │  │   OS    │ │ Discord │ │ Subagent│            │   │
│  │  │  Tools  │ │  Tools  │ │  Tools  │            │   │
│  │  └─────────┘ └─────────┘ └─────────┘            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Agent Types

| Type | Description |
|------|-------------|
| `Agent` | Base conversational agent |
| `GpuAgent` | RL-based GPU scheduling |
| `MultiAgentCoordinator` | Multi-agent orchestration |

### Tool System

```zig
const tools = abi.ai.tools;

var registry = tools.ToolRegistry.init(allocator);
defer registry.deinit();

// Register built-in tools
try tools.registerOsTools(&registry);
try tools.registerDiscordTools(&registry);

// Register custom tool
try registry.register(tools.Tool{
    .name = "calculator",
    .description = "Performs math calculations",
    .handler = calculatorHandler,
});
```

---

## Memory Architecture

The memory module provides multiple strategies for conversation history management.

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                   Memory Manager                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Short-Term Memory                      │   │
│  │  Fixed message buffer (e.g., last 100 messages) │   │
│  └───────────────────────┬─────────────────────────┘   │
│                          │                              │
│                          v                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Sliding Window Memory                    │   │
│  │  Token-based window (e.g., last 4000 tokens)    │   │
│  └───────────────────────┬─────────────────────────┘   │
│                          │                              │
│                          v                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Summarizing Memory                       │   │
│  │  Compresses old messages into summaries          │   │
│  └───────────────────────┬─────────────────────────┘   │
│                          │                              │
│                          v                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Long-Term Memory                        │   │
│  │  Vector-based retrieval for relevant context     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Memory Types

| Type | Use Case |
|------|----------|
| `ShortTermMemory` | Recent messages with fixed capacity |
| `SlidingWindowMemory` | Token-aware context management |
| `SummarizingMemory` | Compress old context into summaries |
| `LongTermMemory` | Vector retrieval for relevant past |

### Usage

```zig
const memory = abi.ai.memory;

var manager = memory.MemoryManager.init(allocator, .{
    .short_term_capacity = 100,
    .max_tokens = 8000,
    .enable_long_term = true,
});
defer manager.deinit();

try manager.addMessage(memory.Message.user("Hello!"));
const context = try manager.getContext(4000); // Get up to 4000 tokens
```

---

## Training Pipeline

The training module provides neural network training with modern optimizations.

### Training Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Data Loader │  │  Model      │  │ Optimizer   │     │
│  │ (batching)  │  │  (forward)  │  │ (SGD/Adam)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         v                v                v             │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Gradient Accumulation                    │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐    │   │
│  │  │ Clipping  │  │ Scaling   │  │ Mixed     │    │   │
│  │  │           │  │           │  │ Precision │    │   │
│  │  └───────────┘  └───────────┘  └───────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │             Checkpointing                        │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐    │   │
│  │  │  Weights  │  │ Optimizer │  │  LoRA     │    │   │
│  │  │  Only     │  │  State    │  │ Adapters  │    │   │
│  │  └───────────┘  └───────────┘  └───────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Training Features

| Feature | Description |
|---------|-------------|
| **Optimizers** | SGD, Adam, AdamW with momentum |
| **LR Schedules** | Constant, cosine, warmup_cosine, step, polynomial, cosine_warm_restarts |
| **Gradient Clipping** | Max norm clipping |
| **Mixed Precision** | FP16/FP32 with loss scaling |
| **LoRA** | Low-rank adaptation for efficient fine-tuning |
| **Checkpointing** | Periodic saves with max checkpoint limit |
| **Early Stopping** | Patience-based convergence detection |
| **Logging** | TensorBoard/W&B metric logging |

### Usage

```zig
const training = abi.ai.training;

const result = try training.trainWithResult(allocator, .{
    .epochs = 10,
    .batch_size = 32,
    .learning_rate = 0.001,
    .optimizer = .adamw,
    .learning_rate_schedule = .warmup_cosine,
    .warmup_steps = 100,
    .gradient_clip_norm = 1.0,
    .checkpoint_interval = 100,
    .checkpoint_path = "./checkpoints",
});
defer result.deinit();

std.debug.print("Final loss: {d:.4}\n", .{result.report.final_loss});
```

---

## Orchestration Layer

The orchestration module coordinates multiple LLM backends.

### Orchestration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Model Registry                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐  │   │
│  │  │ OpenAI  │ │ Ollama  │ │Anthropic│ │ Local │  │   │
│  │  │  GPT-4  │ │ Llama   │ │ Claude  │ │ GGUF  │  │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └───┬───┘  │   │
│  └───────┼───────────┼───────────┼──────────┼──────┘   │
│          │           │           │          │          │
│          v           v           v          v          │
│  ┌─────────────────────────────────────────────────┐   │
│  │                  Router                          │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐      │   │
│  │  │Round Robin│ │Least Load │ │Task-Based │      │   │
│  │  └───────────┘ └───────────┘ └───────────┘      │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐      │   │
│  │  │ Weighted  │ │  Priority │ │Cost/Latency│     │   │
│  │  └───────────┘ └───────────┘ └───────────┘      │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌───────────────────┐     ┌───────────────────┐       │
│  │     Fallback      │     │     Ensemble      │       │
│  │  Priority-based   │     │ Voting/Averaging  │       │
│  │    recovery       │     │    combination    │       │
│  └───────────────────┘     └───────────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Routing Strategies

| Strategy | Description |
|----------|-------------|
| `round_robin` | Cycle through available models |
| `least_loaded` | Route to model with fewest active requests |
| `task_based` | Match model capabilities to task type |
| `weighted` | Distribute based on configured weights |
| `priority` | Use highest priority available model |
| `cost_optimized` | Minimize cost per request |
| `latency_optimized` | Minimize response latency |

### Usage

```zig
const orchestration = abi.ai.orchestration;

var orch = try orchestration.Orchestrator.init(allocator, .{
    .strategy = .task_based,
    .enable_fallback = true,
    .enable_ensemble = false,
});
defer orch.deinit();

try orch.registerModel(.{
    .id = "gpt-4",
    .backend = .openai,
    .capabilities = &.{.reasoning, .coding},
});

const result = try orch.route("Write a sorting function", .coding);
```

---

## RAG Pipeline

The RAG module provides retrieval-augmented generation.

### RAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Document Store                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐            │   │
│  │  │  Text   │ │Markdown │ │  Code   │            │   │
│  │  │ (.txt)  │ │  (.md)  │ │ (.py)   │            │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘            │   │
│  └───────┼───────────┼───────────┼──────────────────┘   │
│          │           │           │                      │
│          v           v           v                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │                  Chunker                         │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐      │   │
│  │  │Fixed Size │ │ Sentence  │ │ Semantic  │      │   │
│  │  │  (512)    │ │ Boundary  │ │ (NLP)     │      │   │
│  │  └───────────┘ └───────────┘ └───────────┘      │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Vector Index                        │   │
│  │  Embeddings + Similarity Search                  │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Context Builder                       │   │
│  │  Query + Retrieved Chunks → Augmented Prompt     │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Usage

```zig
const rag = abi.ai.rag;

var pipeline = rag.RagPipeline.init(allocator, .{
    .max_context_tokens = 2000,
    .deduplicate = true,
});
defer pipeline.deinit();

try pipeline.addText("Machine learning is...", "ML Intro");
try pipeline.addText("Deep learning uses...", "DL Intro");

var response = try pipeline.query("What is machine learning?", 5);
defer response.deinit(allocator);

const augmented_prompt = response.getPrompt();
```

---

## Abbey Cognitive Framework

Abbey is a comprehensive AI framework with emotional intelligence and advanced cognition.

### Abbey Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Abbey Engine                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Three-Tier Memory                      │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐      │   │
│  │  │ Working   │ │ Episodic  │ │ Semantic  │      │   │
│  │  │ Memory    │ │ Memory    │ │ Memory    │      │   │
│  │  │ (active)  │ │ (events)  │ │(knowledge)│      │   │
│  │  └───────────┘ └───────────┘ └───────────┘      │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Advanced Cognition                      │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐      │   │
│  │  │   Meta    │ │Theory of  │ │Compositional│    │   │
│  │  │ Learning  │ │   Mind    │ │ Reasoning │      │   │
│  │  └───────────┘ └───────────┘ └───────────┘      │   │
│  │  ┌───────────────────────────────────────┐      │   │
│  │  │        Self-Reflection                 │      │   │
│  │  │  Bias detection, uncertainty, quality  │      │   │
│  │  └───────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Confidence Calibration                   │   │
│  │  Bayesian updating with evidence tracking        │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Emotional Intelligence                   │   │
│  │  Emotion detection, empathy, adaptive responses  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Advanced Cognition Components

| Component | Description |
|-----------|-------------|
| **Meta-Learning** | Task profiling, few-shot learning, curriculum scheduling |
| **Theory of Mind** | User belief modeling, intention tracking, emotional state |
| **Compositional Reasoning** | Problem decomposition, counterfactual reasoning |
| **Self-Reflection** | Bias detection, uncertainty analysis, quality assessment |

### Usage

```zig
const abbey = abi.ai.abbey;

var engine = try abbey.createEngine(allocator);
defer engine.deinit();

// Process with full cognitive pipeline
const response = try engine.process("Help me understand recursion", .{
    .user_id = "user123",
    .enable_emotions = true,
    .enable_reasoning = true,
});

// Access cognitive state
const cognition = try engine.getCognition();
const tom = cognition.theory_of_mind.getModel("user123");
```

---

## Persona System

The persona system routes requests to specialized AI personas.

### Persona Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Multi-Persona System                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Request Router (Abi)                │   │
│  │  Sentiment → Policy → Classification → Rules     │   │
│  └─────────────────────┬───────────────────────────┘   │
│                        │                                │
│          ┌─────────────┼─────────────┐                 │
│          │             │             │                 │
│          v             v             v                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐            │
│  │   Abbey   │ │   Aviva   │ │    Abi    │            │
│  │ Empathetic│ │  Expert   │ │  Router   │            │
│  │           │ │           │ │           │            │
│  │ Emotion   │ │ Knowledge │ │ Sentiment │            │
│  │ Empathy   │ │ Code      │ │ Policy    │            │
│  │ Reasoning │ │ Facts     │ │ Rules     │            │
│  └───────────┘ └───────────┘ └───────────┘            │
│                        │                                │
│                        v                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Load Balancer                       │   │
│  │  Health-weighted routing with metrics            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Personas

| Persona | Role | Characteristics |
|---------|------|-----------------|
| **Abi** | Router/Moderator | Content moderation, sentiment analysis, policy enforcement |
| **Abbey** | Empathetic Polymath | Supportive, thorough responses with emotional awareness |
| **Aviva** | Direct Expert | Concise, factual, technically rigorous responses |

### Usage

```zig
const personas = abi.ai.personas;

var system = try personas.MultiPersonaSystem.init(allocator, .{
    .default_persona = .abbey,
    .enable_dynamic_routing = true,
});
defer system.deinit();

const response = try system.process(.{
    .content = "I'm frustrated with this bug",
    .user_id = "user-123",
});

// Routed to Abbey due to frustration detection
std.debug.print("Handled by: {s}\n", .{@tagName(response.persona)});
```

---

## Explore Module

The explore module provides AI-powered codebase exploration.

### Explore Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Explore Agent                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Query Understanding                 │   │
│  │  Intent detection, entity extraction             │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Parallel File Explorer                 │   │
│  │  Multi-threaded directory traversal              │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│  ┌───────────────────┐     ┌───────────────────┐       │
│  │   Pattern Search  │     │   AST Parser      │       │
│  │  Regex, glob, etc │     │ Function/struct   │       │
│  └───────────────────┘     └───────────────────┘       │
│                         │                               │
│                         v                               │
│  ┌───────────────────┐     ┌───────────────────┐       │
│  │   Call Graph      │     │ Dependency Graph  │       │
│  │  Function calls   │     │ Module imports    │       │
│  └───────────────────┘     └───────────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Explore Levels

| Level | Description |
|-------|-------------|
| `quick` | Basic pattern search, surface-level |
| `medium` | AST parsing, function analysis |
| `thorough` | Call graphs, dependency analysis |
| `deep` | Full semantic analysis, cross-references |

### Usage

```zig
const explore = abi.ai.explore;

var agent = try explore.createThoroughAgent(allocator);
defer agent.deinit();

const result = try agent.explore(".", "database connection handling");
for (result.matches) |match| {
    std.debug.print("{s}:{d}: {s}\n", .{match.file, match.line, match.snippet});
}
```

---

## Feature Flags & Configuration

### Build-Time Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-Denable-ai` | true | Full AI module |
| `-Denable-llm` | true | LLM inference (requires AI) |
| `-Denable-vision` | true | Vision processing (requires AI) |
| `-Denable-explore` | true | Codebase exploration (requires AI) |

### Stub Pattern

Each feature module has a corresponding `stub.zig` that provides the same API surface when disabled:

```zig
// When AI is enabled:
pub const llm = @import("llm/mod.zig");

// When AI is disabled:
pub const llm = @import("llm/stub.zig");
// stub.zig returns error.LlmDisabled for all operations
```

### Runtime Configuration

```zig
const config = abi.Config.init()
    .withAI(.{
        .llm = .{ .model_path = "./model.gguf" },
        .embeddings = .{ .dimension = 768 },
        .agents = .{ .max_agents = 10 },
        .training = .{ .checkpoint_path = "./checkpoints" },
        .personas = .{ .default_persona = .abbey },
    });

var framework = try abi.Framework.init(allocator, config);
defer framework.deinit();

const ai = try framework.getAi();
```

---

## Integration Patterns

### Framework Integration

```zig
const abi = @import("abi");

// Initialize framework with AI configuration
var fw = try abi.Framework.init(allocator, .{
    .ai = .{
        .llm = .{ .context_size = 4096 },
        .agents = .{ .max_agents = 5 },
    },
});
defer fw.deinit();

// Access AI context
const ai_ctx = try fw.getAi();

// Use LLM
const llm = try ai_ctx.getLlm();
const output = try llm.generate("Hello!");

// Use agents
const agents = try ai_ctx.getAgents();
var my_agent = try agents.createAgent("assistant");
```

### Standalone Usage

```zig
const ai = abi.ai;

// Create agent directly
var agent = try ai.Agent.init(allocator, .{
    .name = "assistant",
    .backend = .echo,
});
defer agent.deinit();

const response = try agent.chat("Hello!", allocator);
defer allocator.free(response);
```

### Connector Integration

```zig
const connectors = abi.connectors;

// OpenAI
var openai = try connectors.openai.Client.init(allocator, .{});
defer openai.deinit();
const response = try openai.chat("Hello", .{});

// Ollama (local)
var ollama = try connectors.ollama.Client.init(allocator, .{
    .host = "http://127.0.0.1:11434",
    .model = "llama3.2",
});
const response = try ollama.generate("Hello");
```

---

## See Also

- [ai.md](../ai.md) - User-facing AI documentation
- [agents.md](../agents.md) - Agent personas and interaction guide
- [training.md](../training.md) - Training pipeline details
- [gpu.md](../gpu.md) - GPU acceleration
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
