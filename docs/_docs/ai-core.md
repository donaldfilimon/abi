---
title: "AI Core"
description: "Agents, tools, prompts, personas, memory, and model discovery in the ABI AI Core module."
section: "AI"
order: 2
---

# AI Core

The AI Core module provides the foundational building blocks for AI-powered
applications: agents, tool registries, prompt builders, personas, memory
management, multi-agent coordination, and model discovery.

- **Build flag:** `-Denable-ai=true` (default: enabled)
- **Namespace:** `abi.ai_core`
- **Source:** `src/features/ai_core/`

## Overview

AI Core is the "agent runtime" layer of the ABI framework. It does not perform
inference itself -- for LLM execution, see the [Inference](ai-inference.html)
module. Instead, AI Core provides the scaffolding that agents use to coordinate
tools, build prompts, manage conversation memory, and discover available models.

Key capabilities:

- **Agents** -- Task-oriented AI agents with configurable behavior
- **Multi-Agent Coordination** -- Orchestrate multiple agents working together
- **Tool Registry** -- Register and execute tools (OS tools, Discord tools, custom)
- **Prompt Builder** -- Construct prompts with persona-aware system instructions
- **Personas** -- 13 built-in personas (Abbey, Abi, Aviva, Ralph, Ava, and more)
- **Memory** -- Conversation and context memory for agents
- **Model Discovery** -- Auto-detect available models and system capabilities
- **GPU Agent** -- GPU-aware agent for hardware-accelerated workloads

## Quick Start

```zig
const abi = @import("abi");

// Initialize AI Core context
var ctx = try abi.ai_core.Context.init(allocator, .{
    .auto_discover = true,  // scan for available models
    .agents = .{ .max_agents = 8 },
});
defer ctx.deinit();

// Create an agent
var agent = try abi.ai_core.createAgent(allocator, "assistant");
defer agent.deinit();

// Build a prompt with a persona
var builder = abi.ai_core.PromptBuilder.init(allocator, .abbey);
defer builder.deinit();
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Module context with agent and discovery management |
| `Agent` | Single AI agent for task execution |
| `MultiAgentCoordinator` | Orchestrates multiple agents |
| `ModelRegistry` | Registry of available models |
| `ModelInfo` | Metadata about a registered model |
| `Tool` | A callable tool that agents can invoke |
| `ToolResult` | Result returned by a tool execution |
| `ToolRegistry` | Collection of registered tools |
| `TaskTool` | Tool for spawning sub-tasks |
| `Subagent` | Lightweight agent spawned by a parent agent |
| `PromptBuilder` | Fluent builder for constructing prompts |
| `Persona` | Persona definition with system prompt and settings |
| `PersonaType` | Enum of available persona identifiers |
| `PromptFormat` | Output format for built prompts |
| `GpuAgent` | Agent that routes workloads to GPU when available |
| `GpuAwareRequest` | Request type for GPU-aware processing |
| `GpuAwareResponse` | Response from GPU-aware processing |
| `WorkloadType` | Classification of GPU workload types |
| `ModelDiscovery` | Auto-discovers models on the local system |
| `DiscoveredModel` | A model found during discovery |
| `DiscoveryConfig` | Configuration for model discovery |
| `SystemCapabilities` | Detected hardware and software capabilities |
| `AdaptiveConfig` | Configuration that adapts to system capabilities |
| `Confidence` | Confidence score with Bayesian updating |
| `ConfidenceLevel` | Discrete confidence level (high, medium, low) |
| `EmotionalState` | Emotional state tracking for personas |
| `EmotionType` | Enum of emotion categories |

### Key Functions

| Function | Description |
|----------|-------------|
| `isEnabled() bool` | Returns `true` if AI Core is compiled in |
| `createRegistry(allocator) ModelRegistry` | Create an empty model registry |
| `createAgent(allocator, name) !Agent` | Create a named agent (returns `error.AiDisabled` if compiled out) |
| `detectCapabilities() SystemCapabilities` | Detect local hardware capabilities |
| `runWarmup(allocator, config) !WarmupResult` | Run model warmup / benchmark |

### Context

The `Context` struct manages agent lifecycle and model discovery:

```zig
var ctx = try abi.ai_core.Context.init(allocator, ai_config);
defer ctx.deinit();

// Access agents sub-context
const agents_ctx = try ctx.getAgents();
```

The context automatically runs model discovery at init time when
`config.auto_discover` is `true`, scanning for locally available models.

## Personas

The persona system provides 13 built-in personalities, each with a unique
system prompt, suggested temperature, and behavioral profile:

| PersonaType | Role |
|-------------|------|
| `assistant` | General-purpose helpful assistant |
| `coder` | Code-focused programming assistant |
| `writer` | Creative writing assistant |
| `analyst` | Data analysis and research assistant |
| `companion` | Friendly conversational companion |
| `docs` | Technical documentation helper |
| `reviewer` | Code review specialist |
| `minimal` | Minimal / direct response mode |
| `abbey` | Opinionated, emotionally intelligent AI |
| `ralph` | Iterative, tireless worker for complex tasks |
| `aviva` | Direct expert for concise, factual output |
| `abi` | Adaptive moderator and router |
| `ava` | Locally-trained assistant based on gpt-oss |

```zig
// Get a persona definition
const persona = abi.ai_core.prompts.getPersona(.abbey);
std.debug.print("Name: {s}\n", .{persona.name});
std.debug.print("Temp: {d}\n", .{persona.suggested_temperature});

// Build a prompt with a persona
var builder = abi.ai_core.prompts.createBuilderWithPersona(allocator, .coder);
defer builder.deinit();

// List all available personas
const all = abi.ai_core.prompts.listPersonas();
```

## Tool Registration

Agents interact with the outside world through tools. The framework ships with
built-in tool sets and supports custom tools:

```zig
// Create a tool registry
var registry = abi.ai_core.ToolRegistry.init(allocator);
defer registry.deinit();

// Register built-in OS tools (file read, directory list, etc.)
abi.ai_core.registerOsTools(&registry);

// Register Discord tools (send message, read channel, etc.)
abi.ai_core.registerDiscordTools(&registry);
```

### Built-in Tool Sets

| Tool Set | Registration Function | Capabilities |
|----------|-----------------------|-------------|
| OS Tools | `registerOsTools()` | File I/O, directory listing, process execution |
| Discord Tools | `registerDiscordTools()` | Send messages, read channels, manage server |

## Memory Management

The memory module provides conversation history and context tracking for agents:

```zig
const mem = abi.ai_core.memory;

// Memory is managed through the agent's internal state
var agent = try abi.ai_core.createAgent(allocator, "my-agent");
defer agent.deinit();
```

## Model Discovery

The discovery system automatically scans for available models and detects
hardware capabilities:

```zig
// Detect what the system can do
const caps = abi.ai_core.detectCapabilities();

// Create a model discovery instance
var disc = abi.ai_core.ModelDiscovery.init(allocator, .{});
defer disc.deinit();

// Scan all known locations for models
try disc.scanAll();
```

## Configuration

AI Core is configured through the `AiConfig` struct passed to `Context.init()`:

```zig
const config: abi.config.AiConfig = .{
    .auto_discover = true,
    .agents = .{
        .max_agents = 8,
    },
};

var ctx = try abi.ai_core.Context.init(allocator, config);
```

## CLI Commands

```bash
# Run an interactive agent session
zig build run -- agent

# Run agent with a specific persona
zig build run -- agent --persona abbey
```

## Disabling at Build Time

```bash
# Compile without AI Core (and all AI features)
zig build -Denable-ai=false
```

When disabled, all public functions return `error.AiDisabled` and `isEnabled()`
returns `false`. The stub module preserves the same type signatures so
downstream code compiles without `#ifdef`-style guards.

## Related

- [AI Overview](ai-overview.html) -- Architecture of all five AI modules
- [Inference](ai-inference.html) -- LLM execution, embeddings, streaming
- [Training](ai-training.html) -- Model training pipelines
- [Reasoning](ai-reasoning.html) -- Abbey engine, RAG, orchestration
