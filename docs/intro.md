# Introduction

Welcome to **ABI**, a modern Zig framework designed for modular AI services, vector search, and systems tooling.

## Philosophy

ABI is built on three core pillars:

1.  **Modularity**: Use only what you need. Core features are isolated, and advanced subsystems (AI, GPU, Database) are opt-in.
2.  **Performance**: Written in Zig 0.16.x, leveraging a work-stealing compute runtime and zero-copy data structures.
3.  **Modernity**: Native support for vector embeddings, AI agents, and GPU acceleration.

## Architecture

The framework is divided into several layers:

### 1. Core Runtime (`src/framework/`)

The backbone of the application. Handles:

- Lifecycle management (`init`, `shutdown`)
- Configuration
- Feature flag orchestration

### 2. Compute Engine (`src/compute/`)

A high-performance parallel execution environment.

- **Work-Stealing Scheduler**: Efficiently distributes tasks across worker threads.
- **GPU Integration**: Seamlessly offloads compatible workloads to GPU backends.
- **Memory Management**: Arena-based allocation strategies for stable runtime performance.

### 3. Feature Stacks (`src/features/`)

Domain-specific modules that plug into the core:

- **AI**: LLM connectors, Agent runtimes.
- **Database**: WDBX vector database, backup/restore.
- **GPU**: Backend implementations (CUDA, Vulkan, Metal).
- **Network**: Distributed node discovery and task serialization.
- **Web**: Async HTTP clients and servers.

## Getting Started

To start using ABI, check out the [Framework Guide](framework.md) to initialize your application.
