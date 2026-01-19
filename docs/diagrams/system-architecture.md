# ABI System Architecture
> **Codebase Status:** Synced with repository as of 2026-01-18.

```mermaid
flowchart TB
    subgraph "Public API"
        ABI[abi.zig<br/>init, shutdown, version]
    end

    subgraph "Framework"
        FRAMEWORK[framework/mod.zig<br/>Lifecycle Management]
    end

    subgraph "Core"
        CORE[core/mod.zig]
        IO[I/O Utilities]
        DIAG[Diagnostics]
        COLL[Collections]
    end

    subgraph "Compute"
        RUNTIME[Runtime Engine]
        CONC[Concurrency<br/>WorkStealingQueue, LockFree*]
        GPU[GPU Module]
        MEM[Memory Management]
        PROF[Profiling]
    end

    subgraph "Features"
        AI[AI Module<br/>LLM, Vision, Agent, Training]
        DB[Database<br/>WDBX Vector DB]
        NET[Network<br/>Distributed Compute]
        WEB[Web Utilities]
        MON[Monitoring]
        CONN[Connectors<br/>OpenAI, Ollama, HuggingFace]
    end

    subgraph "Shared"
        LOG[Logging]
        OBS[Observability]
        SEC[Security]
        UTIL[Utils<br/>Platform, SIMD, Time]
    end

    ABI --> FRAMEWORK
    FRAMEWORK --> CORE
    FRAMEWORK --> COMPUTE
    FRAMEWORK --> Features

    CORE --> IO
    CORE --> DIAG
    CORE --> COLL

    COMPUTE --> RUNTIME
    COMPUTE --> CONC
    COMPUTE --> GPU
    COMPUTE --> MEM
    COMPUTE --> PROF

    AI --> CONN

    CORE --> SHARED
    COMPUTE --> SHARED
    Features --> SHARED
```

## Layer Descriptions

| Layer | Purpose |
|-------|---------|
| Public API | Single entry point (`abi.zig`) for all functionality |
| Framework | Lifecycle management, feature orchestration |
| Core | Fundamental utilities: I/O, diagnostics, collections |
| Compute | High-performance runtime, concurrency, GPU, profiling |
| Features | Domain-specific: AI, Database, Network, Web, Monitoring |
| Shared | Cross-cutting: logging, security, platform abstractions |

## Feature Gating

All features use compile-time gating via build options:

```zig
const impl = if (build_options.enable_feature)
    @import("real.zig")
else
    @import("stub.zig");
```

Disabled features return `error.<Feature>Disabled`.
