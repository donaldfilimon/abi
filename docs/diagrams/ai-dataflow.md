---
title: "ai-dataflow"
tags: []
---
# AI Module Data Flow
> **Codebase Status:** Synced with repository as of 2026-01-30.

```mermaid
flowchart LR
    subgraph "Input"
        TEXT[Text Input]
        IMG[Image Input]
        AUDIO[Audio Input]
    end

    subgraph "Connectors"
        OPENAI[OpenAI Connector]
        OLLAMA[Ollama Connector]
        HF[HuggingFace Connector]
        LOCAL[Local LLM]
    end

    subgraph "Processing"
        PROMPT[Prompt Engine]
        ABBEY[Abbey Engine]
        VISION[Vision Pipeline]
        AGENT[Agent Orchestrator]
    end

    subgraph "Output"
        RESPONSE[Text Response]
        ACTION[Agent Action]
        EMBED[Embeddings]
    end

    TEXT --> PROMPT
    IMG --> VISION
    AUDIO --> PROMPT

    PROMPT --> OPENAI
    PROMPT --> OLLAMA
    PROMPT --> HF
    PROMPT --> LOCAL

    OPENAI --> ABBEY
    OLLAMA --> ABBEY
    HF --> ABBEY
    LOCAL --> ABBEY

    VISION --> AGENT
    ABBEY --> AGENT

    AGENT --> RESPONSE
    AGENT --> ACTION
    ABBEY --> EMBED
```

## Component Responsibilities

| Component | Description |
|-----------|-------------|
| Connectors | External API integrations (OpenAI, Ollama, HuggingFace) |
| Prompt Engine | Template management and prompt construction |
| Abbey Engine | Core LLM inference and response handling |
| Vision Pipeline | Image processing and feature extraction |
| Agent Orchestrator | Multi-agent coordination and task decomposition |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ABI_OPENAI_API_KEY` | OpenAI authentication |
| `ABI_OLLAMA_HOST` | Ollama server endpoint |
| `ABI_HF_API_TOKEN` | HuggingFace API access |

