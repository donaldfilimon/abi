---
title: CLI
description: 28 commands and 8 aliases for AI, GPU, database, and system management
section: Core
order: 4
---

# CLI

ABI includes a comprehensive CLI with 28 commands and 8 aliases for managing AI models,
GPU devices, databases, agents, and system operations.

## Running Commands

```bash
# General syntax
zig build run -- <command> [subcommand] [options]

# Show all commands
zig build run -- --help

# Help for a specific command
zig build run -- help llm
zig build run -- llm --help
```

## Command Reference

### AI and Model Management (7 commands)

| Command | Description | Example Subcommands |
|---------|-------------|---------------------|
| `agent` | Run AI agent (interactive or one-shot) | -- |
| `llm` | LLM inference operations | `info`, `generate`, `chat`, `bench`, `download` |
| `model` | Model management | `list`, `download`, `remove`, `search` |
| `embed` | Generate vector embeddings | OpenAI, Mistral, Cohere, Ollama backends |
| `train` | Training pipeline | `run`, `resume`, `info` |
| `multi-agent` | Multi-agent workflows | -- |
| `explore` | Search and explore codebase | -- |

### GPU and Compute (4 commands)

| Command | Description | Example Subcommands |
|---------|-------------|---------------------|
| `gpu` | GPU device management | `backends`, `devices`, `summary`, `default` |
| `gpu-dashboard` | Interactive GPU + agent monitoring dashboard | -- |
| `simd` | Run SIMD performance demo | -- |
| `bench` | Performance benchmarks | `all`, `simd`, `memory`, `ai`, `quick` |

### Data and Storage (2 commands)

| Command | Description | Example Subcommands |
|---------|-------------|---------------------|
| `db` | Database operations | `add`, `query`, `stats`, `optimize`, `backup` |
| `convert` | Dataset conversion tools | `tokenbin`, `text`, `jsonl`, `wdbx` |

### Network and Services (4 commands)

| Command | Description | Example Subcommands |
|---------|-------------|---------------------|
| `network` | Network registry management | `list`, `register`, `status` |
| `discord` | Discord bot operations | `status`, `guilds`, `send`, `commands` |
| `mcp` | MCP server (Model Context Protocol) | `serve`, `tools` |
| `acp` | Agent Communication Protocol | `card` |

### System and Configuration (11 commands)

| Command | Description | Example Subcommands |
|---------|-------------|---------------------|
| `system-info` | System and feature status | -- |
| `version` | Print framework version | -- |
| `status` | Framework health and component status | -- |
| `config` | Configuration management | `init`, `show`, `validate` |
| `plugins` | Plugin management | `list`, `enable`, `disable`, `info` |
| `profile` | User profile and settings | -- |
| `toolchain` | Zig/ZLS toolchain management | `install`, `update`, `status` |
| `task` | Task management | `add`, `list`, `done`, `stats` |
| `tui` | Launch interactive TUI menu | -- |
| `completions` | Generate shell completions | `bash`, `zsh`, `fish`, `powershell` |
| `help` | Show help for any command | -- |
| `os-agent` | OS-level agent operations | -- |

**Total: 28 commands** (7 AI + 4 GPU + 2 Data + 4 Network + 11 System).

### Aliases (8)

| Alias | Expands To |
|-------|-----------|
| `serve` | `llm serve` |

(Additional aliases are registered in `tools/cli/main.zig` for common workflows.)

## MCP Server

The MCP (Model Context Protocol) server exposes WDBX database operations over
JSON-RPC 2.0 on stdio:

```bash
# Start the MCP server
zig build run -- mcp serve

# List available MCP tools
zig build run -- mcp tools
```

Available MCP tools:

| Tool | Description |
|------|-------------|
| `wdbx_query` | Query the vector database |
| `wdbx_insert` | Insert vectors and documents |
| `wdbx_stats` | Database statistics |
| `wdbx_list` | List collections |
| `wdbx_delete` | Delete entries |

The server reads JSON-RPC requests from stdin and writes responses to stdout, making it
compatible with editors and tools that speak the Model Context Protocol.

## ACP (Agent Communication Protocol)

```bash
# Print the agent card (JSON describing capabilities)
zig build run -- acp card
```

The ACP service provides agent-to-agent communication with task lifecycle management
(create, assign, complete, cancel).

## Common Workflows

### Model Management

```bash
# List available models
zig build run -- model list

# Download a model
zig build run -- model download llama-7b

# Run inference
zig build run -- llm generate --prompt "Explain Zig comptime"

# Interactive chat
zig build run -- llm chat
```

### Database Operations

```bash
# Add a document with vector embedding
zig build run -- db add --text "Hello world" --collection default

# Query by similarity
zig build run -- db query --text "greeting" --top-k 5

# Show database statistics
zig build run -- db stats

# Optimize storage
zig build run -- db optimize

# Backup
zig build run -- db backup --output ./backup/
```

### GPU Status

```bash
# List available backends
zig build run -- gpu backends

# Show detected devices
zig build run -- gpu devices

# Summary of GPU capabilities
zig build run -- gpu summary

# Interactive GPU dashboard
zig build run -- gpu-dashboard
```

### Benchmarks

```bash
# Run all benchmarks
zig build run -- bench all

# Quick benchmark suite
zig build run -- bench quick

# SIMD-specific benchmarks
zig build run -- bench simd

# Or use the build system directly
zig build benchmarks
zig build bench-all
```

### Training

```bash
# Start a training run
zig build run -- train run --config train.json

# Resume interrupted training
zig build run -- train resume --checkpoint ./checkpoints/latest

# Show training info
zig build run -- train info
```

### Configuration

```bash
# Initialize a config file
zig build run -- config init

# Show current configuration
zig build run -- config show

# Validate configuration
zig build run -- config validate
```

## Shell Completions

Generate completions for your shell:

```bash
# Bash
zig build run -- completions bash > ~/.local/share/bash-completion/completions/abi

# Zsh
zig build run -- completions zsh > ~/.zsh/completions/_abi

# Fish
zig build run -- completions fish > ~/.config/fish/completions/abi.fish

# PowerShell
zig build run -- completions powershell > abi.ps1
```

## Build Targets for Examples

Many of the 36 examples are registered as build targets for direct execution:

```bash
zig build run-hello           # Hello world example
zig build run-gpu             # GPU compute example
zig build examples            # Build all 36 examples
```

## CLI Smoke Tests

The build system includes smoke tests that verify all 28 commands accept `--help`:

```bash
zig build cli-tests
```

This runs both top-level commands (e.g., `help`) and nested subcommands
(e.g., `help llm`, `bench micro hash`) to catch registration and parsing errors.

## Further Reading

- [Configuration](configuration.html) -- build flags and environment variables used by CLI commands
- [Framework Lifecycle](framework.html) -- how `config init` and `config validate` interact with the framework
- [Getting Started](getting-started.html) -- first CLI commands to try
