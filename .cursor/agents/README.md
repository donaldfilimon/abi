# Cursor agents

Expert agents for Cursor are defined here. Use the agent whose description matches the task (e.g. GPU/NPU on Apple Silicon).

## Index

| Agent | File | When to use |
|-------|------|-------------|
| **Metal/CoreML GPU/NPU** | [metal-coreml-gpu-npu.md](metal-coreml-gpu-npu.md) | Apple Silicon GPU/ANE, Metal backend, CoreML linking, MPS, macOS M-series. |

## Usage

- In Cursor, select the agent when starting a task that matches its expertise.
- Full index of skills, plans, and agents (including this directory): [CLAUDE.md — Skills, Plans, and Agents](../../CLAUDE.md#skills-plans-and-agents-full-index).

## Adding an agent

1. Add a `.md` file with front matter: `name`, `model` (optional), `description`.
2. Describe when to use the agent and what to deliver.
3. Update this README and the Cursor agents table in CLAUDE.md (Skills, Plans, and Agents section).
