# Connectors

Connectors provide pluggable backends for AI inference calls.

## Core Types
- `CallRequest`: model, prompt, max_tokens, temperature
- `CallResult`: ok, content, token counts, status info

## Implementations
- `connectors.openai`
- `connectors.hf_inference`
- `connectors.ollama`
- `connectors.local_scheduler`
- `connectors.mock`
- `connectors.plugin`

## Usage
```zig
const connectors = abi.connectors;
try connectors.init(allocator);
```
