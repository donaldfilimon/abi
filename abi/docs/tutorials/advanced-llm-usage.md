# Advanced LLM Usage

This tutorial covers advanced usage patterns: fine-tuning, LoRA, and prompt engineering best practices for ABI.

## Topics
- Fine-tuning overview
- LoRA quickstart
- Prompt engineering patterns

## Example: Loading a model (pseudo-code)

```
const model = try ABI.loadModel(allocator, .{ .path = "models/llama-7b.gguf" });
const response = try model.generate("Hello world");
```

## Acceptance criteria for this tutorial
- Includes at least one runnable example (or pseudo-code clearly labeled)
- Lists common pitfalls and performance tips
