# ai-llm API Reference

> Local LLM inference

**Source:** [`src/ai/llm/mod.zig`](../../src/ai/llm/mod.zig)

---

LLM Sub-module

Local LLM inference supporting GGUF models and transformer architectures.

This module provides a pure Zig implementation for loading and running
large language models locally without external dependencies. Supports:
- GGUF model format (llama.cpp compatible)
- BPE tokenization
- Quantized inference (Q4_0, Q8_0)
- KV caching for efficient autoregressive generation
- GPU acceleration with CPU fallback

Usage:
```zig
const llm = @import("llm");

var model = try llm.Model.load(allocator, "model.gguf");
defer model.deinit();

var generator = model.generator(.{});
const output = try generator.generate(allocator, "Hello, world!");
```

---

## API

### `pub const LlmError`

<sup>**const**</sup>

LLM-specific errors

### `pub const InferenceConfig`

<sup>**type**</sup>

Configuration for LLM inference

### `pub const InferenceStats`

<sup>**type**</sup>

Statistics from inference

### `pub const Engine`

<sup>**type**</sup>

High-level interface for loading and running models

### `pub fn loadModel(self: *Engine, path: []const u8) !void`

<sup>**fn**</sup>

Load a model from a GGUF file

### `pub fn generate(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) ![]u8`

<sup>**fn**</sup>

Generate text from a prompt

### `pub fn generateStreaming(`

<sup>**fn**</sup>

Generate with streaming callback (per-token)

### `pub fn tokenize(self: *Engine, allocator: std.mem.Allocator, text: []const u8) ![]u32`

<sup>**fn**</sup>

Tokenize text

### `pub fn detokenize(self: *Engine, allocator: std.mem.Allocator, tokens: []const u32) ![]u8`

<sup>**fn**</sup>

Detokenize tokens

### `pub fn getStats(self: *Engine) InferenceStats`

<sup>**fn**</sup>

Get current statistics

### `pub const Context`

<sup>**type**</sup>

LLM context for framework integration.

### `pub fn getEngine(self: *Context) !*Engine`

<sup>**fn**</sup>

Get or initialize the LLM engine.

### `pub fn generate(self: *Context, prompt: []const u8) ![]u8`

<sup>**fn**</sup>

Generate text from prompt.

### `pub fn tokenize(self: *Context, text: []const u8) ![]u32`

<sup>**fn**</sup>

Tokenize text.

### `pub fn detokenize(self: *Context, tokens: []const u32) ![]u8`

<sup>**fn**</sup>

Detokenize tokens.

### `pub fn isEnabled() bool`

<sup>**fn**</sup>

Check if LLM features are enabled

### `pub fn infer(allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8) ![]u8`

<sup>**fn**</sup>

Quick inference helper

---

*Generated automatically by `zig build gendocs`*
