# ai-embeddings API Reference

> Vector embeddings generation

**Source:** [`src/ai/embeddings/mod.zig`](../../src/ai/embeddings/mod.zig)

---

Embeddings Sub-module

Vector embeddings generation for text and other data types.
Provides models for converting text into dense vector representations.

---

## API

### `pub const EmbeddingConfig`

<sup>**type**</sup>

Configuration for embedding models.

### `pub const EmbeddingModel`

<sup>**type**</sup>

Embedding model for text vectorization.

### `pub fn embed(self: *EmbeddingModel, text: []const u8) ![]f32`

<sup>**fn**</sup>

Generate embedding for a single text.

### `pub fn embedBatch(self: *EmbeddingModel, texts: []const []const u8) ![][]f32`

<sup>**fn**</sup>

Generate embeddings for multiple texts.

### `pub fn cosineSimilarity(_: *EmbeddingModel, a: []const f32, b: []const f32) f32`

<sup>**fn**</sup>

Compute cosine similarity between two embeddings.

### `pub const BatchProcessor`

<sup>**type**</sup>

Batch processor for efficient embedding generation.

### `pub const EmbeddingCache`

<sup>**type**</sup>

Embedding cache for deduplication.

### `pub const Context`

<sup>**type**</sup>

Embeddings context for framework integration.

### `pub fn embed(self: *Context, text: []const u8) ![]f32`

<sup>**fn**</sup>

Generate embedding for text.

### `pub fn embedBatch(self: *Context, texts: []const []const u8) ![][]f32`

<sup>**fn**</sup>

Generate embeddings for multiple texts.

### `pub fn cosineSimilarity(_: *Context, a: []const f32, b: []const f32) f32`

<sup>**fn**</sup>

Compute cosine similarity between two embeddings.

---

*Generated automatically by `zig build gendocs`*
