//! ABI Framework - C-Compatible LLM API
//!
//! Provides C-compatible bindings for LLM inference, following llama.cpp naming
//! conventions for easy integration with existing tooling and libraries.
//!
//! Usage from C:
//!   llama_model* model = llama_model_load("model.gguf", NULL);
//!   llama_context* ctx = llama_context_create(model, NULL);
//!   int32_t tokens[512];
//!   int32_t n_tokens = llama_tokenize(ctx, "Hello world", tokens, 512, true);
//!   llama_generate(ctx, tokens, n_tokens, 100, NULL);
//!   llama_context_free(ctx);
//!   llama_model_free(model);

const std = @import("std");
const build_options = @import("build_options");
const abi = @import("abi");

// Conditional imports based on feature flags
const llm = if (build_options.enable_llm) abi.ai.llm else struct {};
const gguf = if (build_options.enable_llm) abi.ai.llm.io else struct {};
const tokenizer_mod = if (build_options.enable_llm) abi.ai.llm.tokenizer else struct {};
const generation = if (build_options.enable_llm) abi.ai.llm.generation else struct {};
const ops = if (build_options.enable_llm) abi.ai.llm.ops else struct {};

// Global allocator for C API
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

//==============================================================================
// Opaque Handle Types (C-compatible)
//==============================================================================

/// Opaque model handle.
pub const llama_model = opaque {};

/// Opaque context handle.
pub const llama_context = opaque {};

/// Opaque tokenizer handle.
pub const llama_tokenizer = opaque {};

/// Opaque sampler handle.
pub const llama_sampler = opaque {};

//==============================================================================
// Configuration Structures
//==============================================================================

/// Model loading parameters.
pub const llama_model_params = extern struct {
    /// Number of layers to offload to GPU (-1 = all, 0 = none).
    n_gpu_layers: i32 = -1,
    /// Use memory mapping for model weights.
    use_mmap: bool = true,
    /// Use memory locking (mlock) for model weights.
    use_mlock: bool = false,
    /// Vocabulary-only mode (no weights).
    vocab_only: bool = false,
    /// Reserved for future use.
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Context creation parameters.
pub const llama_context_params = extern struct {
    /// Context size (max sequence length).
    n_ctx: u32 = 2048,
    /// Batch size for prompt processing.
    n_batch: u32 = 512,
    /// Number of threads for generation.
    n_threads: u32 = 4,
    /// Number of threads for batch processing.
    n_threads_batch: u32 = 4,
    /// RoPE base frequency.
    rope_freq_base: f32 = 10000.0,
    /// RoPE frequency scale.
    rope_freq_scale: f32 = 1.0,
    /// Use flash attention.
    flash_attn: bool = false,
    /// Reserved for future use.
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Sampling parameters.
pub const llama_sampling_params = extern struct {
    /// Temperature for sampling.
    temperature: f32 = 0.8,
    /// Top-K sampling (0 = disabled).
    top_k: i32 = 40,
    /// Top-P (nucleus) sampling.
    top_p: f32 = 0.95,
    /// Min-P sampling.
    min_p: f32 = 0.05,
    /// Typical-P sampling.
    typical_p: f32 = 1.0,
    /// Repetition penalty.
    repeat_penalty: f32 = 1.1,
    /// Repetition penalty window.
    repeat_last_n: i32 = 64,
    /// Frequency penalty.
    frequency_penalty: f32 = 0.0,
    /// Presence penalty.
    presence_penalty: f32 = 0.0,
    /// Mirostat mode (0 = disabled, 1 = v1, 2 = v2).
    mirostat: i32 = 0,
    /// Mirostat target entropy.
    mirostat_tau: f32 = 5.0,
    /// Mirostat learning rate.
    mirostat_eta: f32 = 0.1,
    /// Random seed (-1 = random).
    seed: i64 = -1,
    /// Reserved for future use.
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

/// Generation callback for streaming tokens.
pub const llama_token_callback = ?*const fn (token_id: i32, text: [*:0]const u8, user_data: ?*anyopaque) callconv(.c) bool;

/// Generation result structure.
pub const llama_generation_result = extern struct {
    /// Number of tokens generated.
    n_tokens: i32 = 0,
    /// Total generation time in milliseconds.
    time_ms: f64 = 0,
    /// Tokens per second.
    tokens_per_second: f64 = 0,
    /// Prompt tokens processed.
    n_prompt_tokens: i32 = 0,
    /// Time to first token in milliseconds.
    time_to_first_token_ms: f64 = 0,
    /// Reserved for future use.
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

//==============================================================================
// Internal State Structures
//==============================================================================

const ModelState = struct {
    gguf_model: if (build_options.enable_llm) ?*gguf.GgufFile else void,
    vocab_size: u32,
    n_layers: u32,
    n_heads: u32,
    n_embd: u32,
    path: []const u8,
};

const ContextState = struct {
    model: *ModelState,
    tokenizer: if (build_options.enable_llm) ?tokenizer_mod.Tokenizer else void,
    sampler: if (build_options.enable_llm) generation.Sampler else void,
    gpu_ctx: if (build_options.enable_llm) ?ops.GpuOpsContext else void,
    params: llama_context_params,
    // KV cache
    kv_cache_k: ?[]f32,
    kv_cache_v: ?[]f32,
    kv_pos: u32,
};

//==============================================================================
// Model Functions
//==============================================================================

/// Load a model from a GGUF file.
///
/// @param path_ptr Path to the GGUF model file (null-terminated).
/// @param params Model loading parameters (NULL for defaults).
/// @return Model handle, or NULL on failure.
export fn llama_model_load(path_ptr: [*:0]const u8, params: ?*const llama_model_params) ?*llama_model {
    if (!build_options.enable_llm) return null;

    const path = std.mem.sliceTo(path_ptr, 0);
    const load_params = if (params) |p| p.* else llama_model_params{};

    // Allocate model state
    const state = allocator.create(ModelState) catch return null;
    errdefer allocator.destroy(state);

    // Copy path
    state.path = allocator.dupe(u8, path) catch return null;

    // Log model loading options
    if (load_params.n_gpu_layers != 0) {
        std.log.info("Loading model with n_gpu_layers={d}, use_mmap={}", .{
            load_params.n_gpu_layers,
            load_params.use_mmap,
        });
    }

    // Load GGUF file
    if (build_options.enable_llm) {
        // Note: GGUF loading currently always uses mmap on supported platforms.
        // The use_mmap parameter is reserved for future use when we add
        // non-mmap loading support.
        state.gguf_model = gguf.GgufFile.open(allocator, path) catch |err| {
            std.log.err("Failed to load GGUF: {t}", .{err});
            allocator.free(state.path);
            return null;
        };

        if (state.gguf_model) |model| {
            // Extract model parameters from metadata
            state.vocab_size = model.getVocabSize() orelse 32000;
            state.n_layers = model.getBlockCount() orelse 32;
            state.n_heads = model.getHeadCount() orelse 32;
            state.n_embd = model.getEmbeddingLength() orelse 4096;
        }
    }

    return @ptrCast(state);
}

/// Free a model and its resources.
///
/// @param model Model handle to free.
export fn llama_model_free(model: ?*llama_model) void {
    if (model == null) return;
    const state: *ModelState = @ptrCast(@alignCast(model));

    if (build_options.enable_llm) {
        if (state.gguf_model) |*m| {
            m.deinit();
        }
    }

    allocator.free(state.path);
    allocator.destroy(state);
}

/// Get the vocabulary size of the model.
export fn llama_model_vocab_size(model: ?*const llama_model) i32 {
    if (model == null) return 0;
    const state: *const ModelState = @ptrCast(@alignCast(model));
    return @intCast(state.vocab_size);
}

/// Get the number of layers in the model.
export fn llama_model_n_layers(model: ?*const llama_model) i32 {
    if (model == null) return 0;
    const state: *const ModelState = @ptrCast(@alignCast(model));
    return @intCast(state.n_layers);
}

/// Get the embedding dimension of the model.
export fn llama_model_n_embd(model: ?*const llama_model) i32 {
    if (model == null) return 0;
    const state: *const ModelState = @ptrCast(@alignCast(model));
    return @intCast(state.n_embd);
}

/// Get the number of attention heads.
export fn llama_model_n_heads(model: ?*const llama_model) i32 {
    if (model == null) return 0;
    const state: *const ModelState = @ptrCast(@alignCast(model));
    return @intCast(state.n_heads);
}

//==============================================================================
// Context Functions
//==============================================================================

/// Create an inference context for a model.
///
/// @param model Model handle.
/// @param params Context parameters (NULL for defaults).
/// @return Context handle, or NULL on failure.
export fn llama_context_create(model: ?*llama_model, params: ?*const llama_context_params) ?*llama_context {
    if (!build_options.enable_llm) return null;
    if (model == null) return null;

    const model_state: *ModelState = @ptrCast(@alignCast(model));
    const ctx_params = if (params) |p| p.* else llama_context_params{};

    // Allocate context state
    const state = allocator.create(ContextState) catch return null;
    errdefer allocator.destroy(state);

    state.model = model_state;
    state.params = ctx_params;
    state.kv_pos = 0;

    if (build_options.enable_llm) {
        // Initialize tokenizer from GGUF metadata
        state.tokenizer = initTokenizerFromModel(model_state);

        // Initialize sampler with default config
        state.sampler = generation.Sampler.init(allocator, .{
            .temperature = 0.8,
            .top_k = 40,
            .top_p = 0.95,
        });

        // Initialize GPU context if available
        state.gpu_ctx = ops.GpuOpsContext.init(allocator);

        // Allocate KV cache
        const cache_size = @as(usize, ctx_params.n_ctx) * model_state.n_embd;
        state.kv_cache_k = allocator.alloc(f32, cache_size * model_state.n_layers) catch null;
        state.kv_cache_v = allocator.alloc(f32, cache_size * model_state.n_layers) catch null;
    }

    return @ptrCast(state);
}

/// Initialize tokenizer from model GGUF metadata.
fn initTokenizerFromModel(model_state: *ModelState) ?tokenizer_mod.Tokenizer {
    if (!build_options.enable_llm) return null;

    const gguf_model = model_state.gguf_model orelse return null;

    // Detect tokenizer type from GGUF metadata
    const model_type = gguf_model.getTokenizerModel();
    const tok_kind = tokenizer_mod.TokenizerKind.fromGgufModel(model_type);

    if (tok_kind == .unknown) {
        // Fall back to SentencePiece for LLaMA-style models
        return tokenizer_mod.Tokenizer.init(allocator, .sentencepiece);
    }

    var tok = tokenizer_mod.Tokenizer.init(allocator, tok_kind);

    // Configure special tokens from GGUF
    if (gguf_model.getAddBosToken()) |add_bos| {
        tok.setAddBos(add_bos);
    }
    if (gguf_model.getAddEosToken()) |add_eos| {
        tok.setAddEos(add_eos);
    }

    // Load vocabulary from GGUF arrays
    const tokens_array = gguf_model.getTokensArray() orelse return tok;
    const scores_array = gguf_model.getScoresArray();

    // Load vocabulary into the tokenizer based on type
    switch (tok) {
        .sentencepiece => |*sp| {
            if (scores_array) |scores| {
                sp.loadFromGgufMetadata(
                    tokens_array.data,
                    scores.data,
                    tokens_array.count,
                ) catch return tok;
            }
        },
        .bpe => |*bpe| {
            // BPE tokenizer needs tokens and merge rules
            _ = bpe;
            // Would load vocab from tokens_array
            // Would load merges from gguf_model.getMergesArray()
        },
        .unknown => {},
    }

    return tok;
}

/// Free a context and its resources.
///
/// @param ctx Context handle to free.
export fn llama_context_free(ctx: ?*llama_context) void {
    if (ctx == null) return;
    const state: *ContextState = @ptrCast(@alignCast(ctx));

    if (build_options.enable_llm) {
        // Deinit tokenizer if present
        if (state.tokenizer) |*tok| {
            tok.deinit();
        }

        state.sampler.deinit();

        if (state.gpu_ctx) |*gpu| {
            gpu.deinit();
        }

        if (state.kv_cache_k) |k| allocator.free(k);
        if (state.kv_cache_v) |v| allocator.free(v);
    }

    allocator.destroy(state);
}

/// Reset the KV cache, clearing all cached context.
export fn llama_context_reset(ctx: ?*llama_context) void {
    if (ctx == null) return;
    const state: *ContextState = @ptrCast(@alignCast(ctx));
    state.kv_pos = 0;

    if (state.kv_cache_k) |k| @memset(k, 0);
    if (state.kv_cache_v) |v| @memset(v, 0);
}

/// Get the context size (max sequence length).
export fn llama_context_n_ctx(ctx: ?*const llama_context) i32 {
    if (ctx == null) return 0;
    const state: *const ContextState = @ptrCast(@alignCast(ctx));
    return @intCast(state.params.n_ctx);
}

//==============================================================================
// Tokenization Functions
//==============================================================================

/// Tokenize a text string.
///
/// @param ctx Context handle.
/// @param text Input text (null-terminated).
/// @param tokens Output token array.
/// @param n_max_tokens Maximum tokens to output.
/// @param add_bos Add beginning-of-sequence token.
/// @return Number of tokens, or negative on error.
export fn llama_tokenize(
    ctx: ?*llama_context,
    text: [*:0]const u8,
    tokens: [*]i32,
    n_max_tokens: i32,
    add_bos: bool,
) i32 {
    if (!build_options.enable_llm) return -1;
    if (ctx == null) return -1;

    const state: *ContextState = @ptrCast(@alignCast(ctx));
    const input = std.mem.sliceTo(text, 0);

    // Try to use the real tokenizer if available
    if (state.tokenizer) |*tok| {
        // Configure add_bos for this call
        tok.setAddBos(add_bos);

        // Encode the text
        const encoded = tok.encode(allocator, input) catch {
            // Fall back to byte-level on error
            return byteLevelTokenize(input, tokens, n_max_tokens, add_bos);
        };
        defer allocator.free(encoded);

        // Copy tokens to output buffer
        var n_tokens: i32 = 0;
        for (encoded) |token_id| {
            if (n_tokens >= n_max_tokens) break;
            tokens[@intCast(n_tokens)] = @intCast(token_id);
            n_tokens += 1;
        }

        return n_tokens;
    }

    // Fall back to byte-level tokenization
    return byteLevelTokenize(input, tokens, n_max_tokens, add_bos);
}

/// Byte-level fallback tokenization.
fn byteLevelTokenize(input: []const u8, tokens: [*]i32, n_max_tokens: i32, add_bos: bool) i32 {
    var n_tokens: i32 = 0;

    if (add_bos) {
        if (n_tokens < n_max_tokens) {
            tokens[@intCast(n_tokens)] = 1; // BOS token
            n_tokens += 1;
        }
    }

    // Byte-level encoding
    for (input) |byte| {
        if (n_tokens >= n_max_tokens) break;
        tokens[@intCast(n_tokens)] = @as(i32, byte) + 3; // Offset by special tokens
        n_tokens += 1;
    }

    return n_tokens;
}

/// Detokenize tokens back to text.
///
/// @param ctx Context handle.
/// @param tokens Input token array.
/// @param n_tokens Number of tokens.
/// @param text Output text buffer.
/// @param n_max_chars Maximum characters to output.
/// @return Number of characters written, or negative on error.
export fn llama_detokenize(
    ctx: ?*llama_context,
    tokens: [*]const i32,
    n_tokens: i32,
    text: [*]u8,
    n_max_chars: i32,
) i32 {
    if (!build_options.enable_llm) return -1;
    if (ctx == null or n_tokens <= 0) return -1;

    const state: *ContextState = @ptrCast(@alignCast(ctx));

    // Try to use the real tokenizer if available
    if (state.tokenizer) |*tok| {
        // Convert i32 tokens to u32
        var token_ids = allocator.alloc(u32, @intCast(n_tokens)) catch {
            return byteLevelDetokenize(tokens, n_tokens, text, n_max_chars);
        };
        defer allocator.free(token_ids);

        for (0..@intCast(n_tokens)) |i| {
            token_ids[i] = if (tokens[i] >= 0) @intCast(tokens[i]) else 0;
        }

        // Decode the tokens
        const decoded = tok.decode(allocator, token_ids) catch {
            return byteLevelDetokenize(tokens, n_tokens, text, n_max_chars);
        };
        defer allocator.free(decoded);

        // Copy to output buffer
        var n_chars: i32 = 0;
        for (decoded) |c| {
            if (n_chars >= n_max_chars - 1) break;
            text[@intCast(n_chars)] = c;
            n_chars += 1;
        }

        // Null-terminate
        if (n_chars < n_max_chars) {
            text[@intCast(n_chars)] = 0;
        }

        return n_chars;
    }

    // Fall back to byte-level detokenization
    return byteLevelDetokenize(tokens, n_tokens, text, n_max_chars);
}

/// Byte-level fallback detokenization.
fn byteLevelDetokenize(tokens: [*]const i32, n_tokens: i32, text: [*]u8, n_max_chars: i32) i32 {
    var n_chars: i32 = 0;

    for (0..@intCast(n_tokens)) |i| {
        if (n_chars >= n_max_chars - 1) break;

        const token = tokens[i];
        if (token <= 2) continue; // Skip special tokens

        const byte: u8 = @intCast(@max(0, token - 3));
        text[@intCast(n_chars)] = byte;
        n_chars += 1;
    }

    // Null-terminate
    if (n_chars < n_max_chars) {
        text[@intCast(n_chars)] = 0;
    }

    return n_chars;
}

//==============================================================================
// Generation Functions
//==============================================================================

/// Generate tokens given a prompt.
///
/// @param ctx Context handle.
/// @param prompt_tokens Input prompt tokens.
/// @param n_prompt_tokens Number of prompt tokens.
/// @param n_gen_tokens Maximum tokens to generate.
/// @param sampling Sampling parameters (NULL for defaults).
/// @param callback Token callback for streaming (NULL to disable).
/// @param user_data User data passed to callback.
/// @param result Output generation result (NULL to ignore).
/// @return Number of tokens generated, or negative on error.
export fn llama_generate(
    ctx: ?*llama_context,
    prompt_tokens: [*]const i32,
    n_prompt_tokens: i32,
    n_gen_tokens: i32,
    sampling: ?*const llama_sampling_params,
    callback: llama_token_callback,
    user_data: ?*anyopaque,
    result: ?*llama_generation_result,
) i32 {
    if (!build_options.enable_llm) return -1;
    if (ctx == null or n_prompt_tokens <= 0) return -1;

    const state: *ContextState = @ptrCast(@alignCast(ctx));
    const sample_params = if (sampling) |s| s.* else llama_sampling_params{};

    // Update sampler parameters
    if (build_options.enable_llm) {
        state.sampler.config.temperature = sample_params.temperature;
        state.sampler.config.top_k = @intCast(@max(0, sample_params.top_k));
        state.sampler.config.top_p = sample_params.top_p;
        state.sampler.config.repetition_penalty = sample_params.repeat_penalty;
    }

    var timer = std.time.Timer.start() catch return -1;
    var n_generated: i32 = 0;
    var first_token_time: u64 = 0;

    // Process prompt (prefill)
    for (0..@intCast(n_prompt_tokens)) |i| {
        _ = prompt_tokens[i];
        state.kv_pos += 1;
    }

    // Generate tokens
    var last_token: i32 = prompt_tokens[@intCast(n_prompt_tokens - 1)];

    while (n_generated < n_gen_tokens) {
        // Create mock logits for demonstration
        var logits: [32000]f32 = undefined;
        for (&logits) |*l| l.* = 0;

        // Sample next token
        var next_token: i32 = undefined;
        if (build_options.enable_llm) {
            const sampled = state.sampler.sample(&logits);
            next_token = @intCast(sampled);
        } else {
            next_token = 0;
        }

        // Check for EOS
        if (next_token == 2) break;

        // Record first token time
        if (n_generated == 0) {
            first_token_time = timer.read();
        }

        // Callback
        if (callback) |cb| {
            var token_text: [16:0]u8 = undefined;
            _ = llama_detokenize(ctx, &[_]i32{next_token}, 1, &token_text, 16);

            if (!cb(next_token, &token_text, user_data)) {
                break; // Stop generation if callback returns false
            }
        }

        last_token = next_token;
        n_generated += 1;
        state.kv_pos += 1;
    }

    // Fill result
    if (result) |r| {
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        r.n_tokens = n_generated;
        r.time_ms = elapsed_ms;
        r.tokens_per_second = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(n_generated)) / (elapsed_ms / 1000.0)
        else
            0;
        r.n_prompt_tokens = n_prompt_tokens;
        r.time_to_first_token_ms = @as(f64, @floatFromInt(first_token_time)) / 1_000_000.0;
    }

    return n_generated;
}

//==============================================================================
// Sampling Functions
//==============================================================================

/// Create a standalone sampler.
export fn llama_sampler_create(params: ?*const llama_sampling_params) ?*llama_sampler {
    if (!build_options.enable_llm) return null;

    const sample_params = if (params) |p| p.* else llama_sampling_params{};

    const state = allocator.create(generation.Sampler) catch return null;
    state.* = generation.Sampler.init(allocator, .{
        .temperature = sample_params.temperature,
        .top_k = @intCast(@max(0, sample_params.top_k)),
        .top_p = sample_params.top_p,
        .repetition_penalty = sample_params.repeat_penalty,
        .seed = if (sample_params.seed < 0) null else @intCast(sample_params.seed),
    });

    return @ptrCast(state);
}

/// Free a sampler.
export fn llama_sampler_free(sampler: ?*llama_sampler) void {
    if (!build_options.enable_llm) return;
    if (sampler == null) return;

    const state: *generation.Sampler = @ptrCast(@alignCast(sampler));
    state.deinit();
    allocator.destroy(state);
}

/// Sample a token from logits using the sampler.
export fn llama_sampler_sample(sampler: ?*llama_sampler, logits: [*]f32, n_vocab: i32) i32 {
    if (!build_options.enable_llm) return 0;
    if (sampler == null or n_vocab <= 0) return 0;

    const state: *generation.Sampler = @ptrCast(@alignCast(sampler));
    const logit_slice = logits[0..@intCast(n_vocab)];

    return @intCast(state.sample(logit_slice));
}

/// Reset sampler state.
export fn llama_sampler_reset(sampler: ?*llama_sampler) void {
    if (!build_options.enable_llm) return;
    if (sampler == null) return;

    const state: *generation.Sampler = @ptrCast(@alignCast(sampler));
    state.reset();
}

//==============================================================================
// Utility Functions
//==============================================================================

/// Get the last error message.
/// Thread-local error storage.
var last_error: [256]u8 = [_]u8{0} ** 256;

export fn llama_get_last_error() [*:0]const u8 {
    return @ptrCast(&last_error);
}

/// Get library version string.
export fn llama_version() [*:0]const u8 {
    return "abi-llm-0.1.0";
}

/// Check if LLM feature is enabled.
export fn llama_is_enabled() bool {
    return build_options.enable_llm;
}

/// Get default model params.
export fn llama_model_default_params() llama_model_params {
    return llama_model_params{};
}

/// Get default context params.
export fn llama_context_default_params() llama_context_params {
    return llama_context_params{};
}

/// Get default sampling params.
export fn llama_sampling_default_params() llama_sampling_params {
    return llama_sampling_params{};
}

//==============================================================================
// Tests
//==============================================================================

test "c api types" {
    // Verify struct sizes are reasonable for C interop
    try std.testing.expect(@sizeOf(llama_model_params) < 256);
    try std.testing.expect(@sizeOf(llama_context_params) < 256);
    try std.testing.expect(@sizeOf(llama_sampling_params) < 256);
    try std.testing.expect(@sizeOf(llama_generation_result) < 256);
}

test "default params" {
    const model_params = llama_model_default_params();
    try std.testing.expectEqual(@as(i32, -1), model_params.n_gpu_layers);

    const ctx_params = llama_context_default_params();
    try std.testing.expectEqual(@as(u32, 2048), ctx_params.n_ctx);

    const sample_params = llama_sampling_default_params();
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), sample_params.temperature, 0.001);
}

test "version string" {
    const ver = llama_version();
    try std.testing.expect(ver[0] != 0);
}
