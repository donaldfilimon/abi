/**
 * ABI Framework - C-Compatible LLM API
 *
 * Provides C-compatible bindings for LLM inference, following llama.cpp naming
 * conventions for easy integration with existing tooling and libraries.
 *
 * Example usage:
 *   llama_model* model = llama_model_load("model.gguf", NULL);
 *   llama_context* ctx = llama_context_create(model, NULL);
 *   int32_t tokens[512];
 *   int32_t n_tokens = llama_tokenize(ctx, "Hello world", tokens, 512, true);
 *   llama_generate(ctx, tokens, n_tokens, 100, NULL, NULL, NULL, NULL);
 *   llama_context_free(ctx);
 *   llama_model_free(model);
 */

#ifndef ABI_LLM_H
#define ABI_LLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*==============================================================================
 * Opaque Handle Types
 *============================================================================*/

/** Opaque model handle. */
typedef struct llama_model llama_model;

/** Opaque context handle. */
typedef struct llama_context llama_context;

/** Opaque tokenizer handle. */
typedef struct llama_tokenizer llama_tokenizer;

/** Opaque sampler handle. */
typedef struct llama_sampler llama_sampler;

/*==============================================================================
 * Configuration Structures
 *============================================================================*/

/** Model loading parameters. */
typedef struct llama_model_params {
    /** Number of layers to offload to GPU (-1 = all, 0 = none). */
    int32_t n_gpu_layers;
    /** Use memory mapping for model weights. */
    bool use_mmap;
    /** Use memory locking (mlock) for model weights. */
    bool use_mlock;
    /** Vocabulary-only mode (no weights). */
    bool vocab_only;
    /** Reserved for future use. */
    uint8_t _reserved[32];
} llama_model_params;

/** Context creation parameters. */
typedef struct llama_context_params {
    /** Context size (max sequence length). */
    uint32_t n_ctx;
    /** Batch size for prompt processing. */
    uint32_t n_batch;
    /** Number of threads for generation. */
    uint32_t n_threads;
    /** Number of threads for batch processing. */
    uint32_t n_threads_batch;
    /** RoPE base frequency. */
    float rope_freq_base;
    /** RoPE frequency scale. */
    float rope_freq_scale;
    /** Use flash attention. */
    bool flash_attn;
    /** Reserved for future use. */
    uint8_t _reserved[32];
} llama_context_params;

/** Sampling parameters. */
typedef struct llama_sampling_params {
    /** Temperature for sampling. */
    float temperature;
    /** Top-K sampling (0 = disabled). */
    int32_t top_k;
    /** Top-P (nucleus) sampling. */
    float top_p;
    /** Min-P sampling. */
    float min_p;
    /** Typical-P sampling. */
    float typical_p;
    /** Repetition penalty. */
    float repeat_penalty;
    /** Repetition penalty window. */
    int32_t repeat_last_n;
    /** Frequency penalty. */
    float frequency_penalty;
    /** Presence penalty. */
    float presence_penalty;
    /** Mirostat mode (0 = disabled, 1 = v1, 2 = v2). */
    int32_t mirostat;
    /** Mirostat target entropy. */
    float mirostat_tau;
    /** Mirostat learning rate. */
    float mirostat_eta;
    /** Random seed (-1 = random). */
    int64_t seed;
    /** Reserved for future use. */
    uint8_t _reserved[32];
} llama_sampling_params;

/** Token callback for streaming generation.
 *  Return true to continue, false to stop. */
typedef bool (*llama_token_callback)(int32_t token_id, const char* text, void* user_data);

/** Generation result structure. */
typedef struct llama_generation_result {
    /** Number of tokens generated. */
    int32_t n_tokens;
    /** Total generation time in milliseconds. */
    double time_ms;
    /** Tokens per second. */
    double tokens_per_second;
    /** Prompt tokens processed. */
    int32_t n_prompt_tokens;
    /** Time to first token in milliseconds. */
    double time_to_first_token_ms;
    /** Reserved for future use. */
    uint8_t _reserved[32];
} llama_generation_result;

/*==============================================================================
 * Model Functions
 *============================================================================*/

/**
 * Load a model from a GGUF file.
 *
 * @param path Path to the GGUF model file (null-terminated).
 * @param params Model loading parameters (NULL for defaults).
 * @return Model handle, or NULL on failure.
 */
llama_model* llama_model_load(const char* path, const llama_model_params* params);

/**
 * Free a model and its resources.
 *
 * @param model Model handle to free.
 */
void llama_model_free(llama_model* model);

/**
 * Get the vocabulary size of the model.
 */
int32_t llama_model_vocab_size(const llama_model* model);

/**
 * Get the number of layers in the model.
 */
int32_t llama_model_n_layers(const llama_model* model);

/**
 * Get the embedding dimension of the model.
 */
int32_t llama_model_n_embd(const llama_model* model);

/**
 * Get the number of attention heads.
 */
int32_t llama_model_n_heads(const llama_model* model);

/*==============================================================================
 * Context Functions
 *============================================================================*/

/**
 * Create an inference context for a model.
 *
 * @param model Model handle.
 * @param params Context parameters (NULL for defaults).
 * @return Context handle, or NULL on failure.
 */
llama_context* llama_context_create(llama_model* model, const llama_context_params* params);

/**
 * Free a context and its resources.
 *
 * @param ctx Context handle to free.
 */
void llama_context_free(llama_context* ctx);

/**
 * Reset the KV cache, clearing all cached context.
 */
void llama_context_reset(llama_context* ctx);

/**
 * Get the context size (max sequence length).
 */
int32_t llama_context_n_ctx(const llama_context* ctx);

/*==============================================================================
 * Tokenization Functions
 *============================================================================*/

/**
 * Tokenize a text string.
 *
 * @param ctx Context handle.
 * @param text Input text (null-terminated).
 * @param tokens Output token array.
 * @param n_max_tokens Maximum tokens to output.
 * @param add_bos Add beginning-of-sequence token.
 * @return Number of tokens, or negative on error.
 */
int32_t llama_tokenize(
    llama_context* ctx,
    const char* text,
    int32_t* tokens,
    int32_t n_max_tokens,
    bool add_bos
);

/**
 * Detokenize tokens back to text.
 *
 * @param ctx Context handle.
 * @param tokens Input token array.
 * @param n_tokens Number of tokens.
 * @param text Output text buffer.
 * @param n_max_chars Maximum characters to output.
 * @return Number of characters written, or negative on error.
 */
int32_t llama_detokenize(
    llama_context* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    char* text,
    int32_t n_max_chars
);

/*==============================================================================
 * Generation Functions
 *============================================================================*/

/**
 * Generate tokens given a prompt.
 *
 * @param ctx Context handle.
 * @param prompt_tokens Input prompt tokens.
 * @param n_prompt_tokens Number of prompt tokens.
 * @param n_gen_tokens Maximum tokens to generate.
 * @param sampling Sampling parameters (NULL for defaults).
 * @param callback Token callback for streaming (NULL to disable).
 * @param user_data User data passed to callback.
 * @param result Output generation result (NULL to ignore).
 * @return Number of tokens generated, or negative on error.
 */
int32_t llama_generate(
    llama_context* ctx,
    const int32_t* prompt_tokens,
    int32_t n_prompt_tokens,
    int32_t n_gen_tokens,
    const llama_sampling_params* sampling,
    llama_token_callback callback,
    void* user_data,
    llama_generation_result* result
);

/*==============================================================================
 * Sampling Functions
 *============================================================================*/

/**
 * Create a standalone sampler.
 */
llama_sampler* llama_sampler_create(const llama_sampling_params* params);

/**
 * Free a sampler.
 */
void llama_sampler_free(llama_sampler* sampler);

/**
 * Sample a token from logits using the sampler.
 */
int32_t llama_sampler_sample(llama_sampler* sampler, float* logits, int32_t n_vocab);

/**
 * Reset sampler state.
 */
void llama_sampler_reset(llama_sampler* sampler);

/*==============================================================================
 * Utility Functions
 *============================================================================*/

/**
 * Get the last error message.
 */
const char* llama_get_last_error(void);

/**
 * Get library version string.
 */
const char* llama_version(void);

/**
 * Check if LLM feature is enabled.
 */
bool llama_is_enabled(void);

/**
 * Get default model params.
 */
llama_model_params llama_model_default_params(void);

/**
 * Get default context params.
 */
llama_context_params llama_context_default_params(void);

/**
 * Get default sampling params.
 */
llama_sampling_params llama_sampling_default_params(void);

#ifdef __cplusplus
}
#endif

#endif /* ABI_LLM_H */
