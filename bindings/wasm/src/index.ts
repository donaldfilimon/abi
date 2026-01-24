/**
 * @anthropic/abi-wasm - WebAssembly bindings for the ABI Framework
 *
 * Provides high-performance AI and vector database operations in JavaScript/TypeScript
 * environments via WebAssembly.
 */

// ============================================================================
// Types
// ============================================================================

/**
 * ABI WASM module exports from the compiled Zig code
 */
export interface AbiWasmExports {
  /** WebAssembly memory */
  memory: WebAssembly.Memory;
  /** Initialize the framework */
  abi_init: () => number;
  /** Shutdown the framework */
  abi_shutdown: () => void;
  /** Get the version string length */
  abi_version_len: () => number;
  /** Copy version string to buffer */
  abi_version_get: (ptr: number, len: number) => number;
  /** Allocate memory in WASM linear memory */
  abi_alloc: (len: number) => number;
  /** Free previously allocated memory */
  abi_free: (ptr: number, len: number) => void;
}

/**
 * Configuration options for initializing the ABI WASM module
 */
export interface AbiInitOptions {
  /**
   * Path or URL to the WASM file.
   * Defaults to looking for 'abi.wasm' in common locations.
   */
  wasmPath?: string | URL;
  /**
   * Pre-loaded WASM binary as ArrayBuffer or Uint8Array.
   * If provided, wasmPath is ignored.
   */
  wasmBinary?: ArrayBuffer | Uint8Array;
  /**
   * Custom import object to extend default imports.
   */
  imports?: WebAssembly.Imports;
}

/**
 * Result of a vector similarity search
 */
export interface SearchResult {
  /** Unique identifier of the vector */
  id: number;
  /** Similarity score (higher is more similar) */
  score: number;
  /** Optional metadata associated with the vector */
  metadata?: Record<string, unknown>;
}

/**
 * Vector database configuration
 */
export interface VectorDatabaseConfig {
  /** Name of the database */
  name: string;
  /** Dimensionality of vectors (auto-detected from first insert if not specified) */
  dimensions?: number;
}

// ============================================================================
// Internal State
// ============================================================================

let wasmInstance: WebAssembly.Instance | null = null;
let wasmExports: AbiWasmExports | null = null;
let isInitialized = false;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Encode a string to UTF-8 bytes in WASM memory
 */
function encodeString(str: string): { ptr: number; len: number } {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized");
  }

  const encoder = new TextEncoder();
  const bytes = encoder.encode(str);
  const ptr = wasmExports.abi_alloc(bytes.length);

  if (ptr === 0) {
    throw new Error("Failed to allocate memory in WASM");
  }

  const memory = new Uint8Array(wasmExports.memory.buffer);
  memory.set(bytes, ptr);

  return { ptr, len: bytes.length };
}

/**
 * Decode a UTF-8 string from WASM memory
 */
function decodeString(ptr: number, len: number): string {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized");
  }

  const memory = new Uint8Array(wasmExports.memory.buffer);
  const bytes = memory.slice(ptr, ptr + len);
  const decoder = new TextDecoder();
  return decoder.decode(bytes);
}

/**
 * Free allocated string memory
 */
function freeString(ptr: number, len: number): void {
  if (wasmExports && ptr !== 0) {
    wasmExports.abi_free(ptr, len);
  }
}

// ============================================================================
// Default WASM Paths
// ============================================================================

const DEFAULT_WASM_PATHS = [
  "./abi.wasm",
  "../wasm/abi.wasm",
  "../../zig-out/wasm/abi.wasm",
  "./node_modules/@anthropic/abi-wasm/wasm/abi.wasm",
];

/**
 * Attempt to fetch WASM from multiple paths
 */
async function fetchWasmBinary(
  customPath?: string | URL
): Promise<ArrayBuffer> {
  const paths = customPath ? [customPath.toString()] : DEFAULT_WASM_PATHS;

  for (const path of paths) {
    try {
      const response = await fetch(path);
      if (response.ok) {
        return response.arrayBuffer();
      }
    } catch {
      // Try next path
      continue;
    }
  }

  throw new Error(
    `Failed to load ABI WASM module. Tried paths: ${paths.join(", ")}. ` +
      "Please provide wasmPath or wasmBinary in init options."
  );
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Initialize the ABI WASM module.
 *
 * This must be called before using any other ABI functions.
 * It is safe to call multiple times - subsequent calls will be no-ops.
 *
 * @param options - Configuration options
 * @returns Promise that resolves when initialization is complete
 *
 * @example
 * ```typescript
 * import { init } from '@anthropic/abi-wasm';
 *
 * // Basic initialization (auto-detects WASM path)
 * await init();
 *
 * // With custom WASM path
 * await init({ wasmPath: '/path/to/abi.wasm' });
 *
 * // With pre-loaded binary
 * const binary = await fetch('/abi.wasm').then(r => r.arrayBuffer());
 * await init({ wasmBinary: binary });
 * ```
 */
export async function init(options: AbiInitOptions = {}): Promise<void> {
  if (isInitialized) {
    return;
  }

  // Get WASM binary
  const binary =
    options.wasmBinary ?? (await fetchWasmBinary(options.wasmPath));

  // Create import object with env stubs
  const importObject: WebAssembly.Imports = {
    env: {
      // Memory will be provided by WASM module
    },
    ...options.imports,
  };

  // Instantiate WASM module
  const result = await WebAssembly.instantiate(binary, importObject);
  wasmInstance = result.instance;
  wasmExports = wasmInstance.exports as unknown as AbiWasmExports;

  // Initialize the framework
  const initResult = wasmExports.abi_init();
  if (initResult !== 0) {
    wasmInstance = null;
    wasmExports = null;
    throw new Error(`Failed to initialize ABI framework (error code: ${initResult})`);
  }

  isInitialized = true;
}

/**
 * Shutdown the ABI framework and release resources.
 *
 * After calling shutdown, you must call init() again before using ABI functions.
 */
export function shutdown(): void {
  if (!isInitialized || !wasmExports) {
    return;
  }

  wasmExports.abi_shutdown();
  wasmInstance = null;
  wasmExports = null;
  isInitialized = false;
}

/**
 * Check if the ABI framework is initialized.
 *
 * @returns true if init() has been called successfully
 */
export function isReady(): boolean {
  return isInitialized;
}

/**
 * Get the ABI framework version string.
 *
 * @returns The version string (e.g., "0.3.0")
 * @throws Error if not initialized
 *
 * @example
 * ```typescript
 * console.log(`ABI version: ${version()}`);
 * ```
 */
export function version(): string {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized. Call init() first.");
  }

  const len = wasmExports.abi_version_len();
  const ptr = wasmExports.abi_alloc(len);

  if (ptr === 0) {
    throw new Error("Failed to allocate memory for version string");
  }

  try {
    const actualLen = wasmExports.abi_version_get(ptr, len);
    return decodeString(ptr, actualLen);
  } finally {
    wasmExports.abi_free(ptr, len);
  }
}

/**
 * Allocate memory in the WASM linear memory.
 *
 * This is useful for advanced use cases where you need to pass
 * binary data to WASM functions directly.
 *
 * @param size - Number of bytes to allocate
 * @returns Pointer to allocated memory, or 0 on failure
 */
export function alloc(size: number): number {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized. Call init() first.");
  }
  return wasmExports.abi_alloc(size);
}

/**
 * Free memory previously allocated with alloc().
 *
 * @param ptr - Pointer to memory to free
 * @param size - Size of the allocation
 */
export function free(ptr: number, size: number): void {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized. Call init() first.");
  }
  wasmExports.abi_free(ptr, size);
}

/**
 * Get direct access to WASM memory.
 *
 * This is useful for advanced use cases where you need to
 * read/write data directly to WASM memory.
 *
 * @returns The WASM Memory object
 */
export function getMemory(): WebAssembly.Memory {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized. Call init() first.");
  }
  return wasmExports.memory;
}

/**
 * Get the raw WASM exports object.
 *
 * This provides access to all exported functions for advanced use cases.
 *
 * @returns The WASM exports object
 */
export function getExports(): AbiWasmExports {
  if (!wasmExports) {
    throw new Error("ABI WASM not initialized. Call init() first.");
  }
  return wasmExports;
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Compute the cosine similarity between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector (must have same length as a)
 * @returns Cosine similarity value between -1 and 1
 *
 * @example
 * ```typescript
 * const a = [1.0, 2.0, 3.0];
 * const b = [4.0, 5.0, 6.0];
 * const similarity = cosineSimilarity(a, b);
 * console.log(`Similarity: ${similarity}`);
 * ```
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) {
    return 0;
  }

  return dotProduct / magnitude;
}

/**
 * Compute the dot product of two vectors.
 *
 * @param a - First vector
 * @param b - Second vector (must have same length as a)
 * @returns Dot product value
 */
export function dotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }
  return result;
}

/**
 * Compute the L2 (Euclidean) norm of a vector.
 *
 * @param vec - Input vector
 * @returns L2 norm value
 */
export function l2Norm(vec: number[]): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}

/**
 * Normalize a vector to unit length.
 *
 * @param vec - Input vector
 * @returns Normalized vector
 */
export function normalize(vec: number[]): number[] {
  const norm = l2Norm(vec);
  if (norm === 0) {
    return vec.slice();
  }
  return vec.map((v) => v / norm);
}

/**
 * Add two vectors element-wise.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Result vector
 */
export function vectorAdd(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }
  return a.map((v, i) => v + b[i]);
}

/**
 * Subtract two vectors element-wise.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Result vector (a - b)
 */
export function vectorSub(a: number[], b: number[]): number[] {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }
  return a.map((v, i) => v - b[i]);
}

/**
 * Scale a vector by a scalar value.
 *
 * @param vec - Input vector
 * @param scalar - Scale factor
 * @returns Scaled vector
 */
export function vectorScale(vec: number[], scalar: number): number[] {
  return vec.map((v) => v * scalar);
}

// ============================================================================
// In-Memory Vector Database (JavaScript implementation)
// ============================================================================

interface VectorEntry {
  id: number;
  vector: number[];
  metadata?: Record<string, unknown>;
}

/**
 * In-memory vector database for similarity search.
 *
 * This is a pure JavaScript implementation suitable for small to medium datasets.
 * For larger datasets, consider using the native database backend.
 *
 * @example
 * ```typescript
 * const db = new VectorDatabase({ name: 'embeddings' });
 *
 * // Add vectors
 * db.add([1.0, 0.0, 0.0], { label: 'x-axis' });
 * db.add([0.0, 1.0, 0.0], { label: 'y-axis' });
 * db.add([0.0, 0.0, 1.0], { label: 'z-axis' });
 *
 * // Search for similar vectors
 * const results = db.search([0.9, 0.1, 0.0], 2);
 * console.log(results);
 * ```
 */
export class VectorDatabase {
  private name: string;
  private dimensions: number | null;
  private vectors: VectorEntry[] = [];
  private nextId = 0;

  constructor(config: VectorDatabaseConfig) {
    this.name = config.name;
    this.dimensions = config.dimensions ?? null;
  }

  /**
   * Get the database name.
   */
  getName(): string {
    return this.name;
  }

  /**
   * Get the number of vectors in the database.
   */
  size(): number {
    return this.vectors.length;
  }

  /**
   * Get the dimensionality of vectors in the database.
   */
  getDimensions(): number | null {
    return this.dimensions;
  }

  /**
   * Add a vector to the database.
   *
   * @param vector - The vector to add
   * @param metadata - Optional metadata to associate with the vector
   * @returns The ID of the added vector
   */
  add(vector: number[], metadata?: Record<string, unknown>): number {
    if (this.dimensions === null) {
      this.dimensions = vector.length;
    } else if (vector.length !== this.dimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`
      );
    }

    const id = this.nextId++;
    this.vectors.push({
      id,
      vector: vector.slice(), // Copy to prevent external modification
      metadata,
    });
    return id;
  }

  /**
   * Remove a vector by ID.
   *
   * @param id - The ID of the vector to remove
   * @returns true if the vector was found and removed
   */
  remove(id: number): boolean {
    const index = this.vectors.findIndex((v) => v.id === id);
    if (index === -1) {
      return false;
    }
    this.vectors.splice(index, 1);
    return true;
  }

  /**
   * Get a vector by ID.
   *
   * @param id - The ID of the vector
   * @returns The vector entry or null if not found
   */
  get(id: number): VectorEntry | null {
    return this.vectors.find((v) => v.id === id) ?? null;
  }

  /**
   * Search for similar vectors.
   *
   * @param query - The query vector
   * @param topK - Number of results to return
   * @returns Array of search results sorted by similarity (descending)
   */
  search(query: number[], topK: number = 10): SearchResult[] {
    if (this.dimensions !== null && query.length !== this.dimensions) {
      throw new Error(
        `Query dimension mismatch: expected ${this.dimensions}, got ${query.length}`
      );
    }

    // Compute similarities
    const results: SearchResult[] = this.vectors.map((entry) => ({
      id: entry.id,
      score: cosineSimilarity(query, entry.vector),
      metadata: entry.metadata,
    }));

    // Sort by score descending and take top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Clear all vectors from the database.
   */
  clear(): void {
    this.vectors = [];
    this.nextId = 0;
    this.dimensions = null;
  }

  /**
   * Export the database to a JSON-serializable object.
   */
  toJSON(): object {
    return {
      name: this.name,
      dimensions: this.dimensions,
      vectors: this.vectors,
      nextId: this.nextId,
    };
  }

  /**
   * Import data from a previously exported JSON object.
   */
  static fromJSON(data: {
    name: string;
    dimensions: number | null;
    vectors: VectorEntry[];
    nextId: number;
  }): VectorDatabase {
    const db = new VectorDatabase({
      name: data.name,
      dimensions: data.dimensions ?? undefined,
    });
    db.vectors = data.vectors;
    db.nextId = data.nextId;
    return db;
  }
}

// ============================================================================
// LLM Streaming API (Mock implementation for WASM)
// ============================================================================

/**
 * Configuration for streaming text generation.
 */
export interface StreamingConfigOptions {
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature (0.0 = greedy, 1.0 = default) */
  temperature?: number;
  /** Top-p nucleus sampling threshold */
  topP?: number;
  /** Top-k sampling (0 = disabled) */
  topK?: number;
  /** Repetition penalty (1.0 = disabled) */
  repetitionPenalty?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/**
 * Configuration for streaming text generation.
 */
export class StreamingConfig {
  readonly maxTokens: number;
  readonly temperature: number;
  readonly topP: number;
  readonly topK: number;
  readonly repetitionPenalty: number;
  readonly seed: number;

  constructor(options: StreamingConfigOptions = {}) {
    this.maxTokens = options.maxTokens ?? 256;
    this.temperature = options.temperature ?? 0.7;
    this.topP = options.topP ?? 0.9;
    this.topK = options.topK ?? 40;
    this.repetitionPenalty = options.repetitionPenalty ?? 1.1;
    this.seed = options.seed ?? 0;
  }
}

/**
 * Token event from streaming generation.
 */
export interface TokenEvent {
  /** The generated text for this token */
  text: string;
  /** Token ID in vocabulary */
  tokenId: number;
  /** Position in generated sequence */
  position: number;
  /** Whether this is the final token */
  isFinal: boolean;
  /** Timestamp in nanoseconds */
  timestampNs: number;
}

/**
 * LLM Engine for text generation.
 *
 * In WASM, this provides a mock implementation suitable for testing.
 * Real inference would require connecting to a backend service.
 */
export class LlmEngine {
  private isModelLoaded = false;

  /**
   * Check if a model is loaded.
   */
  get isLoaded(): boolean {
    return this.isModelLoaded;
  }

  /**
   * Load a model (mock implementation).
   */
  loadModel(path: string): void {
    this.isModelLoaded = true;
  }

  /**
   * Unload the current model.
   */
  unloadModel(): void {
    this.isModelLoaded = false;
  }

  /**
   * Generate text with streaming output.
   *
   * @param prompt - Input prompt
   * @param config - Streaming configuration
   * @returns Async iterator of token events
   */
  async *generateStreaming(
    prompt: string,
    config: StreamingConfig = new StreamingConfig()
  ): AsyncGenerator<TokenEvent> {
    if (!this.isModelLoaded) {
      throw new Error("No model loaded. Call loadModel() first.");
    }

    // Mock streaming generation
    const mockTokens = ["The", " AI", " responds", ":", " ", prompt.slice(0, 20), "..."];
    const limit = Math.min(mockTokens.length, config.maxTokens);

    for (let i = 0; i < limit; i++) {
      const isFinal = i === limit - 1;
      yield {
        text: mockTokens[i],
        tokenId: i + 1,
        position: i,
        isFinal,
        timestampNs: Date.now() * 1_000_000,
      };
      // Small delay to simulate streaming
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }

  /**
   * Generate text (non-streaming).
   *
   * @param prompt - Input prompt
   * @param maxTokens - Maximum tokens to generate
   * @returns Generated text
   */
  async generate(prompt: string, maxTokens: number = 256): Promise<string> {
    const tokens: string[] = [];
    const config = new StreamingConfig({ maxTokens });

    for await (const event of this.generateStreaming(prompt, config)) {
      tokens.push(event.text);
    }

    return tokens.join("");
  }
}

// ============================================================================
// Re-exports for convenience
// ============================================================================

export { encodeString, decodeString, freeString };

// Default export for CommonJS compatibility
export default {
  init,
  shutdown,
  isReady,
  version,
  alloc,
  free,
  getMemory,
  getExports,
  cosineSimilarity,
  dotProduct,
  l2Norm,
  normalize,
  vectorAdd,
  vectorSub,
  vectorScale,
  VectorDatabase,
  StreamingConfig,
  LlmEngine,
};
