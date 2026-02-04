/**
 * ABI Framework JavaScript/WASM Bindings
 *
 * Provides access to ABI functionality in JavaScript environments:
 * - SIMD vector operations (with WASM SIMD support)
 * - Vector database for similarity search
 * - GPU acceleration (via WebGPU when available)
 *
 * @example
 * ```typescript
 * import { Abi, Simd, VectorDatabase } from '@abi/core';
 *
 * // Initialize
 * const abi = await Abi.init();
 *
 * // SIMD operations
 * const dot = Simd.dotProduct([1, 2, 3], [4, 5, 6]);
 *
 * // Vector database
 * const db = new VectorDatabase('test', 128);
 * await db.insert(1, new Float32Array(128));
 * ```
 */

export { Abi, AbiOptions } from './abi';
export { Simd, SimdCaps } from './simd';
export { VectorDatabase, SearchResult, DatabaseConfig } from './database';
export { Gpu, GpuConfig, Backend } from './gpu';
export { AbiError, ErrorCode } from './error';

// Re-export WASM module loader for advanced usage
export { loadWasm, WasmModule } from './wasm';
