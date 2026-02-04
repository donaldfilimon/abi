/**
 * SIMD vector operations.
 *
 * Uses WASM SIMD when available, falls back to JavaScript implementation.
 */

import { WasmModule } from './wasm';

/**
 * SIMD capability flags.
 */
export interface SimdCaps {
  /** WASM SIMD support */
  wasmSimd: boolean;
  /** WebGPU compute support */
  webgpu: boolean;
}

/**
 * SIMD-accelerated vector operations.
 *
 * @example
 * ```typescript
 * const a = [1, 2, 3, 4];
 * const b = [5, 6, 7, 8];
 *
 * const dot = Simd.dotProduct(a, b);
 * const similarity = Simd.cosineSimilarity(a, b);
 * ```
 */
export class Simd {
  /**
   * Check if SIMD is available (WASM SIMD).
   */
  static isAvailable(): boolean {
    // Check for WASM SIMD support
    try {
      if (WasmModule.isLoaded()) {
        return WasmModule.getInstance().getExports().abi_simd_available() !== 0;
      }
    } catch {
      // Fall through to feature detection
    }

    // Feature detection for WASM SIMD
    return typeof WebAssembly !== 'undefined' && WebAssembly.validate(
      new Uint8Array([
        0x00, 0x61, 0x73, 0x6d, // WASM magic
        0x01, 0x00, 0x00, 0x00, // Version 1
        0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // Type section (v128)
        0x03, 0x02, 0x01, 0x00, // Function section
        0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x0b // v128.const
      ])
    );
  }

  /**
   * Get SIMD capabilities.
   */
  static capabilities(): SimdCaps {
    return {
      wasmSimd: Simd.isAvailable(),
      webgpu: typeof navigator !== 'undefined' && 'gpu' in navigator,
    };
  }

  /**
   * Vector element-wise addition.
   */
  static add(a: number[] | Float32Array, b: number[] | Float32Array): Float32Array {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    // Try WASM implementation
    if (WasmModule.isLoaded()) {
      const wasm = WasmModule.getInstance();
      const arrA = a instanceof Float32Array ? a : new Float32Array(a);
      const arrB = b instanceof Float32Array ? b : new Float32Array(b);

      const ptrA = wasm.writeFloat32Array(arrA);
      const ptrB = wasm.writeFloat32Array(arrB);
      const ptrResult = wasm.malloc(arrA.byteLength);

      try {
        wasm.getExports().abi_simd_vector_add(ptrA, ptrB, ptrResult, arrA.length);
        const result = new Float32Array(arrA.length);
        result.set(wasm.getFloat32Array(ptrResult, arrA.length));
        return result;
      } finally {
        wasm.free(ptrA);
        wasm.free(ptrB);
        wasm.free(ptrResult);
      }
    }

    // JavaScript fallback
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      result[i] = (a[i] as number) + (b[i] as number);
    }
    return result;
  }

  /**
   * Vector dot product.
   */
  static dotProduct(a: number[] | Float32Array, b: number[] | Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    // Try WASM implementation
    if (WasmModule.isLoaded()) {
      const wasm = WasmModule.getInstance();
      const arrA = a instanceof Float32Array ? a : new Float32Array(a);
      const arrB = b instanceof Float32Array ? b : new Float32Array(b);

      const ptrA = wasm.writeFloat32Array(arrA);
      const ptrB = wasm.writeFloat32Array(arrB);

      try {
        return wasm.getExports().abi_simd_vector_dot(ptrA, ptrB, arrA.length);
      } finally {
        wasm.free(ptrA);
        wasm.free(ptrB);
      }
    }

    // JavaScript fallback
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] as number) * (b[i] as number);
    }
    return sum;
  }

  /**
   * Vector L2 norm.
   */
  static l2Norm(v: number[] | Float32Array): number {
    // Try WASM implementation
    if (WasmModule.isLoaded()) {
      const wasm = WasmModule.getInstance();
      const arr = v instanceof Float32Array ? v : new Float32Array(v);
      const ptr = wasm.writeFloat32Array(arr);

      try {
        return wasm.getExports().abi_simd_vector_l2_norm(ptr, arr.length);
      } finally {
        wasm.free(ptr);
      }
    }

    // JavaScript fallback
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      const val = v[i] as number;
      sum += val * val;
    }
    return Math.sqrt(sum);
  }

  /**
   * Cosine similarity between two vectors.
   * Returns value between -1 and 1.
   */
  static cosineSimilarity(a: number[] | Float32Array, b: number[] | Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    // Try WASM implementation
    if (WasmModule.isLoaded()) {
      const wasm = WasmModule.getInstance();
      const arrA = a instanceof Float32Array ? a : new Float32Array(a);
      const arrB = b instanceof Float32Array ? b : new Float32Array(b);

      const ptrA = wasm.writeFloat32Array(arrA);
      const ptrB = wasm.writeFloat32Array(arrB);

      try {
        return wasm.getExports().abi_simd_cosine_similarity(ptrA, ptrB, arrA.length);
      } finally {
        wasm.free(ptrA);
        wasm.free(ptrB);
      }
    }

    // JavaScript fallback
    const dot = Simd.dotProduct(a, b);
    const normA = Simd.l2Norm(a);
    const normB = Simd.l2Norm(b);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dot / (normA * normB);
  }

  /**
   * Normalize a vector to unit length.
   */
  static normalize(v: number[] | Float32Array): Float32Array {
    const norm = Simd.l2Norm(v);
    if (norm === 0) {
      return v instanceof Float32Array ? new Float32Array(v) : new Float32Array(v);
    }

    const result = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
      result[i] = (v[i] as number) / norm;
    }
    return result;
  }

  /**
   * Euclidean distance between two vectors.
   */
  static euclideanDistance(a: number[] | Float32Array, b: number[] | Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same length');
    }

    const diff = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      diff[i] = (a[i] as number) - (b[i] as number);
    }
    return Simd.l2Norm(diff);
  }
}
