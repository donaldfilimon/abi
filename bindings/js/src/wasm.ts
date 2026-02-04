/**
 * WASM module loader and interface.
 */

import { AbiError, ErrorCode } from './error';

/**
 * WASM module exports interface.
 */
export interface WasmExports {
  // Memory
  memory: WebAssembly.Memory;

  // Framework lifecycle
  abi_init(): number;
  abi_shutdown(): void;
  abi_version(): number;

  // SIMD operations
  abi_simd_available(): number;
  abi_simd_vector_add(a: number, b: number, result: number, len: number): void;
  abi_simd_vector_dot(a: number, b: number, len: number): number;
  abi_simd_vector_l2_norm(v: number, len: number): number;
  abi_simd_cosine_similarity(a: number, b: number, len: number): number;

  // Memory management
  malloc(size: number): number;
  free(ptr: number): void;
}

/**
 * WASM module wrapper with memory management.
 */
export class WasmModule {
  private static instance: WasmModule | null = null;
  private exports: WasmExports;
  private memory: WebAssembly.Memory;
  private initialized = false;

  private constructor(exports: WasmExports) {
    this.exports = exports;
    this.memory = exports.memory;
  }

  /**
   * Get the WASM module singleton.
   */
  static getInstance(): WasmModule {
    if (!WasmModule.instance) {
      throw new AbiError(ErrorCode.NotInitialized, 'WASM module not loaded');
    }
    return WasmModule.instance;
  }

  /**
   * Check if WASM module is loaded.
   */
  static isLoaded(): boolean {
    return WasmModule.instance !== null;
  }

  /**
   * Load the WASM module.
   */
  static async load(wasmUrl?: string): Promise<WasmModule> {
    if (WasmModule.instance) {
      return WasmModule.instance;
    }

    const url = wasmUrl || '/abi.wasm';

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new AbiError(ErrorCode.IoError, `Failed to fetch WASM: ${response.status}`);
      }

      const wasmBuffer = await response.arrayBuffer();
      const wasmModule = await WebAssembly.instantiate(wasmBuffer, {
        env: {
          // Environment imports if needed
        },
      });

      const exports = wasmModule.instance.exports as unknown as WasmExports;
      WasmModule.instance = new WasmModule(exports);
      return WasmModule.instance;
    } catch (error) {
      if (error instanceof AbiError) {
        throw error;
      }
      throw new AbiError(ErrorCode.WasmError, `Failed to load WASM: ${error}`);
    }
  }

  /**
   * Initialize the ABI framework.
   */
  init(): void {
    if (this.initialized) {
      return;
    }

    const result = this.exports.abi_init();
    if (result !== 0) {
      throw new AbiError(result as ErrorCode);
    }
    this.initialized = true;
  }

  /**
   * Shutdown the ABI framework.
   */
  shutdown(): void {
    if (!this.initialized) {
      return;
    }

    this.exports.abi_shutdown();
    this.initialized = false;
  }

  /**
   * Get the raw exports for advanced usage.
   */
  getExports(): WasmExports {
    return this.exports;
  }

  /**
   * Allocate memory in WASM heap.
   */
  malloc(size: number): number {
    return this.exports.malloc(size);
  }

  /**
   * Free memory in WASM heap.
   */
  free(ptr: number): void {
    this.exports.free(ptr);
  }

  /**
   * Get memory view for reading/writing.
   */
  getMemoryView(): DataView {
    return new DataView(this.memory.buffer);
  }

  /**
   * Get Float32Array view at offset.
   */
  getFloat32Array(ptr: number, length: number): Float32Array {
    return new Float32Array(this.memory.buffer, ptr, length);
  }

  /**
   * Write Float32Array to WASM memory.
   * Returns the pointer to allocated memory.
   */
  writeFloat32Array(array: Float32Array | number[]): number {
    const data = array instanceof Float32Array ? array : new Float32Array(array);
    const ptr = this.malloc(data.byteLength);
    const view = this.getFloat32Array(ptr, data.length);
    view.set(data);
    return ptr;
  }

  /**
   * Read string from WASM memory.
   */
  readString(ptr: number): string {
    const view = new Uint8Array(this.memory.buffer);
    let end = ptr;
    while (view[end] !== 0) {
      end++;
    }
    const bytes = view.slice(ptr, end);
    return new TextDecoder().decode(bytes);
  }
}

/**
 * Load the WASM module (convenience function).
 */
export async function loadWasm(wasmUrl?: string): Promise<WasmModule> {
  return WasmModule.load(wasmUrl);
}
