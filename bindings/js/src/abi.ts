/**
 * Main ABI framework interface.
 */

import { AbiError, ErrorCode } from './error';
import { WasmModule, loadWasm } from './wasm';

/**
 * ABI initialization options.
 */
export interface AbiOptions {
  /** URL to the WASM module (default: '/abi.wasm') */
  wasmUrl?: string;
  /** Enable AI features */
  enableAi?: boolean;
  /** Enable GPU features (WebGPU) */
  enableGpu?: boolean;
  /** Enable database features */
  enableDatabase?: boolean;
}

/**
 * Main ABI framework class.
 *
 * @example
 * ```typescript
 * const abi = await Abi.init();
 * console.log('Version:', abi.version());
 * abi.shutdown();
 * ```
 */
export class Abi {
  private static instance: Abi | null = null;
  private wasm: WasmModule;
  private _initialized = false;

  private constructor(wasm: WasmModule) {
    this.wasm = wasm;
  }

  /**
   * Initialize the ABI framework.
   */
  static async init(options: AbiOptions = {}): Promise<Abi> {
    if (Abi.instance) {
      return Abi.instance;
    }

    // Load WASM module
    const wasm = await loadWasm(options.wasmUrl);
    wasm.init();

    Abi.instance = new Abi(wasm);
    Abi.instance._initialized = true;

    return Abi.instance;
  }

  /**
   * Get the ABI singleton instance.
   * Throws if not initialized.
   */
  static getInstance(): Abi {
    if (!Abi.instance || !Abi.instance._initialized) {
      throw new AbiError(ErrorCode.NotInitialized, 'ABI not initialized. Call Abi.init() first.');
    }
    return Abi.instance;
  }

  /**
   * Check if ABI is initialized.
   */
  static isInitialized(): boolean {
    return Abi.instance !== null && Abi.instance._initialized;
  }

  /**
   * Get the WASM module for advanced usage.
   */
  getWasm(): WasmModule {
    return this.wasm;
  }

  /**
   * Get the ABI version string.
   */
  version(): string {
    const ptr = this.wasm.getExports().abi_version();
    if (ptr === 0) {
      return 'unknown';
    }
    return this.wasm.readString(ptr);
  }

  /**
   * Check if a feature is enabled.
   */
  isFeatureEnabled(feature: string): boolean {
    // In WASM, we check based on build configuration
    switch (feature) {
      case 'simd':
        return this.wasm.getExports().abi_simd_available() !== 0;
      case 'gpu':
        return typeof navigator !== 'undefined' && 'gpu' in navigator;
      default:
        return false;
    }
  }

  /**
   * Shutdown the ABI framework.
   */
  shutdown(): void {
    if (!this._initialized) {
      return;
    }

    this.wasm.shutdown();
    this._initialized = false;
    Abi.instance = null;
  }
}
