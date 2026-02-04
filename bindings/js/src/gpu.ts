/**
 * GPU acceleration via WebGPU.
 */

import { AbiError, ErrorCode } from './error';

/**
 * GPU backend type.
 */
export enum Backend {
  /** Auto-detect best backend */
  Auto = 'auto',
  /** WebGPU (browser) */
  WebGpu = 'webgpu',
  /** CPU fallback */
  Cpu = 'cpu',
}

/**
 * GPU configuration.
 */
export interface GpuConfig {
  /** Backend to use */
  backend?: Backend;
  /** Power preference */
  powerPreference?: 'high-performance' | 'low-power';
}

/**
 * GPU context for WebGPU acceleration.
 *
 * @example
 * ```typescript
 * if (await Gpu.isAvailable()) {
 *   const gpu = await Gpu.init();
 *   console.log('Backend:', gpu.backendName);
 * }
 * ```
 */
export class Gpu {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private _backendName: string;

  private constructor(device: GPUDevice | null, adapter: GPUAdapter | null, backendName: string) {
    this.device = device;
    this.adapter = adapter;
    this._backendName = backendName;
  }

  /**
   * Check if WebGPU is available.
   */
  static async isAvailable(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  /**
   * Initialize GPU context.
   */
  static async init(config: GpuConfig = {}): Promise<Gpu> {
    const backend = config.backend ?? Backend.Auto;

    if (backend === Backend.Cpu) {
      return new Gpu(null, null, 'cpu');
    }

    // Try WebGPU
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: config.powerPreference,
        });

        if (adapter) {
          const device = await adapter.requestDevice();
          return new Gpu(device, adapter, 'webgpu');
        }
      } catch (error) {
        if (backend === Backend.WebGpu) {
          throw new AbiError(ErrorCode.GpuError, `Failed to initialize WebGPU: ${error}`);
        }
      }
    }

    // Fallback to CPU
    if (backend === Backend.Auto) {
      return new Gpu(null, null, 'cpu');
    }

    throw new AbiError(ErrorCode.GpuError, 'No GPU backend available');
  }

  /**
   * Get the backend name.
   */
  get backendName(): string {
    return this._backendName;
  }

  /**
   * Check if GPU is available (not CPU fallback).
   */
  get isGpuBackend(): boolean {
    return this.device !== null;
  }

  /**
   * Get adapter info.
   */
  async getAdapterInfo(): Promise<{
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  } | null> {
    if (!this.adapter) {
      return null;
    }

    const info = await this.adapter.requestAdapterInfo();
    return {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
  }

  /**
   * Get device limits.
   */
  getDeviceLimits(): GPUSupportedLimits | null {
    return this.device?.limits ?? null;
  }

  /**
   * Get the raw WebGPU device for advanced usage.
   */
  getDevice(): GPUDevice | null {
    return this.device;
  }

  /**
   * Destroy the GPU context.
   */
  destroy(): void {
    if (this.device) {
      this.device.destroy();
      this.device = null;
      this.adapter = null;
    }
  }
}
