#!/usr/bin/env python3
"""
ABI Framework GPU Acceleration Example

Demonstrates GPU capabilities including device detection,
backend selection, and accelerated operations.
"""

import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import abi
from abi.gpu import (
    GpuContext,
    GpuConfig,
    GpuBackend,
    is_gpu_available,
    get_best_device,
    list_backends,
)


def main():
    print("=" * 60)
    print("ABI Framework GPU Acceleration Example")
    print("=" * 60)

    # Initialize
    abi.init()
    print(f"\nABI version: {abi.version()}")

    # Check GPU availability
    print("\n1. GPU Availability")
    print("-" * 40)

    available = is_gpu_available()
    print(f"   GPU available: {available}")

    backends = list_backends()
    print(f"   Available backends: {[b.name for b in backends]}")

    best_device = get_best_device()
    if best_device:
        print(f"   Best device: {best_device.name} ({best_device.backend.name})")
    else:
        print("   Best device: None (CPU fallback)")

    # List all devices
    print("\n2. Available Devices")
    print("-" * 40)

    devices = GpuContext.list_devices()
    for device in devices:
        mem_gb = device.memory_total / (1024 ** 3) if device.memory_total else 0
        flags = []
        if device.is_integrated:
            flags.append("integrated")
        if device.is_emulated:
            flags.append("emulated")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        print(f"   [{device.id}] {device.name} - {device.backend.name}, {mem_gb:.1f}GB{flag_str}")

    # Create GPU context with auto-detection
    print("\n3. GPU Context (Auto-detect)")
    print("-" * 40)

    ctx = GpuContext()
    print(f"   Is available: {ctx.is_available}")
    if ctx.device:
        print(f"   Device: {ctx.device.name}")
        print(f"   Backend: {ctx.device.backend.name}")

    # Create GPU context with specific backend
    print("\n4. GPU Context (Specific Backend)")
    print("-" * 40)

    # Try CUDA
    cuda_config = GpuConfig.cuda()
    print(f"   CUDA config: backend={cuda_config.backend.name}, device={cuda_config.device_index}")

    # Try Vulkan
    vulkan_config = GpuConfig.vulkan()
    print(f"   Vulkan config: backend={vulkan_config.backend.name}, device={vulkan_config.device_index}")

    # CPU only
    cpu_config = GpuConfig.cpu_only()
    print(f"   CPU config: backend={cpu_config.backend.name}")

    # Matrix operations
    print("\n5. Matrix Operations")
    print("-" * 40)

    # Matrix multiplication
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]

    print(f"   Matrix A: {a}")
    print(f"   Matrix B: {b}")

    result = ctx.matrix_multiply(a, b)
    print(f"   A x B = {result}")

    # Vector operations
    print("\n6. Vector Operations")
    print("-" * 40)

    v1 = [1.0, 2.0, 3.0, 4.0]
    v2 = [5.0, 6.0, 7.0, 8.0]

    print(f"   Vector 1: {v1}")
    print(f"   Vector 2: {v2}")

    # Vector add
    add_result = ctx.vector_add(v1, v2)
    print(f"   Add: {add_result}")

    # Dot product
    dot_result = ctx.vector_dot(v1, v2)
    print(f"   Dot product: {dot_result}")

    # Activation functions
    print("\n7. Activation Functions")
    print("-" * 40)

    x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    print(f"   Input: {x}")

    # Softmax
    softmax_result = ctx.softmax(x.copy())
    print(f"   Softmax: {[f'{v:.4f}' for v in softmax_result]}")

    # SiLU
    silu_result = ctx.silu(x.copy())
    print(f"   SiLU: {[f'{v:.4f}' for v in silu_result]}")

    # RMS Normalization
    print("\n8. RMS Normalization")
    print("-" * 40)

    x = [1.0, 2.0, 3.0, 4.0]
    weight = [1.0, 1.0, 1.0, 1.0]

    print(f"   Input: {x}")
    print(f"   Weight: {weight}")

    norm_result = ctx.rms_norm(x.copy(), weight)
    print(f"   RMS Norm: {[f'{v:.4f}' for v in norm_result]}")

    # Memory information
    print("\n9. Memory Information")
    print("-" * 40)

    total, free = ctx.memory_info()
    if total > 0:
        print(f"   Total memory: {total / (1024**3):.2f} GB")
        print(f"   Free memory: {free / (1024**3):.2f} GB")
    else:
        print("   Memory info not available (CPU mode)")

    # Statistics
    print("\n10. GPU Statistics")
    print("-" * 40)

    stats = ctx.stats
    print(f"   Total operations: {stats.total_ops}")
    print(f"   Total time: {stats.total_time_seconds:.6f}s")
    print(f"   Fallback operations: {stats.fallback_ops}")
    print(f"   GPU utilization: {stats.gpu_utilization:.1%}")

    # Reset statistics
    ctx.reset_stats()
    print("   Statistics reset")

    # Configuration options
    print("\n11. Configuration Options")
    print("-" * 40)

    config = GpuConfig(
        backend=GpuBackend.AUTO,
        device_index=0,
        memory_limit=4 * 1024 ** 3,  # 4GB
        async_enabled=True,
        cache_kernels=True,
        fallback_to_cpu=True,
    )

    print(f"   Backend: {config.backend.name}")
    print(f"   Device index: {config.device_index}")
    print(f"   Memory limit: {config.memory_limit / (1024**3):.1f} GB")
    print(f"   Async enabled: {config.async_enabled}")
    print(f"   Cache kernels: {config.cache_kernels}")
    print(f"   Fallback to CPU: {config.fallback_to_cpu}")

    # Synchronization
    print("\n12. Synchronization")
    print("-" * 40)

    ctx.synchronize()
    print("   GPU operations synchronized")

    # Cleanup
    print("\n13. Cleanup")
    print("-" * 40)
    abi.shutdown()
    print("   Framework shut down")

    print("\n" + "=" * 60)
    print("GPU Acceleration Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
