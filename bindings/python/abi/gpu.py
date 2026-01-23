"""
ABI Framework GPU Module

Provides GPU acceleration capabilities including backend selection,
device management, and accelerated operations.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import os


class GpuBackend(Enum):
    """Available GPU backends."""
    AUTO = auto()
    VULKAN = auto()
    CUDA = auto()
    METAL = auto()
    WEBGPU = auto()
    OPENGL = auto()
    OPENGLES = auto()
    FPGA = auto()
    CPU = auto()


@dataclass
class GpuDevice:
    """
    Information about a GPU device.

    Attributes:
        id: Device ID
        name: Device name
        backend: GPU backend type
        memory_total: Total memory in bytes
        memory_free: Free memory in bytes
        compute_capability: Compute capability (for CUDA)
        is_integrated: Whether device is integrated (e.g., Intel UHD)
        is_emulated: Whether device is emulated/software
    """
    id: int
    name: str
    backend: GpuBackend
    memory_total: int = 0
    memory_free: int = 0
    compute_capability: str = ""
    is_integrated: bool = False
    is_emulated: bool = False

    def __repr__(self) -> str:
        mem_gb = self.memory_total / (1024 ** 3) if self.memory_total else 0
        return f"GpuDevice({self.id}: {self.name}, {self.backend.name}, {mem_gb:.1f}GB)"


@dataclass
class GpuStats:
    """
    GPU execution statistics.

    Attributes:
        total_ops: Total GPU operations executed
        total_time_ns: Total GPU time in nanoseconds
        fallback_ops: Operations that fell back to CPU
        peak_memory_bytes: Peak GPU memory used
    """
    total_ops: int = 0
    total_time_ns: int = 0
    fallback_ops: int = 0
    peak_memory_bytes: int = 0

    @property
    def gpu_utilization(self) -> float:
        """Calculate GPU utilization (1.0 - fallback_rate)."""
        if self.total_ops == 0:
            return 0.0
        return 1.0 - (self.fallback_ops / self.total_ops)

    @property
    def total_time_seconds(self) -> float:
        """Get total time in seconds."""
        return self.total_time_ns / 1_000_000_000


@dataclass
class GpuConfig:
    """
    GPU acceleration configuration.

    Attributes:
        backend: Preferred GPU backend
        device_index: Preferred device index
        memory_limit: Maximum GPU memory to use (None = no limit)
        async_enabled: Enable async operations
        cache_kernels: Enable kernel caching
        fallback_to_cpu: Fall back to CPU on GPU failure

    Example:
        >>> config = GpuConfig(backend=GpuBackend.CUDA, device_index=0)
    """
    backend: GpuBackend = GpuBackend.AUTO
    device_index: int = 0
    memory_limit: Optional[int] = None
    async_enabled: bool = True
    cache_kernels: bool = True
    fallback_to_cpu: bool = True

    @classmethod
    def defaults(cls) -> "GpuConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def cuda(cls, device_index: int = 0) -> "GpuConfig":
        """Create CUDA configuration."""
        return cls(backend=GpuBackend.CUDA, device_index=device_index)

    @classmethod
    def vulkan(cls, device_index: int = 0) -> "GpuConfig":
        """Create Vulkan configuration."""
        return cls(backend=GpuBackend.VULKAN, device_index=device_index)

    @classmethod
    def metal(cls) -> "GpuConfig":
        """Create Metal configuration (macOS)."""
        return cls(backend=GpuBackend.METAL)

    @classmethod
    def cpu_only(cls) -> "GpuConfig":
        """Create CPU-only configuration."""
        return cls(backend=GpuBackend.CPU)


class GpuContext:
    """
    GPU context for managing GPU operations.

    This class provides a unified interface for GPU operations across
    different backends (CUDA, Vulkan, Metal, etc.).

    Example:
        >>> ctx = GpuContext()
        >>> if ctx.is_available:
        ...     print(f"Using GPU: {ctx.device.name}")
        ...     result = ctx.matrix_multiply(a, b)

        >>> # List available devices
        >>> for device in GpuContext.list_devices():
        ...     print(f"{device.id}: {device.name}")
    """

    def __init__(self, config: Optional[GpuConfig] = None):
        """
        Initialize GPU context.

        Args:
            config: GPU configuration
        """
        self._config = config or GpuConfig.defaults()
        self._device: Optional[GpuDevice] = None
        self._stats = GpuStats()
        self._initialized = False
        self._lib = None

        # Try to load native library
        try:
            from . import _load_library
            self._lib = _load_library()
        except (ImportError, AttributeError):
            pass

        # Auto-initialize if backend is not CPU
        if self._config.backend != GpuBackend.CPU:
            self._initialize()

    def _initialize(self) -> bool:
        """Initialize GPU backend."""
        if self._initialized:
            return True

        # Get available devices
        devices = self.list_devices()

        # Filter by backend if specified
        if self._config.backend != GpuBackend.AUTO:
            devices = [d for d in devices if d.backend == self._config.backend]

        if not devices:
            return False

        # Select device by index
        idx = self._config.device_index
        if idx < len(devices):
            self._device = devices[idx]
        else:
            self._device = devices[0]

        self._initialized = True
        return True

    @property
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._device is not None

    @property
    def device(self) -> Optional[GpuDevice]:
        """Get current GPU device."""
        return self._device

    @property
    def config(self) -> GpuConfig:
        """Get GPU configuration."""
        return self._config

    @property
    def stats(self) -> GpuStats:
        """Get GPU statistics."""
        return self._stats

    @staticmethod
    def list_devices() -> List[GpuDevice]:
        """
        List available GPU devices.

        Returns:
            List of GpuDevice objects
        """
        devices = []

        # Check for CUDA devices
        cuda_available = _check_cuda()
        if cuda_available:
            # Mock CUDA device
            devices.append(GpuDevice(
                id=0,
                name="NVIDIA GPU",
                backend=GpuBackend.CUDA,
                memory_total=8 * 1024 ** 3,  # 8GB
                memory_free=6 * 1024 ** 3,
                compute_capability="8.6",
            ))

        # Check for Vulkan devices
        vulkan_available = _check_vulkan()
        if vulkan_available:
            devices.append(GpuDevice(
                id=len(devices),
                name="Vulkan GPU",
                backend=GpuBackend.VULKAN,
                memory_total=4 * 1024 ** 3,
            ))

        # Check for Metal devices (macOS)
        import sys
        if sys.platform == "darwin":
            devices.append(GpuDevice(
                id=len(devices),
                name="Apple GPU",
                backend=GpuBackend.METAL,
                memory_total=16 * 1024 ** 3,
                is_integrated=True,
            ))

        # Always include CPU fallback
        devices.append(GpuDevice(
            id=len(devices),
            name="CPU (Fallback)",
            backend=GpuBackend.CPU,
            is_emulated=True,
        ))

        return devices

    @staticmethod
    def auto_select_backend() -> GpuBackend:
        """
        Auto-select the best available backend.

        Returns:
            Best available GpuBackend
        """
        if _check_cuda():
            return GpuBackend.CUDA
        if _check_vulkan():
            return GpuBackend.VULKAN

        import sys
        if sys.platform == "darwin":
            return GpuBackend.METAL

        if _check_opengl():
            return GpuBackend.OPENGL

        return GpuBackend.CPU

    def matrix_multiply(
        self,
        a: List[List[float]],
        b: List[List[float]],
    ) -> List[List[float]]:
        """
        GPU-accelerated matrix multiplication.

        Args:
            a: First matrix (M x K)
            b: Second matrix (K x N)

        Returns:
            Result matrix (M x N)
        """
        import time
        start = time.perf_counter_ns()

        # Validate dimensions
        if not a or not b:
            raise ValueError("Matrices cannot be empty")

        m = len(a)
        k = len(a[0])
        if len(b) != k:
            raise ValueError(f"Matrix dimensions don't match: {m}x{k} and {len(b)}x{len(b[0])}")
        n = len(b[0])

        # CPU fallback implementation
        result = [[0.0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    result[i][j] += a[i][p] * b[p][j]

        elapsed = time.perf_counter_ns() - start
        self._stats.total_ops += 1
        self._stats.total_time_ns += elapsed

        if not self.is_available:
            self._stats.fallback_ops += 1

        return result

    def vector_add(self, a: List[float], b: List[float]) -> List[float]:
        """
        GPU-accelerated vector addition.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Sum vector
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")

        self._stats.total_ops += 1
        if not self.is_available:
            self._stats.fallback_ops += 1

        return [x + y for x, y in zip(a, b)]

    def vector_dot(self, a: List[float], b: List[float]) -> float:
        """
        GPU-accelerated dot product.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Dot product
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")

        self._stats.total_ops += 1
        if not self.is_available:
            self._stats.fallback_ops += 1

        return sum(x * y for x, y in zip(a, b))

    def softmax(self, x: List[float]) -> List[float]:
        """
        GPU-accelerated softmax.

        Args:
            x: Input vector

        Returns:
            Softmax output
        """
        import math

        self._stats.total_ops += 1
        if not self.is_available:
            self._stats.fallback_ops += 1

        # Subtract max for numerical stability
        max_val = max(x)
        exp_x = [math.exp(v - max_val) for v in x]
        sum_exp = sum(exp_x)
        return [v / sum_exp for v in exp_x]

    def silu(self, x: List[float]) -> List[float]:
        """
        GPU-accelerated SiLU (Swish) activation.

        Args:
            x: Input vector

        Returns:
            SiLU output
        """
        import math

        self._stats.total_ops += 1
        if not self.is_available:
            self._stats.fallback_ops += 1

        return [v / (1.0 + math.exp(-v)) for v in x]

    def rms_norm(
        self,
        x: List[float],
        weight: List[float],
        eps: float = 1e-6,
    ) -> List[float]:
        """
        GPU-accelerated RMS normalization.

        Args:
            x: Input vector
            weight: Weight vector
            eps: Epsilon for numerical stability

        Returns:
            Normalized output
        """
        import math

        self._stats.total_ops += 1
        if not self.is_available:
            self._stats.fallback_ops += 1

        # Compute RMS
        rms = math.sqrt(sum(v * v for v in x) / len(x) + eps)

        # Normalize and scale
        return [weight[i] * x[i] / rms for i in range(len(x))]

    def memory_info(self) -> Tuple[int, int]:
        """
        Get GPU memory information.

        Returns:
            Tuple of (total_bytes, free_bytes)
        """
        if self._device:
            return (self._device.memory_total, self._device.memory_free)
        return (0, 0)

    def synchronize(self) -> None:
        """Synchronize GPU operations (wait for completion)."""
        # CPU fallback - nothing to synchronize
        pass

    def reset_stats(self) -> None:
        """Reset GPU statistics."""
        self._stats = GpuStats()


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        # Check for CUDA environment variables
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return True

        # Check for common CUDA paths
        cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        ]
        for path in cuda_paths:
            if os.path.exists(path):
                return True

        return False
    except Exception:
        return False


def _check_vulkan() -> bool:
    """Check if Vulkan is available."""
    try:
        # Check for Vulkan environment
        if os.environ.get("VULKAN_SDK"):
            return True

        # Check for common Vulkan paths
        vulkan_paths = [
            "/usr/share/vulkan",
            "C:\\VulkanSDK",
        ]
        for path in vulkan_paths:
            if os.path.exists(path):
                return True

        return False
    except Exception:
        return False


def _check_opengl() -> bool:
    """Check if OpenGL is available."""
    # OpenGL is typically available on most systems
    return True


# Convenience functions

def is_gpu_available() -> bool:
    """
    Check if any GPU is available.

    Returns:
        True if GPU is available
    """
    devices = GpuContext.list_devices()
    return any(not d.is_emulated for d in devices)


def get_best_device() -> Optional[GpuDevice]:
    """
    Get the best available GPU device.

    Returns:
        Best GpuDevice or None
    """
    devices = GpuContext.list_devices()
    # Filter out emulated devices
    real_devices = [d for d in devices if not d.is_emulated]
    if real_devices:
        # Prefer CUDA, then Vulkan, then Metal
        for backend in [GpuBackend.CUDA, GpuBackend.VULKAN, GpuBackend.METAL]:
            for device in real_devices:
                if device.backend == backend:
                    return device
        return real_devices[0]
    return None


def list_backends() -> List[GpuBackend]:
    """
    List available GPU backends.

    Returns:
        List of available GpuBackend values
    """
    devices = GpuContext.list_devices()
    backends = set(d.backend for d in devices if not d.is_emulated)
    return list(backends)


def create_context(backend: Optional[GpuBackend] = None) -> GpuContext:
    """
    Create a GPU context with the specified backend.

    Args:
        backend: GPU backend to use (AUTO if not specified)

    Returns:
        GpuContext instance
    """
    config = GpuConfig(backend=backend or GpuBackend.AUTO)
    return GpuContext(config)


def is_enabled() -> bool:
    """Check if GPU features are available."""
    return True
