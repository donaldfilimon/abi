"""
ABI Framework Python Bindings

A Python interface to the ABI high-performance AI and vector database framework.

Example usage:
    import abi

    # Initialize the framework
    abi.init()

    # Check framework version
    print(f"ABI version: {abi.version()}")

    # Use vector operations
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    similarity = abi.cosine_similarity(a, b)

    # Vector database
    db = abi.VectorDatabase(dimensions=3)
    db.add([1.0, 0.0, 0.0], metadata={"label": "x"})
    results = db.search([0.9, 0.1, 0.0], top_k=5)

    # LLM inference
    engine = abi.LlmEngine()
    engine.load_model("./model.gguf")
    response = engine.generate("Hello!")

    # GPU acceleration
    ctx = abi.GpuContext()
    if ctx.is_available:
        result = ctx.matrix_multiply(a, b)

    # Cleanup
    abi.shutdown()
"""

from typing import List, Optional, Dict, Any
import ctypes
import os
import sys

__version__ = "0.4.0"

# Core exports
__all__ = [
    # Version and info
    "__version__",
    "version",
    # Initialization
    "init",
    "shutdown",
    "is_initialized",
    # SIMD operations
    "has_simd",
    "cosine_similarity",
    "vector_dot",
    "vector_add",
    "l2_norm",
    # Classes
    "VectorDatabase",
    "Agent",
    "Feature",
    # Configuration
    "Config",
    "ConfigBuilder",
    "GpuConfig",
    "AiConfig",
    "LlmConfig",
    "DatabaseConfig",
    "NetworkConfig",
    "ObservabilityConfig",
    # LLM
    "LlmEngine",
    "LlmContext",
    "InferenceConfig",
    "InferenceStats",
    # Database
    "SearchResult",
    "DatabaseStats",
    "DatabaseContext",
    "create_database",
    "open_database",
    # GPU
    "GpuContext",
    "GpuDevice",
    "GpuBackend",
    "GpuStats",
    "is_gpu_available",
    "list_gpu_devices",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingReport",
    "train",
    # Submodules
    "config",
    "llm",
    "database",
    "gpu",
    "training",
]


# Library loading
_lib = None
_lib_path = None
_initialized = False


def _find_library() -> Optional[str]:
    """Find the ABI shared library."""
    possible_names = [
        "libabi.so",
        "libabi.dylib",
        "abi.dll",
    ]

    search_paths = [
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", "lib"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    # Add LD_LIBRARY_PATH entries
    if "LD_LIBRARY_PATH" in os.environ:
        search_paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))

    # Add Windows PATH
    if sys.platform == "win32" and "PATH" in os.environ:
        search_paths.extend(os.environ["PATH"].split(";"))

    for path in search_paths:
        for name in possible_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                return full_path

    return None


def _load_library():
    """Load the ABI shared library."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    _lib_path = _find_library()

    if _lib_path is None:
        # Return a mock library for development
        return _MockLibrary()

    try:
        _lib = ctypes.CDLL(_lib_path)
        _setup_library_functions(_lib)
        return _lib
    except OSError as e:
        raise ImportError(f"Failed to load ABI library from {_lib_path}: {e}")


def _setup_library_functions(lib):
    """Set up the library function signatures."""
    # Version
    lib.abi_version.restype = ctypes.c_char_p
    lib.abi_version.argtypes = []

    # Init/shutdown
    lib.abi_init.restype = ctypes.c_int
    lib.abi_init.argtypes = []
    lib.abi_shutdown.restype = None
    lib.abi_shutdown.argtypes = []
    lib.abi_is_initialized.restype = ctypes.c_bool
    lib.abi_is_initialized.argtypes = []

    # SIMD
    lib.abi_simd_available.restype = ctypes.c_bool
    lib.abi_simd_available.argtypes = []
    lib.abi_vector_dot.restype = ctypes.c_float
    lib.abi_vector_dot.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.abi_cosine_similarity.restype = ctypes.c_float
    lib.abi_cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.abi_l2_norm.restype = ctypes.c_float
    lib.abi_l2_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.abi_vector_add.restype = None
    lib.abi_vector_add.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]

    # Platform
    lib.abi_platform_os.restype = ctypes.c_uint32
    lib.abi_platform_os.argtypes = []
    lib.abi_platform_arch.restype = ctypes.c_uint32
    lib.abi_platform_arch.argtypes = []
    lib.abi_platform_max_threads.restype = ctypes.c_uint32
    lib.abi_platform_max_threads.argtypes = []


class _MockLibrary:
    """Mock library for development without native library."""

    def abi_version(self):
        return b"0.4.0-mock"

    def abi_init(self):
        return 0

    def abi_shutdown(self):
        pass

    def abi_is_initialized(self):
        return True

    def abi_simd_available(self):
        return False

    def abi_vector_dot(self, a, b, length):
        result = sum(a[i] * b[i] for i in range(length))
        return result

    def abi_cosine_similarity(self, a, b, length):
        import math

        dot = sum(a[i] * b[i] for i in range(length))
        norm_a = math.sqrt(sum(x * x for x in a[:length]))
        norm_b = math.sqrt(sum(x * x for x in b[:length]))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def abi_l2_norm(self, vec, length):
        import math

        return math.sqrt(sum(vec[i] ** 2 for i in range(length)))

    def abi_vector_add(self, a, b, result, length):
        for i in range(length):
            result[i] = a[i] + b[i]

    def abi_platform_os(self):
        return 0

    def abi_platform_arch(self):
        return 0

    def abi_platform_max_threads(self):
        return os.cpu_count() or 1


# =============================================================================
# Core API
# =============================================================================


def init(config: Optional["Config"] = None) -> None:
    """
    Initialize the ABI framework.

    Args:
        config: Optional configuration object

    Example:
        >>> import abi
        >>> abi.init()
        >>> # or with configuration
        >>> abi.init(abi.Config.defaults())
    """
    global _initialized
    lib = _load_library()
    result = lib.abi_init()
    if result != 0:
        raise RuntimeError("Failed to initialize ABI framework")
    _initialized = True


def shutdown() -> None:
    """Shutdown the ABI framework and release resources."""
    global _initialized
    lib = _load_library()
    lib.abi_shutdown()
    _initialized = False


def version() -> str:
    """
    Get the ABI framework version.

    Returns:
        Version string
    """
    lib = _load_library()
    return lib.abi_version().decode("utf-8")


def is_initialized() -> bool:
    """
    Check if the framework is initialized.

    Returns:
        True if initialized
    """
    return _initialized


def has_simd() -> bool:
    """
    Check if SIMD operations are available.

    Returns:
        True if SIMD is available
    """
    lib = _load_library()
    return lib.abi_simd_available()


# =============================================================================
# Vector Operations
# =============================================================================


def _to_float_array(vec: List[float]) -> ctypes.Array:
    """Convert Python list to C float array."""
    arr_type = ctypes.c_float * len(vec)
    return arr_type(*vec)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    lib = _load_library()

    if isinstance(lib, _MockLibrary):
        return lib.abi_cosine_similarity(a, b, len(a))

    arr_a = _to_float_array(a)
    arr_b = _to_float_array(b)
    return lib.abi_cosine_similarity(arr_a, arr_b, len(a))


def vector_dot(a: List[float], b: List[float]) -> float:
    """
    Compute dot product of two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Dot product value
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    lib = _load_library()

    if isinstance(lib, _MockLibrary):
        return lib.abi_vector_dot(a, b, len(a))

    arr_a = _to_float_array(a)
    arr_b = _to_float_array(b)
    return lib.abi_vector_dot(arr_a, arr_b, len(a))


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """
    Add two vectors element-wise.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Sum vector
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    lib = _load_library()

    result_type = ctypes.c_float * len(a)
    result = result_type()

    if isinstance(lib, _MockLibrary):
        lib.abi_vector_add(a, b, result, len(a))
    else:
        arr_a = _to_float_array(a)
        arr_b = _to_float_array(b)
        lib.abi_vector_add(arr_a, arr_b, result, len(a))

    return list(result)


def l2_norm(vec: List[float]) -> float:
    """
    Compute L2 norm of a vector.

    Args:
        vec: Input vector

    Returns:
        L2 norm value
    """
    lib = _load_library()

    if isinstance(lib, _MockLibrary):
        return lib.abi_l2_norm(vec, len(vec))

    arr = _to_float_array(vec)
    return lib.abi_l2_norm(arr, len(vec))


# =============================================================================
# Feature Enumeration
# =============================================================================


class Feature:
    """Enumeration of ABI framework features."""

    AI = 0
    GPU = 1
    WEB = 2
    DATABASE = 3
    NETWORK = 4
    PROFILING = 5
    MONITORING = 6
    LLM = 7
    EMBEDDINGS = 8
    AGENTS = 9
    TRAINING = 10


# =============================================================================
# Import Submodules
# =============================================================================

# Configuration
from .config import (
    Config,
    ConfigBuilder,
    GpuConfig,
    AiConfig,
    LlmConfig,
    DatabaseConfig,
    NetworkConfig,
    ObservabilityConfig,
    EmbeddingsConfig,
    AgentsConfig,
    TrainingConfig,
)

# LLM
from .llm import (
    LlmEngine,
    LlmContext,
    InferenceConfig,
    InferenceStats,
    ModelInfo,
    infer as llm_infer,
)

# Database
from .database import (
    VectorDatabase,
    SearchResult,
    DatabaseStats,
    DatabaseContext,
    BatchResult,
    create_database,
    open_database,
)

# GPU
from .gpu import (
    GpuContext,
    GpuDevice,
    GpuBackend,
    GpuStats,
    is_gpu_available,
    list_backends as list_gpu_backends,
)

# Convenience aliases
list_gpu_devices = GpuContext.list_devices


# =============================================================================
# Legacy Classes (backwards compatibility)
# =============================================================================


class Agent:
    """
    AI Agent interface for interacting with language models.

    Example:
        agent = Agent(name="my-agent")
        response = agent.process("Hello, how are you?")
        print(response)
    """

    def __init__(self, name: str = "default-agent"):
        """
        Initialize an AI agent.

        Args:
            name: Agent name
        """
        self.name = name
        self._history: List[Dict[str, str]] = []
        self._llm: Optional[LlmEngine] = None

    def process(self, message: str) -> str:
        """
        Process a message and return the agent's response.

        Args:
            message: User message

        Returns:
            Agent response
        """
        self._history.append({"role": "user", "content": message})

        # Use LLM if available, otherwise placeholder
        if self._llm and self._llm.is_loaded:
            response = self._llm.chat(self._history)
        else:
            response = f"[Agent {self.name}] Received: {message}"

        self._history.append({"role": "assistant", "content": response})
        return response

    def load_model(self, model_path: str) -> None:
        """
        Load an LLM model for the agent.

        Args:
            model_path: Path to GGUF model file
        """
        self._llm = LlmEngine()
        self._llm.load_model(model_path)

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history.copy()


# =============================================================================
# Module-level submodules
# =============================================================================

from . import config
from . import llm
from . import database
from . import gpu
from . import training

# Training exports
from .training import (
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingReport,
    train,
)


# Auto-initialize on import if environment variable is set
if os.environ.get("ABI_AUTO_INIT", "").lower() in ("1", "true", "yes"):
    init()
