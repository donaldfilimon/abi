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

    # Cleanup
    abi.shutdown()
"""

from typing import List, Optional, Dict, Any
import ctypes
import os
import sys

__version__ = "0.3.0"
__all__ = [
    "init",
    "shutdown",
    "version",
    "is_initialized",
    "has_simd",
    "cosine_similarity",
    "vector_dot",
    "vector_add",
    "l2_norm",
    "VectorDatabase",
    "Agent",
    "Feature",
]


# Library loading
_lib = None
_lib_path = None


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
        return b"0.3.0-mock"

    def abi_init(self):
        return 0

    def abi_shutdown(self):
        pass

    def abi_is_initialized(self):
        return True

    def abi_simd_available(self):
        return False

    def abi_vector_dot(self, a, b, length):
        import math

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
        import os

        return os.cpu_count() or 1


# Public API


def init() -> None:
    """Initialize the ABI framework."""
    lib = _load_library()
    result = lib.abi_init()
    if result != 0:
        raise RuntimeError("Failed to initialize ABI framework")


def shutdown() -> None:
    """Shutdown the ABI framework."""
    lib = _load_library()
    lib.abi_shutdown()


def version() -> str:
    """Get the ABI framework version."""
    lib = _load_library()
    return lib.abi_version().decode("utf-8")


def is_initialized() -> bool:
    """Check if the framework is initialized."""
    lib = _load_library()
    return lib.abi_is_initialized()


def has_simd() -> bool:
    """Check if SIMD operations are available."""
    lib = _load_library()
    return lib.abi_simd_available()


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


class Feature:
    """Enumeration of ABI framework features."""

    AI = 0
    GPU = 1
    WEB = 2
    DATABASE = 3
    NETWORK = 4
    PROFILING = 5
    MONITORING = 6


class VectorDatabase:
    """
    Vector database interface for storing and querying vectors.

    Example:
        db = VectorDatabase()
        db.add([1.0, 2.0, 3.0], metadata={"id": "vec1"})
        results = db.query([1.1, 2.1, 3.1], top_k=5)
    """

    def __init__(self, name: str = "default", dimension: int = 0):
        """
        Initialize a vector database.

        Args:
            name: Database name
            dimension: Vector dimension (0 for auto-detect)
        """
        self.name = name
        self.dimension = dimension
        self._vectors: List[Dict[str, Any]] = []

    def add(
        self, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a vector to the database.

        Args:
            vector: Vector to add
            metadata: Optional metadata to associate

        Returns:
            Vector ID
        """
        if self.dimension == 0:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")

        vec_id = len(self._vectors)
        self._vectors.append({"id": vec_id, "vector": vector, "metadata": metadata or {}})
        return vec_id

    def query(
        self, vector: List[float], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query the database for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results to return

        Returns:
            List of results with scores
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Query vector dimension must be {self.dimension}")

        # Compute similarities
        results = []
        for entry in self._vectors:
            score = cosine_similarity(vector, entry["vector"])
            results.append(
                {"id": entry["id"], "score": score, "metadata": entry["metadata"]}
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """Get the number of vectors in the database."""
        return len(self._vectors)

    def clear(self) -> None:
        """Clear all vectors from the database."""
        self._vectors.clear()


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

    def process(self, message: str) -> str:
        """
        Process a message and return the agent's response.

        Args:
            message: User message

        Returns:
            Agent response
        """
        self._history.append({"role": "user", "content": message})

        # Placeholder response - in production this would call the LLM
        response = f"[Agent {self.name}] Received: {message}"

        self._history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self._history.copy()


# Auto-initialize on import if environment variable is set
if os.environ.get("ABI_AUTO_INIT", "").lower() in ("1", "true", "yes"):
    init()
