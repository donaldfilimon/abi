"""ABI Framework Python Bindings.

High-performance vector database and AI inference library.

Example:
    >>> from abi import Framework
    >>> fw = Framework()
    >>> print(fw.version())
    '0.4.0'
    >>> fw.shutdown()
"""

import ctypes
import os
import platform

__version__ = "0.4.0"
__all__ = [
    "Framework", "VectorDatabase", "GPU", "Agent",
    "AbiError", "SimdCaps", "VersionInfo",
    "SearchResult", "AgentResponse", "AgentStats",
]

# ============================================================================
# Error codes (matching C API - negative integers)
# ============================================================================

ABI_OK = 0
ABI_ERROR_INIT_FAILED = -1
ABI_ERROR_ALREADY_INITIALIZED = -2
ABI_ERROR_NOT_INITIALIZED = -3
ABI_ERROR_OUT_OF_MEMORY = -4
ABI_ERROR_INVALID_ARGUMENT = -5
ABI_ERROR_FEATURE_DISABLED = -6
ABI_ERROR_TIMEOUT = -7
ABI_ERROR_IO = -8
ABI_ERROR_GPU_UNAVAILABLE = -9
ABI_ERROR_DATABASE_ERROR = -10
ABI_ERROR_NETWORK_ERROR = -11
ABI_ERROR_AI_ERROR = -12
ABI_ERROR_UNKNOWN = -99

# GPU backend constants
ABI_GPU_BACKEND_AUTO = 0
ABI_GPU_BACKEND_CUDA = 1
ABI_GPU_BACKEND_VULKAN = 2
ABI_GPU_BACKEND_METAL = 3
ABI_GPU_BACKEND_WEBGPU = 4

# Agent backend constants
ABI_AGENT_BACKEND_ECHO = 0
ABI_AGENT_BACKEND_OPENAI = 1
ABI_AGENT_BACKEND_OLLAMA = 2
ABI_AGENT_BACKEND_HUGGINGFACE = 3
ABI_AGENT_BACKEND_LOCAL = 4

# Agent status constants
ABI_AGENT_STATUS_READY = 0
ABI_AGENT_STATUS_BUSY = 1
ABI_AGENT_STATUS_ERROR = 2

_ERROR_MESSAGES = {
    ABI_OK: "Success",
    ABI_ERROR_INIT_FAILED: "Initialization failed",
    ABI_ERROR_ALREADY_INITIALIZED: "Already initialized",
    ABI_ERROR_NOT_INITIALIZED: "Not initialized",
    ABI_ERROR_OUT_OF_MEMORY: "Out of memory",
    ABI_ERROR_INVALID_ARGUMENT: "Invalid argument",
    ABI_ERROR_FEATURE_DISABLED: "Feature disabled at compile time",
    ABI_ERROR_TIMEOUT: "Operation timed out",
    ABI_ERROR_IO: "I/O error",
    ABI_ERROR_GPU_UNAVAILABLE: "GPU not available",
    ABI_ERROR_DATABASE_ERROR: "Database error",
    ABI_ERROR_NETWORK_ERROR: "Network error",
    ABI_ERROR_AI_ERROR: "AI operation error",
    ABI_ERROR_UNKNOWN: "Unknown error",
}


# ============================================================================
# C structures (matching abi.h)
# ============================================================================

class _AbiOptions(ctypes.Structure):
    _fields_ = [
        ("enable_ai", ctypes.c_bool),
        ("enable_gpu", ctypes.c_bool),
        ("enable_database", ctypes.c_bool),
        ("enable_network", ctypes.c_bool),
        ("enable_web", ctypes.c_bool),
        ("enable_profiling", ctypes.c_bool),
    ]

class _AbiVersionInfo(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("patch", ctypes.c_int),
        ("full", ctypes.c_char_p),
    ]

class _AbiSimdCaps(ctypes.Structure):
    _fields_ = [
        ("sse", ctypes.c_bool),
        ("sse2", ctypes.c_bool),
        ("sse3", ctypes.c_bool),
        ("ssse3", ctypes.c_bool),
        ("sse4_1", ctypes.c_bool),
        ("sse4_2", ctypes.c_bool),
        ("avx", ctypes.c_bool),
        ("avx2", ctypes.c_bool),
        ("avx512f", ctypes.c_bool),
        ("neon", ctypes.c_bool),
    ]

class _AbiDatabaseConfig(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dimension", ctypes.c_size_t),
        ("initial_capacity", ctypes.c_size_t),
    ]

class _AbiGpuConfig(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("device_index", ctypes.c_int),
        ("enable_profiling", ctypes.c_bool),
    ]

class _AbiSearchResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("score", ctypes.c_float),
    ]

class _AbiAgentConfig(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("backend", ctypes.c_int),
        ("model", ctypes.c_char_p),
        ("system_prompt", ctypes.c_char_p),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("max_tokens", ctypes.c_uint32),
        ("enable_history", ctypes.c_bool),
    ]

class _AbiAgentResponse(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("length", ctypes.c_size_t),
        ("tokens_used", ctypes.c_uint64),
    ]

class _AbiAgentStats(ctypes.Structure):
    _fields_ = [
        ("history_length", ctypes.c_size_t),
        ("user_messages", ctypes.c_size_t),
        ("assistant_messages", ctypes.c_size_t),
        ("total_characters", ctypes.c_size_t),
        ("total_tokens_used", ctypes.c_uint64),
    ]


# ============================================================================
# Error handling
# ============================================================================

class AbiError(Exception):
    """Exception raised for ABI C API errors."""

    def __init__(self, code, context=""):
        self.code = code
        self.message = _ERROR_MESSAGES.get(code, "Unknown error")
        msg = f"{self.message} (code {code})"
        if context:
            msg = f"{context}: {msg}"
        super().__init__(msg)


def _check(code, context=""):
    """Check a C API return code and raise AbiError on failure."""
    if code != ABI_OK:
        raise AbiError(code, context)


# ============================================================================
# Library loader
# ============================================================================

def _load_lib(lib_path=None):
    """Load the ABI shared library."""
    if lib_path and os.path.exists(lib_path):
        return ctypes.CDLL(lib_path)

    system = platform.system()
    if system == "Darwin":
        lib_name = "libabi.dylib"
    elif system == "Windows":
        lib_name = "abi.dll"
    else:
        lib_name = "libabi.so"

    search_paths = [
        os.path.join(os.getcwd(), "zig-out", "lib", lib_name),
        os.path.join(os.path.dirname(__file__), "..", "..", "zig-out", "lib", lib_name),
        lib_name,
    ]

    for path in search_paths:
        if os.path.exists(path):
            return ctypes.CDLL(path)

    raise FileNotFoundError(f"Could not find {lib_name}. Build with: zig build lib")


def _setup_signatures(lib):
    """Set up ctypes function signatures for the C API."""
    # Error handling
    lib.abi_error_string.restype = ctypes.c_char_p
    lib.abi_error_string.argtypes = [ctypes.c_int]

    # Framework lifecycle
    lib.abi_init.restype = ctypes.c_int
    lib.abi_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    lib.abi_init_with_options.restype = ctypes.c_int
    lib.abi_init_with_options.argtypes = [
        ctypes.POINTER(_AbiOptions), ctypes.POINTER(ctypes.c_void_p)
    ]

    lib.abi_shutdown.restype = None
    lib.abi_shutdown.argtypes = [ctypes.c_void_p]

    lib.abi_version.restype = ctypes.c_char_p
    lib.abi_version.argtypes = []

    lib.abi_version_info.restype = None
    lib.abi_version_info.argtypes = [ctypes.POINTER(_AbiVersionInfo)]

    lib.abi_is_feature_enabled.restype = ctypes.c_bool
    lib.abi_is_feature_enabled.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # SIMD
    lib.abi_simd_get_caps.restype = None
    lib.abi_simd_get_caps.argtypes = [ctypes.POINTER(_AbiSimdCaps)]

    lib.abi_simd_available.restype = ctypes.c_bool
    lib.abi_simd_available.argtypes = []

    lib.abi_simd_vector_add.restype = None
    lib.abi_simd_vector_add.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]

    lib.abi_simd_vector_dot.restype = ctypes.c_float
    lib.abi_simd_vector_dot.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]

    lib.abi_simd_vector_l2_norm.restype = ctypes.c_float
    lib.abi_simd_vector_l2_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

    lib.abi_simd_cosine_similarity.restype = ctypes.c_float
    lib.abi_simd_cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]

    # Database
    lib.abi_database_create.restype = ctypes.c_int
    lib.abi_database_create.argtypes = [
        ctypes.POINTER(_AbiDatabaseConfig), ctypes.POINTER(ctypes.c_void_p),
    ]

    lib.abi_database_close.restype = None
    lib.abi_database_close.argtypes = [ctypes.c_void_p]

    lib.abi_database_insert.restype = ctypes.c_int
    lib.abi_database_insert.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p,
    ]

    lib.abi_database_search.restype = ctypes.c_int
    lib.abi_database_search.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.POINTER(_AbiSearchResult), ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.abi_database_delete.restype = ctypes.c_int
    lib.abi_database_delete.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

    lib.abi_database_count.restype = ctypes.c_int
    lib.abi_database_count.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    # GPU
    lib.abi_gpu_init.restype = ctypes.c_int
    lib.abi_gpu_init.argtypes = [ctypes.POINTER(_AbiGpuConfig), ctypes.POINTER(ctypes.c_void_p)]

    lib.abi_gpu_shutdown.restype = None
    lib.abi_gpu_shutdown.argtypes = [ctypes.c_void_p]

    lib.abi_gpu_is_available.restype = ctypes.c_bool
    lib.abi_gpu_is_available.argtypes = []

    lib.abi_gpu_backend_name.restype = ctypes.c_char_p
    lib.abi_gpu_backend_name.argtypes = [ctypes.c_void_p]

    # Agent
    lib.abi_agent_create.restype = ctypes.c_int
    lib.abi_agent_create.argtypes = [
        ctypes.POINTER(_AbiAgentConfig), ctypes.POINTER(ctypes.c_void_p),
    ]

    lib.abi_agent_destroy.restype = None
    lib.abi_agent_destroy.argtypes = [ctypes.c_void_p]

    lib.abi_agent_send.restype = ctypes.c_int
    lib.abi_agent_send.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_AbiAgentResponse),
    ]

    lib.abi_agent_get_status.restype = ctypes.c_int
    lib.abi_agent_get_status.argtypes = [ctypes.c_void_p]

    lib.abi_agent_get_stats.restype = ctypes.c_int
    lib.abi_agent_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AbiAgentStats)]

    lib.abi_agent_clear_history.restype = ctypes.c_int
    lib.abi_agent_clear_history.argtypes = [ctypes.c_void_p]

    lib.abi_agent_set_temperature.restype = ctypes.c_int
    lib.abi_agent_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float]

    lib.abi_agent_set_max_tokens.restype = ctypes.c_int
    lib.abi_agent_set_max_tokens.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    lib.abi_agent_get_name.restype = ctypes.c_char_p
    lib.abi_agent_get_name.argtypes = [ctypes.c_void_p]

    # Memory management
    lib.abi_free_string.restype = None
    lib.abi_free_string.argtypes = [ctypes.c_void_p]

    lib.abi_free_results.restype = None
    lib.abi_free_results.argtypes = [ctypes.POINTER(_AbiSearchResult), ctypes.c_size_t]

    return lib


# ============================================================================
# Data classes
# ============================================================================

class VersionInfo:
    """Detailed version information."""

    def __init__(self, major, minor, patch, full):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.full = full

    def __repr__(self):
        return f"VersionInfo({self.major}.{self.minor}.{self.patch})"


class SimdCaps:
    """CPU SIMD capabilities."""

    def __init__(self, caps):
        self.sse = caps.sse
        self.sse2 = caps.sse2
        self.sse3 = caps.sse3
        self.ssse3 = caps.ssse3
        self.sse4_1 = caps.sse4_1
        self.sse4_2 = caps.sse4_2
        self.avx = caps.avx
        self.avx2 = caps.avx2
        self.avx512f = caps.avx512f
        self.neon = caps.neon


class SearchResult:
    """Vector search result."""

    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __repr__(self):
        return f"SearchResult(id={self.id}, score={self.score:.4f})"


class AgentResponse:
    """Agent response from a send operation."""

    def __init__(self, text, length, tokens_used):
        self.text = text
        self.length = length
        self.tokens_used = tokens_used

    def __repr__(self):
        return f"AgentResponse(length={self.length}, tokens={self.tokens_used})"


class AgentStats:
    """Agent conversation statistics."""

    def __init__(self, history_length, user_messages, assistant_messages,
                 total_characters, total_tokens_used):
        self.history_length = history_length
        self.user_messages = user_messages
        self.assistant_messages = assistant_messages
        self.total_characters = total_characters
        self.total_tokens_used = total_tokens_used


# ============================================================================
# Framework
# ============================================================================

class Framework:
    """ABI framework instance.

    Example:
        >>> fw = Framework()
        >>> print(fw.version())
        '0.4.0'
        >>> fw.shutdown()
    """

    def __init__(self, lib_path=None, enable_ai=True, enable_gpu=True,
                 enable_database=True, enable_network=True, enable_web=True,
                 enable_profiling=True):
        self._lib = _setup_signatures(_load_lib(lib_path))
        self._handle = ctypes.c_void_p()

        all_defaults = all([enable_ai, enable_gpu, enable_database,
                           enable_network, enable_web, enable_profiling])

        if all_defaults:
            code = self._lib.abi_init(ctypes.byref(self._handle))
        else:
            opts = _AbiOptions(
                enable_ai=enable_ai,
                enable_gpu=enable_gpu,
                enable_database=enable_database,
                enable_network=enable_network,
                enable_web=enable_web,
                enable_profiling=enable_profiling,
            )
            code = self._lib.abi_init_with_options(
                ctypes.byref(opts), ctypes.byref(self._handle)
            )

        _check(code, "Framework init")

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def shutdown(self):
        """Shutdown the framework and release all resources."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.abi_shutdown(self._handle)
            self._handle = None

    def version(self):
        """Return the framework version string."""
        return self._lib.abi_version().decode("utf-8")

    def version_info(self):
        """Return detailed version information."""
        info = _AbiVersionInfo()
        self._lib.abi_version_info(ctypes.byref(info))
        return VersionInfo(info.major, info.minor, info.patch,
                          info.full.decode("utf-8") if info.full else "")

    def is_feature_enabled(self, feature):
        """Check if a feature is enabled.

        Args:
            feature: Feature name ("ai", "gpu", "database", "network", "web", "profiling").
        """
        return self._lib.abi_is_feature_enabled(
            self._handle, feature.encode("utf-8")
        )

    def error_string(self, code):
        """Return human-readable error message for an error code."""
        return self._lib.abi_error_string(code).decode("utf-8")

    # SIMD operations

    def simd_available(self):
        """Check if any SIMD instruction set is available."""
        return self._lib.abi_simd_available()

    def simd_get_caps(self):
        """Query CPU SIMD capabilities."""
        caps = _AbiSimdCaps()
        self._lib.abi_simd_get_caps(ctypes.byref(caps))
        return SimdCaps(caps)

    def simd_vector_add(self, a, b):
        """Element-wise vector addition: result[i] = a[i] + b[i]."""
        n = len(a)
        ca = (ctypes.c_float * n)(*a)
        cb = (ctypes.c_float * n)(*b)
        cr = (ctypes.c_float * n)()
        self._lib.abi_simd_vector_add(ca, cb, cr, n)
        return list(cr)

    def simd_vector_dot(self, a, b):
        """Compute the dot product of two vectors."""
        n = len(a)
        ca = (ctypes.c_float * n)(*a)
        cb = (ctypes.c_float * n)(*b)
        return self._lib.abi_simd_vector_dot(ca, cb, n)

    def simd_vector_l2_norm(self, v):
        """Compute the L2 norm of a vector."""
        n = len(v)
        cv = (ctypes.c_float * n)(*v)
        return self._lib.abi_simd_vector_l2_norm(cv, n)

    def simd_cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        n = len(a)
        ca = (ctypes.c_float * n)(*a)
        cb = (ctypes.c_float * n)(*b)
        return self._lib.abi_simd_cosine_similarity(ca, cb, n)

    # Database operations

    def create_database(self, name="default", dimension=384, initial_capacity=1000):
        """Create a new vector database.

        Args:
            name: Database name.
            dimension: Vector dimension.
            initial_capacity: Initial capacity hint.

        Returns:
            VectorDatabase instance.
        """
        return VectorDatabase(self._lib, name, dimension, initial_capacity)

    # GPU operations

    def gpu_init(self, backend=ABI_GPU_BACKEND_AUTO, device_index=0,
                 enable_profiling=False):
        """Initialize a GPU context.

        Args:
            backend: GPU backend (ABI_GPU_BACKEND_* constant).
            device_index: Device index.
            enable_profiling: Enable profiling.

        Returns:
            GPU instance.
        """
        return GPU(self._lib, backend, device_index, enable_profiling)

    def gpu_is_available(self):
        """Check if any GPU backend is available."""
        return self._lib.abi_gpu_is_available()

    # Agent operations

    def create_agent(self, name="agent", backend=ABI_AGENT_BACKEND_ECHO,
                     model="gpt-4", system_prompt=None, temperature=0.7,
                     top_p=0.9, max_tokens=1024, enable_history=True):
        """Create a new AI agent.

        Args:
            name: Agent name.
            backend: Backend type (ABI_AGENT_BACKEND_* constant).
            model: Model name.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            top_p: Top-p sampling (0.0-1.0).
            max_tokens: Maximum generation tokens.
            enable_history: Enable conversation history.

        Returns:
            Agent instance.
        """
        return Agent(self._lib, name, backend, model, system_prompt,
                     temperature, top_p, max_tokens, enable_history)


# ============================================================================
# VectorDatabase
# ============================================================================

class VectorDatabase:
    """Vector database instance."""

    def __init__(self, lib, name, dimension, initial_capacity):
        self._lib = lib
        self._handle = ctypes.c_void_p()
        config = _AbiDatabaseConfig(
            name=name.encode("utf-8"),
            dimension=dimension,
            initial_capacity=initial_capacity,
        )
        code = self._lib.abi_database_create(
            ctypes.byref(config), ctypes.byref(self._handle)
        )
        _check(code, "Database create")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the database and release resources."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.abi_database_close(self._handle)
            self._handle = None

    def insert(self, id, vector, metadata=None):
        """Insert a vector into the database.

        Args:
            id: Unique vector ID.
            vector: List of floats.
            metadata: Optional metadata string.
        """
        n = len(vector)
        c_vector = (ctypes.c_float * n)(*vector)
        c_meta = metadata.encode("utf-8") if metadata else None
        code = self._lib.abi_database_insert(
            self._handle, ctypes.c_uint64(id), c_vector, n, c_meta
        )
        _check(code, "Database insert")

    def search(self, query, k=10):
        """Search for similar vectors.

        Args:
            query: Query vector (list of floats).
            k: Maximum number of results.

        Returns:
            List of SearchResult.
        """
        n = len(query)
        c_query = (ctypes.c_float * n)(*query)
        results = (_AbiSearchResult * k)()
        count = ctypes.c_size_t(0)
        code = self._lib.abi_database_search(
            self._handle, c_query, n, k, results, ctypes.byref(count)
        )
        _check(code, "Database search")
        return [SearchResult(results[i].id, results[i].score)
                for i in range(count.value)]

    def delete(self, id):
        """Delete a vector by ID."""
        code = self._lib.abi_database_delete(self._handle, ctypes.c_uint64(id))
        _check(code, "Database delete")

    def count(self):
        """Return the number of vectors in the database."""
        c = ctypes.c_size_t(0)
        code = self._lib.abi_database_count(self._handle, ctypes.byref(c))
        _check(code, "Database count")
        return c.value


# ============================================================================
# GPU
# ============================================================================

class GPU:
    """GPU context instance."""

    def __init__(self, lib, backend, device_index, enable_profiling):
        self._lib = lib
        self._handle = ctypes.c_void_p()
        config = _AbiGpuConfig(
            backend=backend,
            device_index=device_index,
            enable_profiling=enable_profiling,
        )
        code = self._lib.abi_gpu_init(
            ctypes.byref(config), ctypes.byref(self._handle)
        )
        _check(code, "GPU init")

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def shutdown(self):
        """Shutdown GPU context and release resources."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.abi_gpu_shutdown(self._handle)
            self._handle = None

    def backend_name(self):
        """Return the active GPU backend name."""
        return self._lib.abi_gpu_backend_name(self._handle).decode("utf-8")


# ============================================================================
# Agent
# ============================================================================

class Agent:
    """AI agent instance."""

    def __init__(self, lib, name, backend, model, system_prompt,
                 temperature, top_p, max_tokens, enable_history):
        self._lib = lib
        self._handle = ctypes.c_void_p()
        config = _AbiAgentConfig(
            name=name.encode("utf-8"),
            backend=backend,
            model=model.encode("utf-8"),
            system_prompt=system_prompt.encode("utf-8") if system_prompt else None,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_history=enable_history,
        )
        code = self._lib.abi_agent_create(
            ctypes.byref(config), ctypes.byref(self._handle)
        )
        _check(code, "Agent create")

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()

    def destroy(self):
        """Destroy the agent and release resources."""
        if hasattr(self, '_handle') and self._handle:
            self._lib.abi_agent_destroy(self._handle)
            self._handle = None

    def send(self, message):
        """Send a message and get a response.

        Args:
            message: User message string.

        Returns:
            AgentResponse with text, length, and tokens_used.
        """
        resp = _AbiAgentResponse()
        code = self._lib.abi_agent_send(
            self._handle, message.encode("utf-8"), ctypes.byref(resp)
        )
        _check(code, "Agent send")
        text = resp.text.decode("utf-8") if resp.text else ""
        return AgentResponse(text, resp.length, resp.tokens_used)

    def status(self):
        """Return the current agent status code."""
        return self._lib.abi_agent_get_status(self._handle)

    def stats(self):
        """Return agent conversation statistics."""
        s = _AbiAgentStats()
        code = self._lib.abi_agent_get_stats(self._handle, ctypes.byref(s))
        _check(code, "Agent get stats")
        return AgentStats(s.history_length, s.user_messages,
                         s.assistant_messages, s.total_characters,
                         s.total_tokens_used)

    def clear_history(self):
        """Clear the conversation history."""
        code = self._lib.abi_agent_clear_history(self._handle)
        _check(code, "Agent clear history")

    def set_temperature(self, temperature):
        """Set the sampling temperature (0.0-2.0)."""
        code = self._lib.abi_agent_set_temperature(self._handle, temperature)
        _check(code, "Agent set temperature")

    def set_max_tokens(self, max_tokens):
        """Set the maximum generation tokens."""
        code = self._lib.abi_agent_set_max_tokens(self._handle, max_tokens)
        _check(code, "Agent set max tokens")

    def name(self):
        """Return the agent's name."""
        return self._lib.abi_agent_get_name(self._handle).decode("utf-8")
