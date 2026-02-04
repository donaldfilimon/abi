"""ABI Framework Python Bindings.

High-performance vector database and AI inference library.

Example:
    >>> from abi import ABI
    >>> abi = ABI()
    >>> print(abi.version())
    '0.4.0'
"""

import ctypes
import os
import platform

__version__ = "0.4.0"
__all__ = ["ABI", "VectorDatabase", "AbiError", "AbiStatus", "GpuBackend", "__version__"]

# Define C types
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64
c_float = ctypes.c_float
c_size_t = ctypes.c_size_t
c_int = ctypes.c_int

# Enum for AbiStatus
class AbiStatus(ctypes.c_int):
    SUCCESS = 0
    ERROR_UNKNOWN = 1
    ERROR_INVALID_ARGUMENT = 2
    ERROR_OUT_OF_MEMORY = 3
    ERROR_INITIALIZATION_FAILED = 4
    ERROR_NOT_INITIALIZED = 5

class AbiError(Exception):
    def __init__(self, status, message="ABI Error"):
        self.status = status
        self.message = message
        super().__init__(f"{message}: {status}")


class GpuBackend:
    """GPU backend selection constants.

    These correspond to the backend values in the C API (abi_gpu_config_t):
        0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu, 5=stdgpu, 6=cpu
    """
    AUTO = 'auto'
    CUDA = 'cuda'
    VULKAN = 'vulkan'
    METAL = 'metal'
    WEBGPU = 'webgpu'
    STDGPU = 'stdgpu'
    CPU = 'cpu'

    # Valid backend names
    VALID_BACKENDS = [AUTO, CUDA, VULKAN, METAL, WEBGPU, STDGPU, CPU]

    # Mapping from string to C API int value
    _TO_INT = {
        AUTO: 0,
        CUDA: 1,
        VULKAN: 2,
        METAL: 3,
        WEBGPU: 4,
        STDGPU: 5,
        CPU: 6,
    }

    @classmethod
    def validate(cls, backend):
        """Validate a backend name and return it normalized.

        Args:
            backend: Backend name string

        Returns:
            Normalized backend name

        Raises:
            ValueError: If backend is not a valid backend name
        """
        if backend not in cls.VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. "
                f"Must be one of: {cls.VALID_BACKENDS}"
            )
        return backend

    @classmethod
    def to_int(cls, backend):
        """Convert backend name to C API integer value.

        Args:
            backend: Backend name string

        Returns:
            Integer value for C API
        """
        return cls._TO_INT.get(backend, 0)

class ABI:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Try to find the library in standard locations
            system = platform.system()
            if system == "Darwin":
                lib_name = "libabi.dylib"
            elif system == "Windows":
                lib_name = "abi.dll"
            else:
                lib_name = "libabi.so"
            
            # Check zig-out/lib relative to current script or cwd
            possible_paths = [
                os.path.join(os.getcwd(), "zig-out", "lib", lib_name),
                os.path.join(os.path.dirname(__file__), "..", "..", "zig-out", "lib", lib_name),
                lib_name
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
            
            if lib_path is None:
                raise FileNotFoundError(f"Could not find {lib_name}")

        self.lib = ctypes.CDLL(lib_path)
        
        # Setup function signatures
        self.lib.abi_init.restype = c_void_p
        self.lib.abi_init.argtypes = []
        
        self.lib.abi_shutdown.restype = None
        self.lib.abi_shutdown.argtypes = [c_void_p]
        
        self.lib.abi_version.restype = c_char_p
        self.lib.abi_version.argtypes = []
        
        self.lib.abi_db_create.restype = c_int
        self.lib.abi_db_create.argtypes = [c_void_p, c_uint32, ctypes.POINTER(c_void_p)]
        
        self.lib.abi_db_insert.restype = c_int
        self.lib.abi_db_insert.argtypes = [c_void_p, c_uint64, ctypes.POINTER(c_float), c_size_t]
        
        self.lib.abi_db_search.restype = c_int
        self.lib.abi_db_search.argtypes = [
            c_void_p, 
            ctypes.POINTER(c_float), 
            c_size_t, 
            c_uint32, 
            ctypes.POINTER(c_uint64), 
            ctypes.POINTER(c_float)
        ]
        
        self.lib.abi_db_destroy.restype = None
        self.lib.abi_db_destroy.argtypes = [c_void_p]

        # Initialize framework
        self.handle = self.lib.abi_init()
        if not self.handle:
            raise AbiError(AbiStatus.ERROR_INITIALIZATION_FAILED, "Failed to initialize ABI framework")

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.abi_shutdown(self.handle)
            self.handle = None

    def version(self):
        return self.lib.abi_version().decode('utf-8')

    def create_db(self, dimension, backend='auto'):
        """Create a new vector database.

        Args:
            dimension: Vector dimension (number of components)
            backend: GPU backend to use for acceleration.
                Options: 'auto', 'cuda', 'vulkan', 'metal', 'webgpu', 'stdgpu', 'cpu'
                Default is 'auto' which selects the best available backend.

        Returns:
            VectorDatabase instance

        Raises:
            ValueError: If backend is not a valid backend name
        """
        return VectorDatabase(self, dimension, backend=backend)

class VectorDatabase:
    """Vector database with optional GPU acceleration.

    Attributes:
        dimension: Vector dimension (number of components per vector)
        backend: GPU backend name ('auto', 'cuda', 'vulkan', 'metal', 'webgpu', 'stdgpu', 'cpu')
    """

    def __init__(self, abi_instance, dimension, backend='auto'):
        """Create a new vector database.

        Args:
            abi_instance: Parent ABI instance
            dimension: Vector dimension (number of components)
            backend: GPU backend to use for acceleration.
                Options: 'auto', 'cuda', 'vulkan', 'metal', 'webgpu', 'stdgpu', 'cpu'
                Default is 'auto' which selects the best available backend.

        Raises:
            ValueError: If backend is not a valid backend name
            AbiError: If database creation fails
        """
        self.abi = abi_instance
        self.handle = c_void_p()
        self.dimension = dimension

        # Validate and store backend
        self.backend = GpuBackend.validate(backend)
        self._backend_int = GpuBackend.to_int(backend)

        # Create the database
        # Note: The current C API (abi_db_create) doesn't support backend selection yet.
        # When the C API is extended to support GPU-accelerated database operations,
        # we'll pass self._backend_int to the native function.
        status = self.abi.lib.abi_db_create(self.abi.handle, dimension, ctypes.byref(self.handle))
        if status != AbiStatus.SUCCESS:
            raise AbiError(status, "Failed to create database")

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.abi.lib.abi_db_destroy(self.handle)
            self.handle = None

    def insert(self, id, vector):
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")
        
        c_vector = (c_float * len(vector))(*vector)
        status = self.abi.lib.abi_db_insert(self.handle, id, c_vector, len(vector))
        
        if status != AbiStatus.SUCCESS:
            raise AbiError(status, "Failed to insert vector")

    def search(self, vector, k=10):
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")
        
        c_vector = (c_float * len(vector))(*vector)
        ids = (c_uint64 * k)()
        scores = (c_float * k)()
        
        status = self.abi.lib.abi_db_search(
            self.handle, 
            c_vector, 
            len(vector), 
            k, 
            ids, 
            scores
        )
        
        if status != AbiStatus.SUCCESS:
            raise AbiError(status, "Failed to search vectors")
            
        results = []
        for i in range(k):
            # ABI returns 0 for ID/Score if fewer results found? 
            # Or we should probably return the count of results found.
            # The current C API assumes we fill the buffer up to k.
            # We filter out 0 IDs if that's the sentinel, or just return all for now.
            results.append({'id': ids[i], 'score': scores[i]})
            
        return results
