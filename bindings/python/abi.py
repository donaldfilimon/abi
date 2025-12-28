#!/usr/bin/env python3
"""
Python bindings for ABI framework

Provides a Pythonic interface to the ABI compute runtime.
"""

import ctypes
import platform
import os
from typing import Optional, List, Tuple, Callable

# Load the shared library
if platform.system() == "Windows":
    LIB_PATH = os.path.join(os.path.dirname(__file__), "abi.dll")
else:
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libabi.so")

try:
    _abi = ctypes.CDLL(LIB_PATH)
except OSError:
    _abi = None
    _lib_path = LIB_PATH


# Error codes
class AbiErrorCode:
    SUCCESS = 0
    OUT_OF_MEMORY = 1
    INVALID_ARGUMENT = 2
    NOT_FOUND = 3
    TIMEOUT = 4
    QUEUE_FULL = 5
    IO_ERROR = 6
    UNKNOWN_ERROR = 999


# Log levels
class AbiLogLevel:
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4


class AbiError(Exception):
    """ABI framework error"""

    pass


class AbiNotInitializedError(AbiError):
    """ABI library not initialized"""

    pass


# ctypes definitions for ABI config
class AbiConfigStruct(ctypes.Structure):
    _fields_ = [
        ("max_tasks", ctypes.c_size_t),
        ("numa_enabled", ctypes.c_int),
        ("cpu_affinity_enabled", ctypes.c_int),
    ]


# Define function signatures if library is available
if _abi:
    _abi.abi_init.argtypes = [ctypes.POINTER(AbiConfigStruct)]
    _abi.abi_init.restype = ctypes.c_int

    _abi.abi_deinit.argtypes = []
    _abi.abi_deinit.restype = None

    _abi.abi_get_cpu_count.argtypes = []
    _abi.abi_get_cpu_count.restype = ctypes.c_size_t

    _abi.abi_get_numa_node_count.argtypes = []
    _abi.abi_get_numa_node_count.restype = ctypes.c_size_t

    _abi.abi_set_thread_affinity.argtypes = [ctypes.c_size_t]
    _abi.abi_set_thread_affinity.restype = ctypes.c_int

    _abi.abi_get_current_cpu.argtypes = []
    _abi.abi_get_current_cpu.restype = ctypes.c_int

    _abi.abi_submit_task.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _abi.abi_submit_task.restype = ctypes.c_uint64

    _abi.abi_wait_result.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    _abi.abi_wait_result.restype = ctypes.c_int

    _abi.abi_database_create.argtypes = [ctypes.c_char_p]
    _abi.abi_database_create.restype = ctypes.c_void_p

    _abi.abi_database_insert.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    _abi.abi_database_insert.restype = ctypes.c_int

    _abi.abi_database_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _abi.abi_database_search.restype = ctypes.c_int

    _abi.abi_database_destroy.argtypes = [ctypes.c_void_p]
    _abi.abi_database_destroy.restype = ctypes.c_int

    _abi.abi_string_free.argtypes = [ctypes.c_char_p]
    _abi.abi_string_free.restype = None

    _abi.abi_get_last_error.argtypes = []
    _abi.abi_get_last_error.restype = ctypes.c_char_p

    _abi.abi_set_log_callback.argtypes = [ctypes.c_void_p]
    _abi.abi_set_log_callback.restype = None


class AbiConfig:
    """Configuration for ABI framework"""

    def __init__(
        self,
        max_tasks: int = 1024,
        numa_enabled: bool = False,
        cpu_affinity_enabled: bool = False,
    ):
        self.max_tasks = max_tasks
        self.numa_enabled = 1 if numa_enabled else 0
        self.cpu_affinity_enabled = 1 if cpu_affinity_enabled else 0

    def to_ctype(self):
        """Convert to ctypes structure"""
        return AbiConfigStruct(
            max_tasks=self.max_tasks,
            numa_enabled=self.numa_enabled,
            cpu_affinity_enabled=self.cpu_affinity_enabled,
        )


class AbiFramework:
    """Main ABI framework interface"""

    def __init__(self, config: Optional[AbiConfig] = None):
        if _abi is None:
            raise AbiNotInitializedError(
                "ABI library not found at {}. Build the C API first.".format(_lib_path)
            )

        self._config = config or AbiConfig()
        self._initialized = False

    def init(self):
        """Initialize the ABI framework"""
        if self._initialized:
            return

        config_ctype = self._config.to_ctype()
        result = _abi.abi_init(ctypes.byref(config_ctype))

        if result != AbiErrorCode.SUCCESS:
            raise AbiError("Failed to initialize ABI: error {}".format(result))

        self._initialized = True

    def deinit(self):
        """Cleanup the ABI framework"""
        if not self._initialized:
            return

        _abi.abi_deinit()
        self._initialized = False

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deinit()

    @property
    def cpu_count(self):
        """Get the number of CPUs"""
        return _abi.abi_get_cpu_count()

    @property
    def numa_node_count(self):
        """Get the number of NUMA nodes"""
        return _abi.abi_get_numa_node_count()

    def set_thread_affinity(self, cpu_id):
        """Set current thread affinity to specific CPU"""
        result = _abi.abi_set_thread_affinity(cpu_id)
        if result != AbiErrorCode.SUCCESS:
            raise AbiError("Failed to set thread affinity: error {}".format(result))

    def get_current_cpu(self):
        """Get current CPU ID"""
        result = _abi.abi_get_current_cpu()
        if result == AbiErrorCode.IO_ERROR:
            raise AbiError("Failed to get current CPU")
        return result

    def submit_task(self, task_fn, user_data=None):
        """Submit a task to the compute engine"""
        # This is a simplified version - real implementation needs more work
        task_ctype = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
        return _abi.abi_submit_task(task_ctype(task_fn), user_data)

    def wait_result(self, task_id, timeout_ms=0):
        """Wait for a task result"""
        result = _abi.abi_wait_result(task_id, timeout_ms)
        if result == AbiErrorCode.UNKNOWN_ERROR:
            raise AbiError("Unknown error waiting for task {}".format(task_id))
        elif result == AbiErrorCode.TIMEOUT:
            raise AbiError("Task {} timed out".format(task_id))


# Convenience functions
def get_cpu_count():
    """Get system CPU count"""
    if _abi is None:
        return os.cpu_count()
    return _abi.abi_get_cpu_count()


def get_numa_node_count():
    """Get NUMA node count"""
    if _abi is None:
        return 1
    return _abi.abi_get_numa_node_count()


def get_library_path():
    """Get ABI library path"""
    return _lib_path if _abi else None


def is_available():
    """Check if ABI library is available"""
    return _abi is not None


__all__ = [
    "AbiFramework",
    "AbiConfig",
    "AbiError",
    "AbiErrorCode",
    "AbiLogLevel",
    "AbiNotInitializedError",
    "get_cpu_count",
    "get_numa_node_count",
    "get_library_path",
    "is_available",
]
