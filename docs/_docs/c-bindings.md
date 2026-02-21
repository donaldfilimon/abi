---
title: C API Bindings
description: 36 C-compatible function exports for FFI
section: Reference
order: 3
---

# C API Bindings

ABI provides 36 C-compatible function exports in `bindings/c/src/abi_c.zig`,
enabling integration from C, C++, Python (via ctypes/cffi), Rust (via FFI),
and any other language with C interop.

## Building the C Library

```bash
# Build the shared library (produces libabi.so / libabi.dylib)
zig build lib

# Install the C header file
zig build c-header
```

The build system produces:
- **Shared library**: `zig-out/lib/libabi.{so,dylib,dll}` (dynamic linkage)
- **Header file**: `zig-out/include/abi.h` (installed via `zig build c-header`)

The C library links libc and imports the `abi` and `build_options` modules from
the main framework.

---

## Error Codes

All C API functions return integer status codes. Zero indicates success;
negative values indicate errors.

| Code | Constant | Description |
|------|----------|-------------|
| `0` | `ABI_OK` | Success |
| `-1` | `ABI_ERROR_INIT_FAILED` | Initialization failed |
| `-2` | `ABI_ERROR_ALREADY_INITIALIZED` | Already initialized |
| `-3` | `ABI_ERROR_NOT_INITIALIZED` | Not initialized |
| `-4` | `ABI_ERROR_OUT_OF_MEMORY` | Out of memory |
| `-5` | `ABI_ERROR_INVALID_ARGUMENT` | Invalid argument |
| `-6` | `ABI_ERROR_FEATURE_DISABLED` | Feature disabled at compile time |
| `-7` | `ABI_ERROR_TIMEOUT` | Operation timed out |
| `-8` | `ABI_ERROR_IO` | I/O error |
| `-9` | `ABI_ERROR_GPU_UNAVAILABLE` | GPU not available |
| `-10` | `ABI_ERROR_DATABASE_ERROR` | Database error |
| `-11` | `ABI_ERROR_NETWORK_ERROR` | Network error |
| `-12` | `ABI_ERROR_AI_ERROR` | AI operation error |
| `-99` | `ABI_ERROR_UNKNOWN` | Unknown error |

---

## API Categories

### Framework Lifecycle

| Function | Description |
|----------|-------------|
| `abi_init(options)` | Initialize framework with options (NULL for defaults) |
| `abi_deinit(fw)` | Shut down framework and free resources |
| `abi_version()` | Return version string (e.g., `"0.4.0"`) |
| `abi_version_info(major, minor, patch)` | Parse version into components |
| `abi_is_feature_enabled(name)` | Check if a feature is compiled in |

### GPU Operations

| Function | Description |
|----------|-------------|
| `abi_gpu_is_available(fw)` | Check GPU hardware availability |
| `abi_gpu_create_context(fw, backend)` | Create GPU context for a backend |
| `abi_gpu_destroy_context(ctx)` | Release GPU context |
| `abi_gpu_dispatch_kernel(fw, name, data, len)` | Dispatch a compute kernel |
| `abi_gpu_get_device_count(fw)` | Return number of GPU devices |
| `abi_gpu_get_backend(fw)` | Return active backend enum |

### SIMD Operations

| Function | Description |
|----------|-------------|
| `abi_simd_has_support()` | Check for SIMD hardware support |
| `abi_simd_get_capabilities(caps)` | Fill capabilities struct |
| `abi_simd_vector_add(a, b, result, len)` | Element-wise vector addition |
| `abi_simd_vector_dot(a, b, len)` | Dot product of two vectors |
| `abi_simd_cosine_similarity(a, b, len)` | Cosine similarity between vectors |
| `abi_simd_vector_l2_norm(v, len)` | L2 norm of a vector |

### Database Operations

| Function | Description |
|----------|-------------|
| `abi_db_open(path, config)` | Open or create a WDBX database |
| `abi_db_close(db)` | Close a database handle |
| `abi_db_insert(db, key, vector, dim)` | Insert a vector |
| `abi_db_query(db, vector, dim, k, results)` | k-NN similarity search |
| `abi_db_delete(db, key)` | Delete a vector by key |
| `abi_db_count(db)` | Return number of stored vectors |

### AI and Agent Operations

| Function | Description |
|----------|-------------|
| `abi_agent_create(fw, config)` | Create an AI agent |
| `abi_agent_destroy(agent)` | Destroy an agent |
| `abi_agent_query(agent, prompt, response, len)` | Send a query to the agent |
| `abi_agent_add_tool(agent, name, callback)` | Register a tool with the agent |
| `abi_agent_get_stats(agent, stats)` | Retrieve agent statistics |

### Memory Management

| Function | Description |
|----------|-------------|
| `abi_free_string(str)` | Free a string allocated by the C API |
| `abi_free_results(results, count)` | Free a search results array |

### Error Handling

| Function | Description |
|----------|-------------|
| `abi_error_string(code)` | Return a human-readable error message |

---

## Usage Example (C)

```c
#include <stdio.h>
#include "abi.h"

int main(void) {
    // Initialize the framework
    AbiFramework* fw = abi_init(NULL);
    if (!fw) {
        printf("Init failed: %s\n", abi_error_string(ABI_ERROR_INIT_FAILED));
        return 1;
    }

    // Print version
    printf("ABI version: %s\n", abi_version());

    // Check feature availability
    if (abi_is_feature_enabled("gpu")) {
        printf("GPU is available: %s\n",
               abi_gpu_is_available(fw) ? "yes" : "no");
    }

    // SIMD operations
    if (abi_simd_has_support()) {
        float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float result[4];

        abi_simd_vector_add(a, b, result, 4);
        printf("Vector add: [%.1f, %.1f, %.1f, %.1f]\n",
               result[0], result[1], result[2], result[3]);

        float dot = abi_simd_vector_dot(a, b, 4);
        printf("Dot product: %.1f\n", dot);
    }

    // Database operations
    AbiDatabase* db = abi_db_open("./vectors.wdbx", NULL);
    if (db) {
        float vec[] = {0.1f, 0.2f, 0.3f};
        abi_db_insert(db, "item-1", vec, 3);

        printf("Stored vectors: %d\n", abi_db_count(db));
        abi_db_close(db);
    }

    // Cleanup
    abi_deinit(fw);
    return 0;
}
```

## Usage Example (C++)

```cpp
#include <iostream>
#include <vector>

extern "C" {
#include "abi.h"
}

int main() {
    AbiFramework* fw = abi_init(nullptr);
    if (!fw) {
        std::cerr << "Init failed\n";
        return 1;
    }

    std::cout << "ABI v" << abi_version() << std::endl;

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};

    float similarity = abi_simd_cosine_similarity(
        a.data(), b.data(), a.size()
    );
    std::cout << "Cosine similarity: " << similarity << std::endl;

    abi_deinit(fw);
    return 0;
}
```

---

## Linking Against ABI

### With Zig Build System

The simplest approach is to use Zig as both the compiler and linker:

```bash
# Build the shared library
zig build lib

# Compile and link a C program
zig cc -o myapp myapp.c -Lzig-out/lib -labi -Izig-out/include
```

### With CMake

```cmake
find_library(ABI_LIB abi PATHS ${ABI_ROOT}/zig-out/lib)
find_path(ABI_INCLUDE abi.h PATHS ${ABI_ROOT}/zig-out/include)

add_executable(myapp main.c)
target_link_libraries(myapp ${ABI_LIB})
target_include_directories(myapp PRIVATE ${ABI_INCLUDE})
```

### With GCC/Clang

```bash
gcc -o myapp myapp.c -L/path/to/zig-out/lib -labi -I/path/to/zig-out/include
# On macOS, set DYLD_LIBRARY_PATH for runtime:
export DYLD_LIBRARY_PATH=/path/to/zig-out/lib:$DYLD_LIBRARY_PATH
# On Linux, set LD_LIBRARY_PATH:
export LD_LIBRARY_PATH=/path/to/zig-out/lib:$LD_LIBRARY_PATH
```

---

## Feature Gating in C

When a feature is disabled at compile time, the corresponding C API functions
return `ABI_ERROR_FEATURE_DISABLED` (-6). Always check the return code:

```c
int rc = abi_gpu_create_context(fw, ABI_GPU_BACKEND_METAL);
if (rc == ABI_ERROR_FEATURE_DISABLED) {
    printf("GPU support was disabled at compile time\n");
} else if (rc < 0) {
    printf("GPU error: %s\n", abi_error_string(rc));
}
```

---

## Testing

The C API has comprehensive integration tests in `src/services/tests/integration/`:

| Test File | Coverage |
|-----------|----------|
| `c_api_test.zig` | Core lifecycle, version, feature detection, error codes, memory safety |
| `c_api_simd_test.zig` | SIMD capabilities, vector operations, struct layout |
| `c_api_database_test.zig` | Database CRUD, count, delete, configuration |
| `c_api_gpu_test.zig` | GPU availability, lifecycle, backend detection |
| `c_api_agent_test.zig` | Agent CRUD, messaging, stats, history |

Run with:

```bash
zig build test --summary all
```

---

## Related Pages

- [API Overview](api.html) -- Full Zig API surface
- [GPU Module](gpu.html) -- GPU backends and kernel DSL
- [Troubleshooting](troubleshooting.html) -- Common errors and debugging

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
