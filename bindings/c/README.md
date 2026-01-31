# ABI C Bindings

C-compatible bindings for the ABI framework, enabling integration with C, C++, Rust, Go, Python, and other languages via FFI.

## Building

```bash
cd bindings/c
zig build
```

This produces:
- `zig-out/lib/libabi.so` (or `.dylib`/`.dll`) - Shared library
- `zig-out/lib/libabi_static.a` - Static library
- `zig-out/include/` - C headers

## Running Tests

```bash
cd bindings/c
zig build test
```

## Usage

### Basic Example

```c
#include <abi.h>

int main() {
    abi_framework_t fw = NULL;

    // Initialize with defaults
    if (abi_init(&fw) != ABI_OK) {
        return 1;
    }

    printf("ABI v%s\n", abi_version());

    // Check SIMD
    abi_simd_caps_t caps;
    abi_simd_get_caps(&caps);
    printf("AVX2: %s\n", caps.avx2 ? "yes" : "no");

    // Use SIMD operations
    float a[] = {1, 2, 3, 4};
    float b[] = {4, 3, 2, 1};
    float dot = abi_simd_vector_dot(a, b, 4);

    abi_shutdown(fw);
    return 0;
}
```

Compile with:
```bash
# Dynamic linking
gcc -I/path/to/include -L/path/to/lib -labi example.c -o example
LD_LIBRARY_PATH=/path/to/lib ./example

# Static linking
gcc -I/path/to/include example.c /path/to/lib/libabi_static.a -o example -lm
```

### Vector Database Example

```c
#include <abi.h>

int main() {
    abi_database_t db = NULL;
    abi_database_config_t config = {
        .name = "vectors",
        .dimension = 384,
        .initial_capacity = 10000
    };

    // Create database
    if (abi_database_create(&config, &db) != ABI_OK) {
        return 1;
    }

    // Insert vectors
    float embedding[384] = { /* ... */ };
    abi_database_insert(db, 1, embedding, 384);

    // Search
    abi_search_result_t results[10];
    size_t count = 0;
    abi_database_search(db, embedding, 384, 10, results, &count);

    for (size_t i = 0; i < count; i++) {
        printf("ID: %llu, Score: %.4f\n",
               (unsigned long long)results[i].id,
               results[i].score);
    }

    abi_database_close(db);
    return 0;
}
```

### Agent Example

```c
#include <abi.h>

int main() {
    abi_framework_t fw = NULL;
    abi_agent_t agent = NULL;
    char* response = NULL;

    abi_init(&fw);

    abi_agent_config_t config = {
        .name = "assistant",
        .persona = NULL,
        .temperature = 0.7f,
        .enable_history = true
    };

    abi_agent_create(fw, &config, &agent);

    if (abi_agent_chat(agent, "Hello!", &response) == ABI_OK) {
        printf("Response: %s\n", response);
        abi_free_string(response);
    }

    abi_agent_destroy(agent);
    abi_shutdown(fw);
    return 0;
}
```

## API Reference

### Framework Lifecycle

| Function | Description |
|----------|-------------|
| `abi_init(fw*)` | Initialize with defaults |
| `abi_init_with_options(opts*, fw*)` | Initialize with custom options |
| `abi_shutdown(fw)` | Release resources |
| `abi_version()` | Get version string |
| `abi_version_info(info*)` | Get detailed version info |
| `abi_is_feature_enabled(fw, name)` | Check feature status |

### SIMD Operations

| Function | Description |
|----------|-------------|
| `abi_simd_get_caps(caps*)` | Query CPU capabilities |
| `abi_simd_available()` | Check if SIMD available |
| `abi_simd_vector_add(a, b, result, len)` | Element-wise addition |
| `abi_simd_vector_dot(a, b, len)` | Dot product |
| `abi_simd_vector_l2_norm(v, len)` | L2 norm |
| `abi_simd_cosine_similarity(a, b, len)` | Cosine similarity |

### Database Operations

| Function | Description |
|----------|-------------|
| `abi_database_create(config*, db*)` | Create vector database |
| `abi_database_close(db)` | Close database |
| `abi_database_insert(db, id, vec, len)` | Insert vector |
| `abi_database_search(db, query, len, k, results, count*)` | Search similar |
| `abi_database_delete(db, id)` | Delete vector |
| `abi_database_count(db, count*)` | Get vector count |

### GPU Operations

| Function | Description |
|----------|-------------|
| `abi_gpu_init(config*, gpu*)` | Initialize GPU context |
| `abi_gpu_shutdown(gpu)` | Release GPU resources |
| `abi_gpu_is_available()` | Check GPU availability |
| `abi_gpu_backend_name(gpu)` | Get backend name |

### Agent Operations

| Function | Description |
|----------|-------------|
| `abi_agent_create(fw, config*, agent*)` | Create AI agent |
| `abi_agent_destroy(agent)` | Destroy agent |
| `abi_agent_chat(agent, msg, response**)` | Send message |
| `abi_agent_clear_history(agent)` | Clear conversation |

### Memory Management

| Function | Description |
|----------|-------------|
| `abi_free_string(str)` | Free allocated string |
| `abi_free_results(results, count)` | Free search results |

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `ABI_OK` | Success |
| -1 | `ABI_ERROR_INIT_FAILED` | Initialization failed |
| -2 | `ABI_ERROR_ALREADY_INITIALIZED` | Already initialized |
| -3 | `ABI_ERROR_NOT_INITIALIZED` | Not initialized |
| -4 | `ABI_ERROR_OUT_OF_MEMORY` | Allocation failed |
| -5 | `ABI_ERROR_INVALID_ARGUMENT` | Invalid parameter |
| -6 | `ABI_ERROR_FEATURE_DISABLED` | Feature not compiled |
| -7 | `ABI_ERROR_TIMEOUT` | Operation timed out |
| -8 | `ABI_ERROR_IO` | I/O error |
| -9 | `ABI_ERROR_GPU_UNAVAILABLE` | GPU not available |
| -10 | `ABI_ERROR_DATABASE_ERROR` | Database error |
| -11 | `ABI_ERROR_NETWORK_ERROR` | Network error |
| -12 | `ABI_ERROR_AI_ERROR` | AI error |
| -99 | `ABI_ERROR_UNKNOWN` | Unknown error |

## Thread Safety

- **Framework initialization**: NOT thread-safe (call once at startup)
- **SIMD operations**: Thread-safe (stateless)
- **Database operations**: Thread-safe (internal locking)
- **GPU operations**: NOT thread-safe (use one context per thread)
- **Agent operations**: NOT thread-safe (use one agent per thread)

## Configuration Defaults

```c
// All features enabled
abi_options_t opts = ABI_OPTIONS_DEFAULT;

// 384-dimensional database with 1000 initial capacity
abi_database_config_t db_cfg = ABI_DATABASE_CONFIG_DEFAULT;

// Auto-detect GPU backend, first device
abi_gpu_config_t gpu_cfg = ABI_GPU_CONFIG_DEFAULT;

// Assistant agent with temperature 0.7
abi_agent_config_t agent_cfg = ABI_AGENT_CONFIG_DEFAULT;
```

## FFI from Other Languages

### Rust

```rust
use std::ffi::CString;
use std::os::raw::c_char;

#[link(name = "abi")]
extern "C" {
    fn abi_version() -> *const c_char;
    fn abi_simd_available() -> bool;
}

fn main() {
    unsafe {
        let version = std::ffi::CStr::from_ptr(abi_version());
        println!("ABI v{}", version.to_str().unwrap());
        println!("SIMD: {}", abi_simd_available());
    }
}
```

### Python (ctypes)

```python
import ctypes

lib = ctypes.CDLL("./libabi.so")
lib.abi_version.restype = ctypes.c_char_p
lib.abi_simd_available.restype = ctypes.c_bool

print(f"ABI v{lib.abi_version().decode()}")
print(f"SIMD: {lib.abi_simd_available()}")
```

### Go (cgo)

```go
package main

/*
#cgo LDFLAGS: -L. -labi
#include "abi.h"
*/
import "C"
import "fmt"

func main() {
    fmt.Printf("ABI v%s\n", C.GoString(C.abi_version()))
    fmt.Printf("SIMD: %v\n", C.abi_simd_available())
}
```

## License

MIT - See [LICENSE](../../LICENSE)
