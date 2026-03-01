# C Bindings

C-compatible FFI bindings for the ABI framework.

## Overview

These bindings provide a C-compatible API for the ABI framework, enabling integration with C, C++, and other languages that support C FFI.

## Building

```bash
cd bindings/c
zig build
```

This produces:
- `zig-out/lib/libabi.dylib` (macOS) or `libabi.so` (Linux) - Dynamic library
- `zig-out/lib/libabi.a` - Static library
- `zig-out/include/abi.h` - C header file

## Usage

```c
#include <abi.h>

int main(void) {
    abi_framework_t *fw = NULL;
    int err = abi_init(&fw);
    if (err != ABI_OK) {
        fprintf(stderr, "Init failed: %s\n", abi_error_string(err));
        return 1;
    }

    printf("ABI version: %s\n", abi_version());

    abi_shutdown(fw);
    return 0;
}
```

## API Categories

### Framework Lifecycle
- `abi_init()` - Initialize framework
- `abi_shutdown()` - Shutdown framework
- `abi_version()` - Get version string

### SIMD Operations
- `abi_simd_vector_add()` - Vector addition
- `abi_simd_vector_dot()` - Dot product
- `abi_simd_cosine_similarity()` - Cosine similarity

### Database Operations
- `abi_db_create()` - Create database
- `abi_db_insert()` - Insert vector
- `abi_db_search()` - Search vectors
- `abi_db_close()` - Close database

### GPU Operations
- `abi_gpu_init()` - Initialize GPU
- `abi_gpu_info()` - Get GPU information

## Linking

### macOS
```bash
export DYLD_LIBRARY_PATH=/path/to/bindings/c/zig-out/lib:$DYLD_LIBRARY_PATH
clang -I/path/to/bindings/c/zig-out/include -L/path/to/bindings/c/zig-out/lib -labi myprogram.c -o myprogram
```

### Linux
```bash
export LD_LIBRARY_PATH=/path/to/bindings/c/zig-out/lib:$LD_LIBRARY_PATH
gcc -I/path/to/bindings/c/zig-out/include -L/path/to/bindings/c/zig-out/lib -labi myprogram.c -o myprogram
```

## Error Handling

All functions return error codes. Check against `ABI_OK` for success:

```c
int err = abi_some_function(...);
if (err != ABI_OK) {
    fprintf(stderr, "Error: %s\n", abi_error_string(err));
}
```

## Memory Management

- Framework and database handles must be explicitly released with `abi_shutdown()` and `abi_db_close()`
- String results are typically static or managed by the library
- Vector data passed to functions is copied internally

## Regenerating Header

If the C API changes:

```bash
zig build c-header  # From project root
```
