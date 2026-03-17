# Language Bindings

High-level language bindings for the ABI framework.

## Status

This directory is reserved for future bindings targeting languages such as
Python, JavaScript/TypeScript, and others. Contributions are welcome.

## Existing Low-Level Bindings

The `bindings/` directory already provides two integration surfaces:

- **C** (`bindings/c/`) -- C header and source files exposing the core API,
  including the plugin registry (`abi_plugin_register`, etc.). Use these to
  embed ABI in any language with a C FFI.
- **WASM** (`bindings/wasm/`) -- WebAssembly target for browser and edge
  runtimes.

High-level bindings in this directory will typically wrap the C bindings with
idiomatic APIs for their respective language ecosystems.

## Adding a New Binding

1. Create a subdirectory named after the target language (e.g., `lang/python/`).
2. Provide a build script or manifest appropriate for that ecosystem.
3. Wrap the C API from `bindings/c/include/abi.h`.
4. Include a `README.md` with setup and usage instructions.
