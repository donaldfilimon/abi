---
title: "GPU Backend Completeness"
tags: [gpu, backends, hardware]
---
# GPU Backend Completeness
> **Codebase Status:** Synced with repository as of 2026-01-30.

<p align="center">
  <img src="https://img.shields.io/badge/Vulkan-Implemented-success?style=for-the-badge" alt="Vulkan"/>
  <img src="https://img.shields.io/badge/Metal-Implemented-success?style=for-the-badge" alt="Metal"/>
  <img src="https://img.shields.io/badge/WebGPU-Implemented-success?style=for-the-badge" alt="WebGPU"/>
  <img src="https://img.shields.io/badge/CUDA-Stubbed-yellow?style=for-the-badge" alt="CUDA"/>
</p>

The ABI framework supports multiple GPU backends. The current implementation includes:

| Backend | Status | Notes |
|--------|--------|-------|
| **Vulkan** | ✅ Implemented | Fully functional (see `src/gpu/backends/vulkan.zig`).
| **Metal**   | ✅ Implemented | Works on macOS Apple Silicon and Intel Macs (see `src/gpu/backends/metal.zig`).
| **WebGPU**  | ✅ Implemented | Uses the Zig `std.gpu` abstraction (see `src/gpu/backends/webgpu.zig`).
| **CUDA**   | ❌ Stubbed | Placeholder stub present in `src/gpu/backends/cuda.zig`. Enable with `-Denable-gpu` and appropriate driver.
| **OpenGL / OpenGLES / WebGL2** | ❌ Stubbed | Stubs exist for future expansion.
| **FPGA**    | ❌ Stubbed | Experimental vtable present, not yet production‑ready.

## How to Enable

Pass the desired backends via the `-Dgpu-backend` build flag, e.g.:

```batch
zig build -Denable-gpu=true -Dgpu-backend="vulkan,metal,webgpu"
```

The build system will automatically select the first available backend from the list.

## Testing

The **CLI smoke‑test** (`scripts/run_cli_tests.bat`) exercises the Vulkan and Metal backends through the example commands `run-compute` and `run-discord`.  To run the full GPU test suite, use:

```batch
zig build test -Denable-gpu=true -Dgpu-backend="vulkan,metal,webgpu"
```

If a backend is unavailable on the current platform (e.g., Metal on Windows), the tests will be skipped gracefully.

## Future Work

* Implement full CUDA support and expose a `run-cuda` example.
* Add OpenGL / OpenGLES demo programs.
* Provide a WebGPU‑only fallback for browsers via WASM.

