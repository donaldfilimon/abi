# GPU Architecture Diagram
> **Codebase Status:** Synced with repository as of 2026-01-18.

```mermaid
flowchart TB
    subgraph "Unified API Layer"
        API[Gpu API<br/>vectorAdd, matrixMultiply, etc.]
    end

    subgraph "Dispatch Layer"
        DISP[KernelDispatcher]
        CACHE[Kernel Cache]
        BUILTIN[Builtin Kernels]
    end

    subgraph "DSL Layer"
        BUILDER[KernelBuilder]
        IR[Kernel IR]
        CODEGEN[Code Generators]
    end

    subgraph "Backend Layer"
        FACTORY[Backend Factory]
        subgraph "VTable Backends"
            CUDA[CUDA VTable]
            VULKAN[Vulkan VTable]
            METAL[Metal VTable]
            WEBGPU[WebGPU VTable]
            STDGPU[STDGPU CPU Fallback]
        end
    end

    subgraph "Device Layer"
        DEVICE[Device Manager]
        BUFFER[Unified Buffer]
        STREAM[Stream Manager]
    end

    API --> DISP
    DISP --> CACHE
    DISP --> BUILTIN
    BUILTIN --> BUILDER
    BUILDER --> IR
    IR --> CODEGEN
    CODEGEN --> FACTORY
    FACTORY --> CUDA
    FACTORY --> VULKAN
    FACTORY --> METAL
    FACTORY --> WEBGPU
    FACTORY --> STDGPU
    API --> DEVICE
    API --> BUFFER
    API --> STREAM
```

## Components

| Component | File | Description |
|-----------|------|-------------|
| Gpu API | `unified.zig` | High-level unified interface |
| KernelDispatcher | `dispatcher.zig` | Routes kernels to backends |
| Builtin Kernels | `builtin_kernels.zig` | Pre-defined kernel IR |
| KernelBuilder | `dsl/builder.zig` | Fluent API for kernel construction |
| Backend Factory | `backend_factory.zig` | Creates VTable backends |
| Device Manager | `device.zig` | Device discovery and selection |
| Unified Buffer | `unified_buffer.zig` | Memory management across backends |
