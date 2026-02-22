# ABI Framework - Gemini CLI Context

## Project Overview

**ABI Framework** is a modern, high-performance Zig framework designed for AI services, vector search, and distributed compute systems.

**Key Features:**
*   **AI Runtime:** LLM inference (Llama-CPP parity), agent runtime, training pipelines.
*   **Vector Database:** WDBX with HNSW/IVF-PQ indexing, hybrid search.
*   **GPU Acceleration:** Support for CUDA, Vulkan, Metal, WebGPU, TPU (stub), and FPGA.
*   **Compute Engine & Distributed Network:** Work-stealing scheduler, Raft consensus, load balancing.
*   **Modular Architecture:** Every feature is toggleable at compile-time via feature flags.

**Technology Stack:**
*   **Language:** Zig (Strictly pinned to `0.16.0-dev.2623+27eec9bd6` as per `.zigversion`).
*   **Toolchain Manager:** `zvm` is recommended for managing the Zig version.

## Building and Running

The project relies on the standard Zig build system. 

*   **Build the project:**
    ```bash
    zig build
    ```
*   **Run the CLI (Help):**
    ```bash
    zig build run -- --help
    ```
    *(If installed system-wide or in PATH, the binary is `abi`)*
*   **Run all tests:**
    ```bash
    zig build test --summary all
    ```
*   **Run benchmarks:**
    ```bash
    zig build benchmarks
    ```
*   **Formatting and Linting:**
    ```bash
    zig build fix   # Format sources in place
    zig build lint  # Lint check
    ```

## Development Conventions

*   **Zig Version Rigidity:** The project uses a very specific development build of Zig. Always ensure `zig version` matches the contents of `.zigversion`. Use `zig run tools/scripts/toolchain_doctor.zig` if version drift is suspected.
*   **Modularity:** The framework is highly modular. When making changes, respect the feature flags (`-Denable-ai`, `-Denable-gpu`, etc.) to ensure the framework can still be built minimally without overhead.
*   **AI & Agent Guidelines:** The repository contains rich documentation for AI assistants and contributors. When in doubt about patterns, refer to:
    *   `CLAUDE.md` for Zig 0.16 patterns, gotchas, and specific AI assistant guidelines.
    *   `AGENTS.md` for baseline agent guidelines and multi-agent roles.
    *   `CONTRIBUTING.md` for the development workflow and PR checklist.
*   **Performance:** Code changes should prioritize zero-cost abstractions, comptime optimization, and SIMD-accelerated operations where applicable, maintaining the framework's bare-metal performance goals.
