//! Shared Utilities Overview

The **shared** package contains cross‑cutting concerns used throughout the ABI framework.

* `logging` – Structured, leveled logging with optional back‑ends.
* `observability` – Metrics, tracing, and diagnostic helpers.
* `platform` – OS‑specific abstractions (file handles, threading, etc.).
* `plugins` – Plugin registration and discovery system.
* `utils` – General‑purpose utility libraries (crypto, encoding, file‑system, HTTP, JSON, math, networking, string handling).

Each subdirectory ships a `mod.zig` that re‑exports its symbols, and the top‑level `src/shared/mod.zig` aggregates them for convenient importing.

