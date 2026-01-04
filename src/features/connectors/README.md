//! Connectors Feature Overview
//!
//! Connectors provide integration points to external AI services and other
//! platforms. Implementations include OpenAI, Ollama, HuggingFace, and a local
//! scheduler. The `mod.zig` file re‑exports each connector's public API.
//!
//! Adding a new connector generally involves:
//!   1. Implementing a struct with the required request/response methods.
//!   2. Registering it in `shared.zig` for discovery.
//!   3. Writing tests in `../tests`.
