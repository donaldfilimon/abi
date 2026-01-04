//! Database Feature Overview
//!
//! This feature implements persistent storage utilities used by the ABI
//! framework. It includes a high‑level `Database` abstraction, HTTP helpers for
//! remote access, and a unified interface (`unified.zig`) that hides backend
//! details. The directory layout:
//!   - `mod.zig` – Public API entry point.
//!   - `database.zig` – Core DB logic.
//!   - `http.zig` – HTTP client wrappers.
//!   - `index.zig` – Indexing utilities.
//!   - `cli.zig` – Command‑line interface for manual inspection.
//!   - `db_helpers.zig` – Miscellaneous helper functions.
//!
//! Extend with additional backends by conforming to the `Database` interface
//! and updating `mod.zig`.
