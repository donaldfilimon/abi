//! Plugin System
//!
//! Provides a lightweight plugin architecture allowing optional modules to be
//! loaded at runtime. Plugins must expose an `init` function conforming to the
//! `Plugin` interface defined in `mod.zig`.
//!
//! To create a new plugin, add a sub‑directory with its implementation and
//! register it in the plugin registry.
