//! Platform Abstractions
//!
//! Abstracts OS‑specific functionality (e.g., thread spawning, file I/O) so
//! the rest of the codebase can remain portable across Windows, Linux, and
//! macOS. The `mod.zig` file defines the cross‑platform interfaces.
//!
//! Extending for a new platform involves implementing the required functions in
//! a platform‑specific file and updating `mod.zig` with conditional compilation.
