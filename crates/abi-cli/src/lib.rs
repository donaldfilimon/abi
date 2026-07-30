//! Command metadata, help rendering, and process dispatch for ABI's CLI.
//!
//! During the Rust migration, the help surface is a stable contract boundary
//! checked against output captured from the Zig implementation. Command
//! handlers are attached incrementally behind the same process dispatcher.

pub mod app;
pub mod usage;

mod wdbx;
