//! Frozen command metadata and help rendering for ABI's command-line surface.
//!
//! This crate is intentionally independent from command handlers. During the
//! Rust migration it provides a small, stable contract boundary that can be
//! checked against output captured from the Zig implementation.

pub mod usage;
