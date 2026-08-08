//! Compatibility facade for the shared [`abi_compute`] crate.
//!
//! WDBX historically owned these symbols. Re-exporting them from this module
//! keeps `abi_wdbx::compute::*` and the crate-root API source-compatible while
//! compute ownership moves to the cycle-free leaf crate.

pub use abi_compute::*;
