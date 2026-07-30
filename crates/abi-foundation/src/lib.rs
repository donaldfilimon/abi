//! Shared primitives for the ABI framework.
//!
//! This crate is the Rust successor to `src/foundation/` in the Zig tree. It has
//! no dependencies on any other ABI crate, so everything else can build on it.
//!
//! ## Notes on the port
//!
//! Several Zig modules existed only to work around toolchain constraints and
//! have no Rust counterpart. Where that is the case it is recorded here rather
//! than left as an unexplained gap:
//!
//! - `env.zig` required an explicit `install(init.environ_map)` from every
//!   entrypoint because that Zig std exposes no global `getenv`. Rust reads the
//!   environment directly, so [`mod@env`] drops the ceremony and keeps only the
//!   canonical variable names plus the "empty means unset" rule.
//! - `pool_allocator.zig` was a fixed-block allocator layered over
//!   `std.mem.Allocator`. Rust's allocator is not a threaded parameter, so
//!   pooling becomes a per-container concern rather than a foundation module.
//! - `io/` wrapped `std.Io` readers, writers and file streams. `std::io` already
//!   provides those traits, so the wrappers are gone; only the byte-counting
//!   [`io::CountingWriter`] carried real behaviour and survives.
//! - `sync.zig` wrapped `std.Thread.Mutex`. `std::sync` covers it directly.

pub mod credentials;
pub mod env;
pub mod errors;
pub mod hash_feature;
pub mod http;
pub mod io;
pub mod json;
pub mod logger;
pub mod plugin_manifest;
pub mod system;
pub mod temp_path;
pub mod text;
pub mod time;
pub mod validation;
pub mod wyhash;

pub use hash_feature::{Hash128, fnv1a_64, hash64, wyhash64, wyhash128};

pub use errors::{AbiError, ErrorContext, Result};
pub use logger::{Level, LogEntry, Logger};
