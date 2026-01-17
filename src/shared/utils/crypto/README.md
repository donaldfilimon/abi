//! Crypto Utilities
//!
//! Helper functions for common cryptographic operations used throughout the
//! framework (hashing, HMAC, etc.). The module currently only provides a thin
//! wrapper around `std.crypto`. Extend by adding additional algorithms and
//! exposing them via `mod.zig`.

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.
