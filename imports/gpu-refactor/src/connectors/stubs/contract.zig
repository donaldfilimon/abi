//! Shared disabled-provider contract helpers.
//!
//! Keeps provider stubs behavior uniform: all runtime operations fail with
//! `error.ConnectorsDisabled` while static type surfaces remain available.

pub fn disabled(comptime T: type) error{ConnectorsDisabled}!T {
    return error.ConnectorsDisabled;
}

pub fn unavailable() bool {
    return false;
}
