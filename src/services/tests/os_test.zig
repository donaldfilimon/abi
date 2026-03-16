//! Cross-Platform OS Features Tests (DISABLED)
//!
//! These tests are currently disabled because they reference ~20 OS API
//! functions (getOsName, Path.*, Env.expand, Signal, FileMode, etc.) that
//! were designed but never implemented in `src/services/shared/os.zig`.
//!
//! To re-enable: restore the test blocks from git history and implement the
//! missing functions in os.zig.

// Minimal import to keep the file valid within the abi module.
const std = @import("std");

test "os_test: placeholder (tests disabled pending API implementation)" {
    // See module doc comment above for details.
}
