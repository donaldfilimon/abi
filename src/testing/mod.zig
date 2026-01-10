//! Testing infrastructure for ABI framework.
//!
//! Provides property-based testing, fuzzing, and test utilities
//! for comprehensive framework testing.

pub const proptest = @import("proptest.zig");

pub const Generator = proptest.Generator;
pub const Generators = proptest.Generators;
pub const PropTest = proptest.PropTest;
pub const PropTestConfig = proptest.PropTestConfig;
pub const PropTestResult = proptest.PropTestResult;
pub const Assertions = proptest.Assertions;
pub const Fuzzer = proptest.Fuzzer;
pub const forAll = proptest.forAll;

test {
    @import("std").testing.refAllDecls(@This());
}
