//! Secure password hashing module with modern algorithms.
//!
//! This module provides:
//! - Argon2id (recommended, memory-hard)
//! - PBKDF2-SHA256 (fallback, widely compatible)
//! - scrypt (alternative memory-hard)
//! - Password strength validation
//! - Secure comparison
//! - Hash format detection and migration

const std = @import("std");

const types = @import("password/types.zig");
const hasher_mod = @import("password/hasher.zig");
const strength_mod = @import("password/strength.zig");
const parse_mod = @import("password/parse.zig");

// Re-export types
pub const Algorithm = types.Algorithm;
pub const Argon2Params = types.Argon2Params;
pub const Pbkdf2Params = types.Pbkdf2Params;
pub const ScryptParams = types.ScryptParams;
pub const PasswordConfig = types.PasswordConfig;
pub const PasswordStrength = types.PasswordStrength;
pub const StrengthAnalysis = types.StrengthAnalysis;
pub const HashedPassword = types.HashedPassword;
pub const GenerateOptions = types.GenerateOptions;
pub const PasswordError = types.PasswordError;

// Re-export hasher
pub const PasswordHasher = hasher_mod.PasswordHasher;
pub const generatePassword = hasher_mod.generatePassword;

// Re-export strength analysis
pub const analyzeStrength = strength_mod.analyzeStrength;

// Re-export parsing
pub const detectAlgorithm = parse_mod.detectAlgorithm;

test {
    std.testing.refAllDecls(@This());
}
