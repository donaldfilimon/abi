//! Pinned TFHE-rs interoperability demo.
//!
//! This module is available only with `full-fhe`. It exercises TFHE-rs 1.7.0
//! boolean, radix-integer, shortint, and programmable-bootstrapping APIs using
//! ephemeral in-memory keys. ABI supplies only the integration demonstration;
//! it makes no independent cryptographic-audit or production-safety claim.

use tfhe::boolean::prelude as boolean;
use tfhe::integer;
use tfhe::shortint;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;

/// Non-sensitive results from one ephemeral TFHE-rs run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TfheDemoReport {
    /// Whether the Boolean NAND result matched its plaintext oracle.
    pub boolean_nand_matches: bool,
    /// Whether radix integer addition matched its plaintext oracle.
    pub integer_add_matches: bool,
    /// Whether a shortint lookup table produced the expected value through PBS.
    pub programmable_bootstrap_matches: bool,
}

impl TfheDemoReport {
    /// Whether every exercised TFHE-rs API matched its plaintext oracle.
    #[must_use]
    pub const fn all_match(self) -> bool {
        self.boolean_nand_matches && self.integer_add_matches && self.programmable_bootstrap_matches
    }
}

/// Run the bounded, ephemeral TFHE-rs interoperability demonstration.
///
/// Keys and ciphertexts never leave this function and are not included in the
/// returned report. The shortint lookup-table application is a programmable
/// bootstrap supplied by TFHE-rs, not an ABI cryptographic implementation.
#[must_use]
pub fn run_tfhe_demo() -> TfheDemoReport {
    use boolean::{BinaryBooleanGates as _, ClientKey as BooleanClientKey};

    let (boolean_client, boolean_server) = boolean::gen_keys();
    let encrypted_true = BooleanClientKey::encrypt(&boolean_client, true);
    let encrypted_false = BooleanClientKey::encrypt(&boolean_client, false);
    let encrypted_nand = boolean_server.nand(&encrypted_true, &encrypted_false);
    let boolean_nand_matches = boolean_client.decrypt(&encrypted_nand);

    let (integer_client, integer_server) =
        integer::gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, 4);
    let encrypted_left = integer_client.encrypt(13_u64);
    let encrypted_right = integer_client.encrypt(29_u64);
    let encrypted_sum = integer_server.unchecked_add(&encrypted_left, &encrypted_right);
    let integer_sum: u64 = integer_client.decrypt(&encrypted_sum);

    let (shortint_client, shortint_server) = shortint::gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);
    let encrypted_value = shortint_client.encrypt(3);
    let lookup = shortint_server.generate_lookup_table(|value| (value + 1) % 4);
    let bootstrapped = shortint_server.apply_lookup_table(&encrypted_value, &lookup);
    let programmable_bootstrap_matches = shortint_client.decrypt(&bootstrapped) == 0;

    TfheDemoReport {
        boolean_nand_matches,
        integer_add_matches: integer_sum == 42,
        programmable_bootstrap_matches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_tfhe_apis_execute_with_ephemeral_keys() {
        assert!(run_tfhe_demo().all_match());
    }
}
