//! Cross-implementation Abbey corpus qualification.

use abi_contracts::{ContractError, Corpus, canonicalize_jcs};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("contracts/abbey")
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().fold(
        String::with_capacity(bytes.len() * 2),
        |mut output, byte| {
            write!(output, "{byte:02x}").expect("writing to a String cannot fail");
            output
        },
    )
}

#[test]
fn independent_digest() {
    let verified = Corpus::open(corpus_root())
        .expect("corpus opens")
        .verify()
        .expect("corpus verifies");
    assert!(verified.artifact_count() > 70);

    let vector: Value = serde_json::from_slice(
        &fs::read(corpus_root().join("v1/fixtures/valid/corpus-digest-vector.json"))
            .expect("digest vector reads"),
    )
    .expect("digest vector parses");
    let document = &vector["document"];
    let mut input = b"abbey-contract-corpus-v1\0".to_vec();
    for entry in document["entries"].as_array().expect("entries array") {
        input.extend_from_slice(entry["path"].as_str().expect("path").as_bytes());
        input.push(0);
        input.extend_from_slice(entry["bytes"].as_str().expect("bytes").as_bytes());
        input.push(0);
        input.extend_from_slice(entry["sha256"].as_str().expect("sha256").as_bytes());
        input.push(b'\n');
    }
    assert_eq!(
        to_hex(&Sha256::digest(&input)),
        "68f12c1e9aa7a0351750030e55a77a6662bb62d84c3a116e9f80084244313e31"
    );
    assert_eq!(document["expected_digest"], to_hex(&Sha256::digest(&input)));
}

#[test]
fn jcs_vector_is_domain_separated_and_canonical() {
    let vector: Value = serde_json::from_slice(
        &fs::read(corpus_root().join("v1/fixtures/valid/jcs-vector.json"))
            .expect("JCS vector reads"),
    )
    .expect("JCS vector parses");
    let bytes = canonicalize_jcs("approval", 1, &vector["document"]["input"])
        .expect("bounded JCS input canonicalizes");
    assert_eq!(to_hex(&bytes), vector["document"]["expected_hex"]);
    assert_eq!(
        to_hex(&Sha256::digest(&bytes)),
        "262faad7d219c868992828da29beb28a646a028afe64a8ff4a1d1108f1dc659c"
    );
}

#[test]
fn jcs_rejects_out_of_domain_numbers() {
    let value = serde_json::json!({"unsafe": 9_007_199_254_740_992_u64});
    assert!(matches!(
        canonicalize_jcs("claim", 1, &value),
        Err(ContractError::NumericDomain { .. })
    ));
}

#[test]
fn every_fixture_has_the_declared_behavior() {
    let verified = Corpus::open(corpus_root())
        .expect("corpus opens")
        .verify()
        .expect("corpus verifies");
    for path in verified.fixture_paths() {
        let outcome = verified.validate_fixture(&path);
        assert_eq!(outcome.actual(), outcome.expected(), "{}", path.display());
    }
}
