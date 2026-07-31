//! The WDBX v1 on-disk format: records, segments, and the manifest.
//!
//! Ported from `src/features/wdbx/persistence.zig`, `persistence_parse.zig` and
//! `segments.zig`. Specified in `tests/golden/wdbx-format.md`, which was derived
//! from a census of the live store rather than from the source alone.
//!
//! **This must read existing data.** The store at `~/.abi/` holds ~300 segments
//! and ~180 MB of the user's real completions and embeddings; a
//! format-incompatible reader silently orphans all of it. Every tolerance below
//! is there because the real data or the Zig serializer requires it, not for
//! defensiveness:
//!
//! - A segment may contain only its magic line. Several real ones do.
//! - The final line may be torn by an interrupted write, and must be dropped
//!   rather than failing the whole segment.
//! - `hash` and `prev_hash` are each either a JSON array of 32 integers **or** a
//!   32-character string. Both are `[32]u8` in Zig and written identically, but
//!   Zig's stringify emits a byte array as a string when the bytes are valid
//!   UTF-8. A digest never is; an all-zero genesis `prev_hash` always is. So a
//!   reader that fixes one encoding per field fails on the genesis block of every
//!   segment — 40 of them in the observed store.
//! - All six record types must be handled, though the live store contains three.
//! - An optional `# checksum:<hex>` trailer may follow the records.

pub use crate::hash::{FormatError, HASH_LEN, Hash, Result};
pub use crate::manifest::{MANIFEST_HEADER, Manifest, StorePaths};
pub use crate::record::{
    BlockRecord, KvRecord, Record, SpatialRecord, TemporalKind, TemporalRecord, VectorRecord,
};
pub use crate::segment::{
    CHECKSUM_PREFIX, MIRROR_EPOCH_HEADER, SEGMENT_HEADER, Segment, parse_segment, read_segment,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::hex_digest;
    use sha2::{Digest as _, Sha256};
    use std::path::PathBuf;

    #[test]
    fn genesis_hash_is_all_zero_and_reports_itself() {
        assert!(Hash::GENESIS.is_genesis());
        assert_eq!(Hash::GENESIS.bytes(), &[0u8; HASH_LEN]);
        assert_eq!(Hash::GENESIS.to_hex(), "0".repeat(64));
        assert!(!Hash([1; HASH_LEN]).is_genesis());
    }

    #[test]
    fn hash_reads_the_array_encoding() {
        let value = serde_json::json!((0..32).collect::<Vec<u8>>());
        let hash = Hash::from_json(&value, "hash").expect("array form");
        assert_eq!(hash.bytes()[0], 0);
        assert_eq!(hash.bytes()[31], 31);
    }

    #[test]
    fn hash_reads_the_string_encoding_used_by_genesis_blocks() {
        // The case a from-scratch reader misses. Zig emits this for every
        // all-zero prev_hash because zero bytes are valid UTF-8.
        let zeros: String = std::iter::repeat_n('\u{0}', HASH_LEN).collect();
        let hash =
            Hash::from_json(&serde_json::Value::String(zeros), "prev_hash").expect("string form");
        assert_eq!(hash, Hash::GENESIS);
    }

    #[test]
    fn hash_reads_a_non_zero_string_encoding() {
        // Any byte sequence that is valid UTF-8 takes the string form.
        let text: String = (1u8..=32).map(char::from).collect();
        let hash = Hash::from_json(&serde_json::Value::String(text), "hash").expect("string form");
        assert_eq!(hash.bytes()[0], 1);
        assert_eq!(hash.bytes()[31], 32);
    }

    #[test]
    fn hash_rejects_the_wrong_length_in_either_encoding() {
        let short_array = serde_json::json!([1, 2, 3]);
        assert!(matches!(
            Hash::from_json(&short_array, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
        let short_string = serde_json::Value::String("abc".to_string());
        assert!(matches!(
            Hash::from_json(&short_string, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
    }

    #[test]
    fn hash_rejects_out_of_range_values() {
        let too_big = serde_json::json!((0..32).map(|_| 300).collect::<Vec<u32>>());
        assert!(matches!(
            Hash::from_json(&too_big, "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
        // A codepoint above U+00FF cannot have come from one original byte.
        let wide: String = std::iter::repeat_n('世', HASH_LEN).collect();
        assert!(matches!(
            Hash::from_json(&serde_json::Value::String(wide), "hash"),
            Err(FormatError::InvalidHash { .. })
        ));
    }

    #[test]
    fn latin1_range_characters_decode_to_their_byte_values() {
        // Not an error case, despite looking like one: Zig wrote raw byte 0xE9 as
        // the two-byte UTF-8 encoding of U+00E9, so reading that char back as 233
        // is the correct round-trip. Every byte in 0x80..=0xFF behaves this way.
        let text: String = std::iter::repeat_n('é', HASH_LEN).collect();
        let hash = Hash::from_json(&serde_json::Value::String(text), "prev_hash")
            .expect("U+00E9 is byte 233");
        assert_eq!(hash.bytes(), &[233u8; HASH_LEN]);
    }

    #[test]
    fn hash_serializes_as_an_array_so_it_always_round_trips() {
        // The string form only works for UTF-8-valid bytes; the array form
        // always works, so writing should use it unconditionally.
        let hash = Hash([200; HASH_LEN]);
        let text = serde_json::to_string(&hash).expect("serialize");
        assert!(text.starts_with('['));
        let back = Hash::from_json(&serde_json::from_str(&text).expect("parse"), "hash")
            .expect("round trip");
        assert_eq!(back, hash);
    }

    #[test]
    fn parses_a_kv_record_keeping_the_value_opaque() {
        let line = r#"{"type":"kv","key":"completion:1","value":"{\"kind\":\"completion\"}"}"#;
        let Record::Kv(record) = Record::parse(line).expect("parses") else {
            panic!("expected a kv record");
        };
        assert_eq!(record.key, "completion:1");
        // Double-encoded: still a string, not parsed.
        assert_eq!(record.value, r#"{"kind":"completion"}"#);
    }

    #[test]
    fn parses_a_kv_record_whose_value_is_not_json() {
        let line = r#"{"type":"kv","key":"k","value":"not json at all"}"#;
        let Record::Kv(record) = Record::parse(line).expect("parses") else {
            panic!("expected a kv record");
        };
        assert_eq!(record.value, "not json at all");
    }

    #[test]
    fn parses_a_vector_record_of_any_dimension() {
        for dims in [1usize, 3, 32, 128] {
            let values: Vec<f32> = (0..dims)
                .map(|i| f32::from(u8::try_from(i).unwrap_or(0)))
                .collect();
            let line = serde_json::json!({"type":"vector","id":7,"values":values}).to_string();
            let Record::Vector(record) = Record::parse(&line).expect("parses") else {
                panic!("expected a vector record");
            };
            assert_eq!(record.id, 7);
            assert_eq!(record.values.len(), dims);
        }
    }

    #[test]
    fn parses_a_block_with_either_hash_encoding() {
        let zeros: String = std::iter::repeat_n('\u{0}', HASH_LEN).collect();
        let line = serde_json::json!({
            "type":"block",
            "hash": (0..32).collect::<Vec<u8>>(),
            "prev_hash": zeros,
            "timestamp_ms": 1_753_000_000_000_i64,
            "sequence": 0,
            "profile": "abi",
            "query_id": 1,
            "response_id": 2,
            "metadata": "m"
        })
        .to_string();
        let Record::Block(block) = Record::parse(&line).expect("parses") else {
            panic!("expected a block record");
        };
        assert!(block.prev_hash.is_genesis());
        assert_eq!(block.timestamp_ms, 1_753_000_000_000);
        assert_eq!(block.profile, "abi");
    }

    #[test]
    fn parses_spatial_and_temporal_records() {
        let spatial = r#"{"type":"spatial","id":7,"x":1.5,"y":-2.5,"z":0.0,"payload":"p"}"#;
        let Record::Spatial(record) = Record::parse(spatial).expect("parses") else {
            panic!("expected a spatial record");
        };
        assert_eq!(record.id, 7);
        assert!((record.x - 1.5).abs() < f32::EPSILON);

        for (line, kind) in [
            (r#"{"type":"temporal_node","id":1}"#, TemporalKind::Node),
            (
                r#"{"type":"temporal_edge","from":1,"to":2}"#,
                TemporalKind::Edge,
            ),
        ] {
            let Record::Temporal(record) = Record::parse(line).expect("parses") else {
                panic!("expected a temporal record");
            };
            assert_eq!(record.kind, kind);
        }
    }

    #[test]
    fn rejects_an_unknown_record_type() {
        let err = Record::parse(r#"{"type":"nope"}"#).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnknownRecordType {
                found: "nope".to_string()
            }
        );
    }

    #[test]
    fn reports_a_missing_field_by_name() {
        let err = Record::parse(r#"{"type":"kv","key":"k"}"#).unwrap_err();
        assert_eq!(
            err,
            FormatError::MissingField {
                record: "kv",
                field: "value"
            }
        );
    }

    #[test]
    fn record_type_names_round_trip() {
        assert_eq!(
            Record::Kv(KvRecord {
                key: String::new(),
                value: String::new()
            })
            .type_name(),
            "kv"
        );
        assert_eq!(
            Record::Temporal(TemporalRecord {
                kind: TemporalKind::Edge,
                fields: serde_json::Map::new()
            })
            .type_name(),
            "temporal_edge"
        );
    }

    fn segment_path() -> PathBuf {
        PathBuf::from("wdbx.seg.0.jsonl")
    }

    #[test]
    fn parses_a_segment_containing_only_its_magic_line() {
        // Several real segments look exactly like this.
        let segment = parse_segment(3, &segment_path(), "# ABI-WDBX v1\n").expect("parses");
        assert!(segment.records.is_empty());
        assert!(!segment.truncated_tail);
        assert_eq!(segment.epoch, 3);
    }

    #[test]
    fn rejects_a_segment_with_a_bad_magic_line() {
        let err = parse_segment(0, &segment_path(), "# WRONG\n{}\n").unwrap_err();
        assert!(matches!(err, FormatError::InvalidHeader { .. }));
    }

    #[test]
    fn rejects_an_empty_segment_file() {
        // No magic line at all is corruption, not emptiness.
        assert!(matches!(
            parse_segment(0, &segment_path(), "").unwrap_err(),
            FormatError::InvalidHeader { .. }
        ));
    }

    #[test]
    fn drops_a_torn_final_line_and_reports_it() {
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}\n",
            "{\"type\":\"kv\",\"key\":\"b\",\"valu"
        );
        let segment = parse_segment(0, &segment_path(), content).expect("parses");
        assert_eq!(segment.records.len(), 1);
        assert!(
            segment.truncated_tail,
            "an interrupted write must be reported, not silently swallowed"
        );
    }

    #[test]
    fn corruption_that_is_not_the_final_line_is_an_error() {
        // The distinction that keeps "tolerate truncation" from becoming
        // "ignore all corruption".
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"valu\n",
            "{\"type\":\"kv\",\"key\":\"b\",\"value\":\"2\"}\n"
        );
        assert!(parse_segment(0, &segment_path(), content).is_err());
    }

    #[test]
    fn a_complete_final_line_is_not_treated_as_torn() {
        // No trailing newline, but the line parses — so it is complete.
        let content = concat!(
            "# ABI-WDBX v1\n",
            "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}"
        );
        let segment = parse_segment(0, &segment_path(), content).expect("parses");
        assert_eq!(segment.records.len(), 1);
        assert!(!segment.truncated_tail);
    }

    #[test]
    fn reads_the_optional_checksum_trailer() {
        let body = "{\"type\":\"kv\",\"key\":\"a\",\"value\":\"1\"}\n";
        let checksum = hex_digest(Sha256::digest(body.as_bytes()));
        let content = format!("{SEGMENT_HEADER}\n{body}{CHECKSUM_PREFIX}{checksum}\n");
        let segment = parse_segment(0, &segment_path(), &content).expect("parses");
        assert_eq!(segment.checksum.as_deref(), Some(checksum.as_str()));
        assert_eq!(segment.records.len(), 1);
    }

    #[test]
    fn parses_a_manifest() {
        let manifest =
            Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=301\nactive=0,1,2,299\n")
                .expect("parses");
        assert_eq!(manifest.next_epoch, 301);
        assert_eq!(manifest.active, [0, 1, 2, 299]);
    }

    #[test]
    fn manifest_active_is_sorted_and_deduplicated() {
        // Ascending order is what makes epoch replay produce the right shadowing.
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=5\nactive=3,1,2,1\n")
            .expect("parses");
        assert_eq!(manifest.active, [1, 2, 3]);
    }

    #[test]
    fn manifest_tolerates_a_sparse_active_list() {
        // Compaction leaves gaps; assuming density would read collected segments.
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nnext_epoch=10\nactive=0,4,9\n")
            .expect("parses");
        assert_eq!(manifest.active, [0, 4, 9]);
    }

    #[test]
    fn manifest_rejects_a_bad_magic_line() {
        assert!(matches!(
            Manifest::parse("# WRONG\nnext_epoch=1\n").unwrap_err(),
            FormatError::InvalidManifest { .. }
        ));
    }

    #[test]
    fn manifest_ignores_unknown_keys_for_forward_compatibility() {
        // Refusing to open the store because a newer writer added a key would be
        // worse than ignoring it.
        let manifest = Manifest::parse(
            "# ABI-WDBX-SEGMENTS v1\nnext_epoch=2\nactive=0,1\nfuture_key=whatever\n",
        )
        .expect("parses");
        assert_eq!(manifest.next_epoch, 2);
        assert_eq!(manifest.active, [0, 1]);
    }

    #[test]
    fn manifest_infers_next_epoch_when_absent() {
        let manifest = Manifest::parse("# ABI-WDBX-SEGMENTS v1\nactive=0,1,2\n").expect("parses");
        assert_eq!(manifest.next_epoch, 3);
    }

    #[test]
    fn manifest_renders_in_the_readable_form() {
        let manifest = Manifest {
            next_epoch: 4,
            active: vec![0, 2, 3],
        };
        assert_eq!(
            manifest.render(),
            "# ABI-WDBX-SEGMENTS v1\nnext_epoch=4\nactive=0,2,3\n"
        );
        assert_eq!(
            Manifest::parse(&manifest.render()).expect("parses"),
            manifest
        );
    }

    #[test]
    fn empty_manifest_renders_and_reparses() {
        let manifest = Manifest::empty();
        assert_eq!(
            Manifest::parse(&manifest.render()).expect("parses"),
            manifest
        );
    }

    #[test]
    fn store_paths_match_the_observed_layout() {
        let paths = StorePaths::new("/home/u/.abi");
        assert_eq!(paths.index(), PathBuf::from("/home/u/.abi/wdbx"));
        assert_eq!(
            paths.manifest(),
            PathBuf::from("/home/u/.abi/wdbx.manifest")
        );
        assert_eq!(
            paths.mirror_epoch(),
            PathBuf::from("/home/u/.abi/wdbx.mirror-epoch")
        );
        assert_eq!(
            paths.segment(42),
            PathBuf::from("/home/u/.abi/wdbx.seg.42.jsonl")
        );
    }

    #[test]
    fn a_missing_manifest_is_an_empty_store_not_an_error() {
        // What a first run looks like.
        let dir = abi_foundation::temp_path::temp_file_path("abi_wdbx_no_manifest", "d");
        let paths = StorePaths::new(&dir);
        assert_eq!(paths.read_manifest().expect("no error"), Manifest::empty());
    }

    #[test]
    fn mirror_epoch_defaults_legacy_to_zero_and_rejects_corruption() {
        let dir = abi_foundation::temp_path::temp_file_path("abi_wdbx_mirror_epoch", "d");
        let paths = StorePaths::new(&dir);
        assert_eq!(paths.read_mirror_epoch().expect("legacy default"), 0);

        std::fs::create_dir_all(&dir).expect("fixture directory");
        std::fs::write(paths.mirror_epoch(), "not provenance\n").expect("corrupt sidecar");
        assert!(matches!(
            paths.read_mirror_epoch(),
            Err(FormatError::InvalidManifest { .. })
        ));
    }
}
