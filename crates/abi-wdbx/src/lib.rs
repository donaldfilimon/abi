//! WDBX: the ABI framework's vector store.
//!
//! Step 4 of the Zig→Rust port. This crate now covers the **on-disk format**,
//! checkpoint publication and salvage, CRC-framed WAL recovery, and a
//! deterministic exact-search reference index plus layered HNSW graph. Durable
//! store integration and the loopback REST surface are ported; the cluster
//! protocol, compression and homomorphic-encryption demos are still Zig; see
//! `RUST-REWRITE-PLAN.md`.
//!
//! ## Read-compatibility is a requirement, not a goal
//!
//! `~/.abi/` holds ~300 segments and ~180 MB of the user's real completions and
//! embeddings. A rewrite that cannot read them silently orphans all of it, so the
//! format came first and is specified in `tests/golden/wdbx-format.md` from a
//! census of the actual store rather than from the Zig source alone.
//!
//! That census found something the source does not make obvious: `hash` and
//! `prev_hash` are both `[32]u8` written with the same call, but Zig's JSON
//! stringify emits a byte array as a *string* when the bytes are valid UTF-8 and
//! as an *array* when they are not. A SHA-256 digest never is; an all-zero genesis
//! `prev_hash` always is. So each field appears in both encodings across the real
//! data — 4136 arrays and 40 strings for `prev_hash` — and a reader that fixes one
//! encoding per field fails on the genesis block of every segment.

pub mod cluster;
pub mod durable;
pub mod format;
pub mod hnsw;
pub mod index;
pub mod persistence;
pub mod rate_limit;
pub mod rest;
pub mod segments;
pub mod store;
pub mod temporal;
pub mod wal;

pub use cluster::{
    AppendReply, Cluster, ClusterError, LogEntry, Node, Role, VoteReply, apply_append, apply_vote,
};
pub use durable::{DurableError, DurableStore};
pub use format::{
    BlockRecord, FormatError, Hash, KvRecord, Manifest, Record, Segment, SpatialRecord, StorePaths,
    TemporalKind, TemporalRecord, VectorRecord,
};
pub use hnsw::{HnswError, HnswIndex, VectorStorage};
pub use index::{ExactIndex, IndexError, SearchResult, cosine_similarity};
pub use persistence::{flush, serialize_snapshot, write_snapshot};
pub use rate_limit::{RateLimitStats, RateLimiter};
pub use segments::{CompactionResult, SegmentError};
pub use store::{Snapshot, SnapshotStats};
pub use temporal::{
    HybridScorer, RankedNode, ScoreComponents, TemporalCausalGraph, hybrid_search, temporal_weight,
};
pub use wal::{Recovered, RecoverySource, Wal, WalError};
