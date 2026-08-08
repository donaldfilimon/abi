//! WDBX: the ABI framework's vector store.
//!
//! Step 4 of the Zig→Rust port. This crate now covers the **on-disk format**,
//! checkpoint publication and salvage, CRC-framed WAL recovery, and a
//! deterministic exact-search reference index plus layered HNSW graph. Durable
//! store integration, loopback REST, the reference-scoped cluster protocol,
//! deterministic compute selection/remote DOT reference transport, the 3-D
//! spatial index, reference quantization/Huffman/rANS/autoencoder codecs,
//! persona/spatial retrieval, additive HE, and the depth-limited DGHV SHE demo
//! are ported. The append-only MVCC chain and bounded multiway engine include
//! canonical export, resume, DOT, and WDBX persistence. These are reference
//! implementations, not production cryptography, SOTA compression, or a
//! complete-ruliad claim. TLS path validation is ported, but native TLS
//! termination is not linked; see `RUST-REWRITE-PLAN.md`.
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

pub mod ans;
pub mod cluster;
pub mod cluster_rpc;
pub mod codecs;
pub mod compression;
pub mod compute;
pub mod crypto_he;
pub mod durable;
pub mod entropy;
pub mod fhe;
pub mod format;
pub mod hash;
pub mod hnsw;
pub mod index;
pub mod manifest;
pub mod multiway;
pub mod mvcc;
pub mod net_line;
pub mod neural_compress;
pub mod persistence;
pub mod rate_limit;
pub mod record;
pub mod remote_compute;
pub mod rest;
pub mod retrieval;
pub mod segment;
pub mod segments;
pub mod spatial;
pub mod store;
pub mod temporal;
pub mod tls_config;
pub mod v2;
pub mod versioned;
pub mod wal;

pub use ans::{AnsEncoded, AnsError, AnsMode, ans_decode, ans_encode, ans_encode_order1};
pub use cluster::{
    AppendReply, Cluster, ClusterDataError, ClusterError, KvReadResult, LogEntry, MembershipChange,
    MembershipError, MembershipLedger, MembershipRecord, Node, NodeDescriptor, NodeState,
    RebalancePlan, RebalanceProgress, RebalanceStep, RepairAction, RepairPlan, ReplicaSearchHit,
    ReplicaTransport, ReplicationReceipt, Role, SignedMembershipRecord, TransportApplyError,
    TransportError, VoteReply, apply_append, apply_vote, build_rebalance_plan, read_kv_fanout,
    replicate_committed, resume_rebalance, search_fanout, sign_membership_record,
};
pub use cluster_rpc::{
    CLUSTER_RPC_TIMEOUT, ClusterAuth, ClusterDataRequest, ClusterDataResponse, ClusterKvState,
    ClusterPolicy, ClusterRpcServer, MAX_CLUSTER_DATA_FRAME_SIZE, RpcError, dial_append, dial_data,
    dial_shutdown, dial_vote, read_append_reply, read_data_reply, read_shutdown_reply,
    read_vote_reply,
};
pub use codecs::{
    AutoencoderArchitectureV1, AutoencoderArtifactV1, AutoencoderConfigV1, CodecError,
    CodecMetrics, CodecRecord, NormalizationParamsV1, PersistedAutoencoderV1, PqArtifactV1,
    PqConfigV1, ProductQuantizerV1, codec_source_digest, train_autoencoder_v1,
    train_product_quantizer_v1,
};
pub use compression::{CompressionError, Quantized, dequantize, max_error, quantize};
pub use compute::{
    Accelerator, Backend, Capability, CapabilityEvidence, CapabilityInvariantError,
    CapabilityState, ComputeError, CpuBackend, ScoredIndex, Selection, ane_hardware_present,
    baseline_capability_states, best_cpu_backend, capabilities, cosine, dot, norm, select,
    simd_lanes, top_k,
};
pub use crypto_he::{HE_MODULUS, HeCipher, HeError, HeKey, he_add};
pub use durable::{DurableError, DurableStore};
pub use entropy::{
    EntropyEncoded, EntropyError, EntropyMode, entropy_decode, entropy_encode, entropy_round_trip,
};
pub use fhe::{
    DghvCipher, DghvKeypair, DghvRng, REF_NOISE_BITS, REF_P_BITS, VERIFIED_MUL_DEPTH, dghv_add,
    dghv_decrypt, dghv_encrypt, dghv_keygen, dghv_mul,
};
pub use format::{
    BlockRecord, FormatError, Hash, KvRecord, Manifest, Record, Segment, SpatialRecord, StorePaths,
    TemporalKind, TemporalRecord, VectorRecord,
};
pub use hnsw::{HnswError, HnswIndex, VectorStorage};
pub use index::{ExactIndex, IndexError, SearchResult, cosine_similarity};
pub use multiway::{
    CausalEdge as MultiwayCausalEdge, ConfigError as MultiwayConfigError, MAX_MULTIWAY_RULES,
    MULTIWAY_EXPORT_KEY_LATEST, MULTIWAY_FORMAT_VERSION, MultiwayConfig, MultiwayEvent,
    MultiwayExportError, MultiwayMetrics, MultiwayPersistError, MultiwayResult, MultiwayState,
    ParseRuleError, Rule as MultiwayRule, Termination as MultiwayTermination, build_causal_edges,
    compute_multiway_metrics, export_multiway_dot, export_multiway_json,
    hash_payload as hash_multiway_payload, load_multiway_export, multiway_config_hash,
    multiway_export_hash_hex, parse_rule, persist_multiway, resume_multiway, run_multiway,
    validate_config as validate_multiway_config,
};
pub use mvcc::{MvccBlock, MvccChain, MvccError, MvccSnapshot};
pub use neural_compress::{Autoencoder, AutoencoderError};
pub use persistence::{flush, serialize_snapshot, write_snapshot};
pub use rate_limit::{RateLimitStats, RateLimiter};
pub use remote_compute::{
    ENDPOINT_ENV as REMOTE_COMPUTE_ENDPOINT_ENV,
    MAX_MESSAGE_SIZE as REMOTE_COMPUTE_MAX_MESSAGE_SIZE, RemoteError, dial_dot, dot_or_local,
    dot_or_local_at, endpoint as remote_compute_endpoint, local_dot, parse_endpoint_port,
    read_dot_reply, serve_once as serve_remote_compute_once,
};
pub use rest::{RestConfig, RestServer};
pub use retrieval::{
    RankedSpatialResult, RankedVector, hybrid_search_scoped, hybrid_search_with_persona,
    hybrid_spatial_search, without_vector,
};
pub use segments::{CompactionResult, SegmentError};
pub use spatial::{
    DistanceMetric, Point3D, SpatialIndex3D, SpatialRecord3D, SpatialSearchResult,
    calculate_distance, cosine_distance, euclidean_distance, manhattan_distance,
};
pub use store::{Snapshot, SnapshotStats};
pub use temporal::{
    HybridScorer, RankedNode, ScoreComponents, TemporalCausalGraph, hybrid_search, temporal_weight,
};
pub use tls_config::{TLS_CERT_ENV, TLS_KEY_ENV, TlsConfig};
pub use v2::{
    CausalHeads, CommittedTransaction, CompactionReport, ConflictSet, RecordId, SegmentCodecKind,
    SegmentCodecPolicy, V2Error, V2Mutation, V2Snapshot, V2Store, Version, VersionedSnapshot,
    open_versioned_read_only,
};
pub use versioned::{VersionedError, VersionedSearchResult, VersionedStats, VersionedStore};
pub use wal::{Recovered, RecoverySource, Wal, WalError};
