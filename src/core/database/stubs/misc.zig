const std = @import("std");
const parallel_mod = @import("parallel.zig");

pub const cli = struct {
    pub fn run(_: std.mem.Allocator, _: []const [:0]const u8) !void {
        return error.DatabaseDisabled;
    }
};

pub const parallel_search = struct {
    pub const ParallelSearchConfig = parallel_mod.ParallelSearchConfig;
    pub const ParallelSearchExecutor = parallel_mod.ParallelSearchExecutor;
    pub const ParallelBeamState = parallel_mod.ParallelBeamState;
    pub const ParallelWorkQueue = parallel_mod.ParallelWorkQueue;
    pub const BatchSearchResult = parallel_mod.BatchSearchResult;
    pub const ParallelSearchStats = parallel_mod.ParallelSearchStats;
    pub const batchCosineDistances = parallel_mod.batchCosineDistances;
};
pub const database = struct {
    pub const DatabaseError = error{
        DuplicateId,
        VectorNotFound,
        InvalidDimension,
        PoolExhausted,
        PersistenceError,
        ConcurrencyError,
        FeatureDisabled,
        DatabaseDisabled,
    };
    pub const Database = struct {
        pub fn init(_: std.mem.Allocator) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const db_helpers = struct {};
pub const http = struct {};

// ============================================================================
// Full-text search stubs
// ============================================================================

pub const fulltext = struct {
    pub const InvertedIndex = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const Bm25Config = struct {
        k1: f32 = 1.2,
        b: f32 = 0.75,
    };
    pub const TokenizerConfig = struct {};
    pub const TextSearchResult = struct {
        id: u64 = 0,
        score: f32 = 0.0,
    };
    pub const QueryParser = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};

// ============================================================================
// Hybrid search stubs
// ============================================================================

pub const hybrid = struct {
    pub const HybridSearchEngine = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const HybridConfig = struct {};
    pub const HybridResult = struct {
        id: u64 = 0,
        score: f32 = 0.0,
    };
    pub const FusionMethod = enum { rrf, weighted, cascade };
};

// ============================================================================
// Filter stubs
// ============================================================================

pub const filter = struct {
    pub const FilterBuilder = struct {
        pub fn init() @This() {
            return .{};
        }
    };
    pub const FilterExpression = struct {};
    pub const FilterCondition = struct {};
    pub const FilterOperator = enum { eq, ne, gt, gte, lt, lte, in_set, contains };
    pub const MetadataValue = union(enum) {
        string: []const u8,
        int: i64,
        float: f64,
        boolean: bool,
    };
    pub const MetadataStore = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const FilteredSearch = struct {};
    pub const FilteredResult = struct {
        id: u64 = 0,
        score: f32 = 0.0,
    };
};

// ============================================================================
// Batch stubs
// ============================================================================

pub const batch = struct {
    pub const BatchProcessor = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const BatchConfig = struct {};
    pub const BatchRecord = struct {};
    pub const BatchResult = struct {
        success_count: usize = 0,
        error_count: usize = 0,
    };
    pub const BatchWriter = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const BatchOperationBuilder = struct {
        pub fn init() @This() {
            return .{};
        }
    };
    pub const BatchImporter = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ImportFormat = enum { csv, json, zon };
};

// ============================================================================
// Clustering stubs
// ============================================================================

pub const clustering = struct {
    pub const KMeans = struct {
        pub fn init(_: std.mem.Allocator, _: usize, _: usize) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ClusterStats = struct {
        inertia: f64 = 0.0,
        iterations: usize = 0,
    };
    pub const FitOptions = struct {};
    pub const FitResult = struct {};
    pub fn euclideanDistance(_: []const f32, _: []const f32) f32 {
        return 0.0;
    }
    pub fn cosineSimilarity(_: []const f32, _: []const f32) f32 {
        return 0.0;
    }
    pub fn silhouetteScore(_: anytype, _: anytype) f32 {
        return 0.0;
    }
    pub fn elbowMethod(_: anytype) []const f64 {
        return &.{};
    }
};

// ============================================================================
// Quantization stubs
// ============================================================================

pub const quantization = struct {
    pub const ScalarQuantizer = struct {
        pub fn init(_: u8) @This() {
            return .{};
        }
    };
    pub const ProductQuantizer = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const QuantizationError = error{
        InvalidDimension,
        CodebookNotTrained,
        DatabaseDisabled,
    };
};

// ============================================================================
// GPU acceleration stubs
// ============================================================================

pub const gpu_accel = struct {
    pub const GpuAccelerator = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const GpuAccelConfig = struct {};
    pub const GpuAccelStats = struct {};
};

// ============================================================================
// Formats stubs
// ============================================================================

pub const formats = struct {
    pub const UnifiedFormat = struct {
        pub fn deinit(_: *@This()) void {}
    };
    pub const unified = struct {
        pub const UnifiedFormatBuilder = struct {
            pub fn init() @This() {
                return .{};
            }
        };
    };
    pub const FormatHeader = struct {};
    pub const FormatFlags = struct {};
    pub const TensorDescriptor = struct {};
    pub const DataType = enum { f32, f16, bf16, i32, i16, i8, u8, q4_0, q4_1, q8_0 };
    pub const Converter = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ConversionOptions = struct {};
    pub const TargetFormat = enum { gguf, safetensors, zon };
    pub const CompressionType = enum { none, lz4, zstd };
    pub const StreamingWriter = struct {};
    pub const StreamingReader = struct {};
    pub const MappedFile = struct {};
    pub const MemoryCursor = struct {};
    pub const VectorDatabase = struct {};
    pub const VectorRecord = struct {};
    pub const SearchResult = struct {};
    pub const ZonFormat = struct {};
    pub const ZonDatabase = struct {};
    pub const ZonRecord = struct {};
    pub const ZonDatabaseConfig = struct {};
    pub const ZonDistanceMetric = enum { cosine, euclidean, dot_product };
    pub const GgufTensorType = enum { f32, f16, q4_0, q4_1, q8_0 };
    pub fn fromGguf(_: std.mem.Allocator, _: anytype) !UnifiedFormat {
        return error.DatabaseDisabled;
    }
    pub fn toGguf(_: anytype, _: std.mem.Allocator) !void {
        return error.DatabaseDisabled;
    }
    pub fn exportToZon(_: anytype, _: std.mem.Allocator) !void {
        return error.DatabaseDisabled;
    }
    pub fn importFromZon(_: std.mem.Allocator, _: anytype) !void {
        return error.DatabaseDisabled;
    }
};

// ============================================================================
// Storage stubs (unified storage API)
// ============================================================================

pub const storage = struct {
    pub const FileHeader = struct {};
    pub const FileFooter = struct {};
    pub const BloomFilter = struct {};
    pub const Crc32 = struct {};
    pub const StorageV2Config = struct {};
    pub fn saveDatabaseV2(_: anytype, _: anytype) !void {
        return error.DatabaseDisabled;
    }
    pub fn loadDatabaseV2(_: std.mem.Allocator, _: anytype) !void {
        return error.DatabaseDisabled;
    }
};

// ============================================================================
// BlockChain stubs
// ============================================================================

pub const block_chain = struct {
    pub const BlockChain = struct {
        current_head: ?u64 = null,

        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
        pub fn addBlock(_: *@This(), _: BlockConfig) !u64 {
            return error.DatabaseDisabled;
        }
        pub fn getBlock(_: *const @This(), _: u64) ?ConversationBlock {
            return null;
        }
    };
    pub const ConversationBlock = struct {
        hash: [32]u8 = .{0} ** 32,
        previous_hash: [32]u8 = .{0} ** 32,
        parent_block_id: ?u64 = null,

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    pub const BlockConfig = struct {
        query_embedding: []const f32 = &.{},
        response_embedding: ?[]const f32 = null,
        profile_tag: ProfileTag = .{ .primary_profile = .abbey },
        routing_weights: RoutingWeights = .{},
        intent: IntentCategory = .general,
        risk_score: f32 = 0.0,
        policy_flags: PolicyFlags = .{},
        parent_block_id: ?u64 = null,
        previous_hash: [32]u8 = .{0} ** 32,
    };
    pub const BlockChainConfig = struct {};
    pub const BlockChainError = error{ DatabaseDisabled, InvalidBlock, ChainCorrupted };
    pub const ProfileTag = struct {
        primary_profile: ProfileType,
        blend_coefficient: f32 = 0.0,
        secondary_profile: ?ProfileType = null,

        pub const ProfileType = enum {
            abbey,
            aviva,
            abi,
            blended,
        };
    };
    pub const RoutingWeights = struct {
        abbey_weight: f32 = 0.0,
        aviva_weight: f32 = 0.0,
        abi_weight: f32 = 0.0,

        pub fn getPrimaryProfile(_: @This()) ProfileTag.ProfileType {
            return .abbey;
        }
        pub fn getBlendCoefficient(_: @This()) f32 {
            return 0.0;
        }
    };
    pub const IntentCategory = enum {
        general,
        empathy_seeking,
        technical_problem,
        factual_inquiry,
        creative_generation,
        policy_check,
        safety_critical,
    };
    pub const PolicyFlags = struct {
        is_safe: bool = true,
        requires_moderation: bool = false,
        sensitive_topic: bool = false,
        pii_detected: bool = false,
        violation_details: ?[]const u8 = null,
    };
    pub const MvccStore = struct {
        pub fn init(_: std.mem.Allocator) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};

// ============================================================================
// Distributed stubs
// ============================================================================

pub const distributed = struct {
    pub const ShardManager = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ShardConfig = struct {};
    pub const ShardKey = struct {};
    pub const ShardManagerError = error{ DatabaseDisabled, ShardNotFound };
    pub const HashRing = struct {};
    pub const LoadStats = struct {};
    pub const BlockExchangeManager = struct {};
    pub const BlockExchangeError = error{ DatabaseDisabled, ExchangeFailed };
    pub const SyncState = enum { idle, syncing, complete };
    pub const VersionVector = struct {};
    pub const VersionComparison = enum { equal, before, after, concurrent };
    pub const SyncRequest = struct {};
    pub const SyncResponse = struct {};
    pub const BlockConflict = struct {};
    pub const DistributedBlockChain = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const DistributedBlockChainConfig = struct {};
    pub const DistributedBlockChainError = error{ DatabaseDisabled, ConsensusFailure };
    pub const DistributedConfig = struct {};
    pub const Context = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };

    // Cluster bootstrap & membership (parity with distributed/cluster.zig)
    pub const NodeRole = enum { primary, replica, observer };
    pub const NodeState = enum { joining, active, draining, failed, removed };
    pub const TransportType = enum { tcp, tls, thunderbolt, auto };
    pub const NodeInfo = struct {
        node_id: u64 = 0,
        address: [64]u8 = [_]u8{0} ** 64,
        address_len: u8 = 0,
        port: u16 = 9200,
        role: NodeRole = .primary,
        state: NodeState = .joining,
        transport: TransportType = .tcp,
        last_heartbeat: i64 = 0,
        shard_count: u32 = 0,
        vector_count: u64 = 0,
    };
    pub const ClusterConfig = struct {
        node_id: u64 = 0,
        listen_port: u16 = 9200,
        transport: TransportType = .tcp,
        replication_factor: u8 = 3,
        heartbeat_interval_ms: u32 = 1000,
        failure_timeout_ms: u32 = 5000,
        auto_rebalance: bool = true,
        max_nodes: u16 = 256,
        bootstrap_peers: [512]u8 = [_]u8{0} ** 512,
        bootstrap_peers_len: u16 = 0,
    };
    pub const ClusterStatus = struct {
        node_count: u16 = 0,
        active_nodes: u16 = 0,
        total_shards: u32 = 0,
        total_vectors: u64 = 0,
        replication_health: f32 = 0.0,
        leader_id: u64 = 0,
        self_role: NodeRole = .primary,
    };
    pub const ClusterError = error{ BufferTooSmall, InvalidMessage, PeerNotFound };
    pub const MessageType = enum(u8) { heartbeat = 1, join_request = 2, join_response = 3, shard_transfer = 4, leader_announce = 5 };
    pub const ClusterMessage = struct {
        msg_type: MessageType = .heartbeat,
        sender_id: u64 = 0,
        payload: [1024]u8 = [_]u8{0} ** 1024,
        payload_len: u32 = 0,
        pub fn serialize(_: *const @This(), _: []u8) ClusterError!usize {
            return error.BufferTooSmall;
        }
        pub fn deserialize(_: []const u8) ClusterError!@This() {
            return error.InvalidMessage;
        }
    };
    pub const PeerAddress = struct {
        host: [64]u8 = [_]u8{0} ** 64,
        host_len: u8 = 0,
        port: u16 = 9200,
        pub fn fromString(_: []const u8) PeerAddress {
            return .{};
        }
    };
    pub const ClusterManager = struct {
        pub fn init(_: std.mem.Allocator, _: ClusterConfig) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
        pub fn start(_: *@This()) ClusterError!void {
            return error.PeerNotFound;
        }
        pub fn stop(_: *@This()) void {}
        pub fn onHeartbeat(_: *@This(), _: u64, _: u64) void {}
        pub fn checkHealth(_: *@This()) void {}
        pub fn addPeer(_: *@This(), _: NodeInfo) ClusterError!void {
            return error.PeerNotFound;
        }
        pub fn removePeer(_: *@This(), _: u64) void {}
        pub fn getStatus(_: *const @This()) ClusterStatus {
            return .{};
        }
        pub fn activeNodeCount(_: *const @This()) u16 {
            return 0;
        }
        pub fn electLeader(_: *@This()) void {}
        pub fn serializeHeartbeat(_: *const @This()) ClusterError!ClusterMessage {
            return error.InvalidMessage;
        }
    };
};

// ============================================================================
// DiskANN stubs
// ============================================================================

pub const diskann = struct {
    pub const DiskANNIndex = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const DiskANNConfig = struct {};
    pub const PQCodebook = struct {};
    pub const IndexStats = struct {};
    pub const VamanaIndex = struct {
        pub fn init(_: std.mem.Allocator, _: u32, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const VamanaConfig = struct {
        max_degree: u32 = 64,
        alpha: f32 = 1.2,
        build_list_size: u32 = 128,
        search_list_size: u32 = 64,
        beam_width: u32 = 4,
    };
    pub const VamanaSearchResult = struct {
        id: u32,
        distance: f32,
    };
};

// ============================================================================
// ScaNN stubs
// ============================================================================

pub const scann = struct {
    pub const ScaNNIndex = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ScaNNConfig = struct {};
    pub const QuantizationType = enum { scalar, product, avq };
    pub const AVQCodebook = struct {};
    pub const IndexStats = struct {};
};

// ============================================================================
// Parallel HNSW stubs
// ============================================================================

pub const parallel_hnsw = struct {
    pub const ParallelHnswBuilder = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.DatabaseDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const ParallelBuildConfig = struct {};
    pub const ParallelBuildStats = struct {};
};

// ============================================================================
// HNSW / index / search_state / distance_cache namespace stubs
// ============================================================================

pub const hnsw = struct {};
pub const index = struct {};
pub const search_state = struct {};
pub const distance_cache = struct {};

// ============================================================================
// Time namespace stub
// ============================================================================

pub const time = struct {};

test {
    std.testing.refAllDecls(@This());
}
