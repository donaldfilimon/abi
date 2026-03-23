//! Stubbed WDBX memory namespace.

pub const block_chain = @import("../stubs/misc.zig").block_chain;
const semantic = @import("../semantic_store/stub.zig");

pub const BlockChain = block_chain.BlockChain;
pub const BlockConfig = block_chain.BlockConfig;
pub const ConversationBlock = block_chain.ConversationBlock;
pub const ProfileTag = block_chain.ProfileTag;
pub const RoutingWeights = block_chain.RoutingWeights;
pub const IntentCategory = block_chain.IntentCategory;
pub const PolicyFlags = block_chain.PolicyFlags;
pub const MvccStore = block_chain.MvccStore;

pub const MemoryBlock = semantic.MemoryBlock;
pub const MemoryBlockConfig = semantic.MemoryBlockConfig;
pub const WeightInputs = semantic.WeightInputs;
pub const Lineage = semantic.Lineage;
pub const InfluenceTrace = semantic.InfluenceTrace;
pub const RetrievalHit = semantic.RetrievalHit;
