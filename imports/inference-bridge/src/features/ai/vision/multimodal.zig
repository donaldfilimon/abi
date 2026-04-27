//! Multi-Modal Fusion Architecture
//!
//! This module implements cross-modal learning and fusion capabilities for combining
//! vision and language understanding. It supports CLIP-style contrastive learning,
//! cross-attention fusion, and unified embedding spaces.
//!
//! ## Architecture Components
//!
//! 1. **ContrastiveLoss**: InfoNCE loss for aligning image and text embeddings
//! 2. **CrossAttention**: Cross-modal attention between vision and language
//! 3. **MultiModalEncoder**: Unified encoder for joint understanding
//! 4. **CLIPModel**: Complete CLIP-style contrastive learning model
//!
//! ## Submodules
//!
//! - `preprocessing` — Configuration and unified embedding space management
//! - `encoders` — Text embedding and transformer-based text encoders
//! - `fusion` — Cross-attention, fusion blocks, contrastive loss, CLIP model
//!
//! ## Usage
//!
//! ```zig
//! const mm = @import("multimodal.zig");
//!
//! // Create CLIP-style model
//! var clip = try mm.CLIPModel.init(allocator, .{
//!     .vision_config = .base(224, 16),
//!     .text_hidden_size = 512,
//!     .projection_dim = 512,
//! });
//! defer clip.deinit();
//!
//! // Compute similarity
//! const similarity = try clip.computeSimilarity(image_embedding, text_embedding);
//! ```

// ============================================================================
// Submodules
// ============================================================================

/// Configuration and unified multi-modal embedding space
pub const preprocessing = @import("multimodal/preprocessing.zig");

/// Text embedding and transformer-based text encoders
pub const encoders = @import("multimodal/encoders.zig");

/// Cross-modal attention, fusion blocks, contrastive loss, and CLIP model
pub const fusion = @import("multimodal/fusion.zig");

// ============================================================================
// Re-exports (preserve original public API)
// ============================================================================

/// Configuration for multi-modal fusion
pub const MultiModalConfig = preprocessing.MultiModalConfig;

/// Unified embedding space for multiple modalities
pub const UnifiedEmbeddingSpace = preprocessing.UnifiedEmbeddingSpace;

/// Text embedding layer (token + position embeddings)
pub const TextEmbedding = encoders.TextEmbedding;

/// Transformer-based text encoder with projection
pub const TextEncoder = encoders.TextEncoder;

/// InfoNCE contrastive loss for aligning embeddings
pub const ContrastiveLoss = fusion.ContrastiveLoss;

/// Cross-attention layer for fusing vision and language
pub const CrossAttention = fusion.CrossAttention;

/// Fusion block combining vision and language through cross-attention
pub const FusionBlock = fusion.FusionBlock;

/// Complete CLIP-style contrastive learning model
pub const CLIPModel = fusion.CLIPModel;

// ============================================================================
// Tests
// ============================================================================

test {
    _ = preprocessing;
    _ = encoders;
    _ = fusion;
}
