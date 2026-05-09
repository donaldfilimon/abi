const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

pub const llm_trainer = @import("llm_trainer.zig");
pub const self_learning = @import("self_learning.zig");
pub const vision_trainer = @import("vision_trainer.zig");
pub const multimodal_trainer = @import("multimodal_trainer.zig");
pub const distributed = @import("distributed.zig");

// Re-exports
pub const LlamaTrainer = llm_trainer.LlamaTrainer;
pub const LlmTrainingConfig = llm_trainer.LlmTrainingConfig;
pub const trainLlm = llm_trainer.trainLlm;

pub const SelfLearningSystem = self_learning.SelfLearningSystem;
pub const SelfLearningConfig = self_learning.SelfLearningConfig;

pub fn selfLearningConfigFromCore(core_cfg: config_module.TrainingConfig) SelfLearningConfig {
    return .{
        .enable_vision = core_cfg.enable_vision,
        .enable_video = core_cfg.enable_video,
        .enable_audio = core_cfg.enable_audio,
        .enable_all_modalities = core_cfg.enable_all_modalities,
        .enable_rlhf = true,
        .enable_documents = true,
        .replay_buffer_size = 10000,
        .batch_size = core_cfg.batch_size,
        .min_buffer_size = 100,
        .update_frequency = 64,
    };
}
pub const LearningExperience = self_learning.LearningExperience;
pub const ExperienceBuffer = self_learning.ExperienceBuffer;
pub const RewardModel = self_learning.RewardModel;
pub const SelfLearningVisionTrainer = self_learning.VisionTrainer;
pub const DocumentTrainer = self_learning.DocumentTrainer;
pub const ExperienceType = self_learning.ExperienceType;
pub const FeedbackType = self_learning.FeedbackType;
pub const DataKind = self_learning.DataKind;

pub const TrainableViTModel = vision_trainer.TrainableViTModel;
pub const TrainableViTConfig = vision_trainer.TrainableViTConfig;
pub const TrainableViTWeights = vision_trainer.TrainableViTWeights;
pub const TrainableViTLayerWeights = vision_trainer.TrainableViTLayerWeights;
pub const ViTActivationCache = vision_trainer.ViTActivationCache;
pub const VisionTrainingError = vision_trainer.VisionTrainingError;

pub const TrainableCLIPModel = multimodal_trainer.TrainableCLIPModel;
pub const CLIPTrainingConfig = multimodal_trainer.CLIPTrainingConfig;
pub const TrainableTextEncoderWeights = multimodal_trainer.TrainableTextEncoderWeights;
pub const TextTransformerLayerWeights = multimodal_trainer.TextTransformerLayerWeights;
pub const MultimodalTrainingError = multimodal_trainer.MultimodalTrainingError;

pub const DistributedConfig = distributed.DistributedConfig;
pub const DistributedTrainer = distributed.DistributedTrainer;
