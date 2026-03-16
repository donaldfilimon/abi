//! Advanced Cognition Wrapper
//!
//! Provides a flattened import path for the Abbey advanced cognition module.
//! The detailed implementation remains in `advanced/mod.zig` and its
//! sub-modules.

pub const AdvancedCognition = @import("advanced.zig").AdvancedCognition;
pub const CognitiveResult = @import("advanced.zig").CognitiveResult;
pub const CognitiveState = @import("advanced.zig").CognitiveState;
// Re-export other public symbols as needed
pub const TaskProfile = @import("advanced.zig").TaskProfile;
pub const TaskDomain = @import("advanced.zig").TaskDomain;
pub const LearningStrategy = @import("advanced.zig").LearningStrategy;
pub const MetaLearner = @import("advanced.zig").MetaLearner;
pub const FewShotLearner = @import("advanced.zig").FewShotLearner;
pub const CurriculumScheduler = @import("advanced.zig").CurriculumScheduler;
pub const MentalModel = @import("advanced.zig").MentalModel;
pub const BeliefSystem = @import("advanced.zig").BeliefSystem;
pub const KnowledgeState = @import("advanced.zig").KnowledgeState;
pub const IntentionTracker = @import("advanced.zig").IntentionTracker;
pub const EmotionalModel = @import("advanced.zig").EmotionalModel;
pub const TheoryOfMind = @import("advanced.zig").TheoryOfMind;
pub const ProblemDecomposer = @import("advanced.zig").ProblemDecomposer;
pub const SelfEvaluation = @import("advanced.zig").SelfEvaluation;
pub const UncertaintyArea = @import("advanced.zig").UncertaintyArea;
pub const DetectedBias = @import("advanced.zig").DetectedBias;
pub const ReasoningQuality = @import("advanced.zig").ReasoningQuality;
pub const SelfReflectionEngine = @import("advanced.zig").SelfReflectionEngine;
