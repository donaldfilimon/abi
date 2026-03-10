//! Advanced Cognition Wrapper
//!
//! Provides a flattened import path for the Abbey advanced cognition module.
//! The detailed implementation remains in `advanced/mod.zig` and its
//! sub-modules.

pub const AdvancedCognition = @import("advanced").AdvancedCognition;
pub const CognitiveResult = @import("advanced").CognitiveResult;
pub const CognitiveState = @import("advanced").CognitiveState;
// Re-export other public symbols as needed
pub const TaskProfile = @import("advanced").TaskProfile;
pub const TaskDomain = @import("advanced").TaskDomain;
pub const LearningStrategy = @import("advanced").LearningStrategy;
pub const MetaLearner = @import("advanced").MetaLearner;
pub const FewShotLearner = @import("advanced").FewShotLearner;
pub const CurriculumScheduler = @import("advanced").CurriculumScheduler;
pub const MentalModel = @import("advanced").MentalModel;
pub const BeliefSystem = @import("advanced").BeliefSystem;
pub const KnowledgeState = @import("advanced").KnowledgeState;
pub const IntentionTracker = @import("advanced").IntentionTracker;
pub const EmotionalModel = @import("advanced").EmotionalModel;
pub const TheoryOfMind = @import("advanced").TheoryOfMind;
pub const ProblemDecomposer = @import("advanced").ProblemDecomposer;
pub const SelfEvaluation = @import("advanced").SelfEvaluation;
pub const UncertaintyArea = @import("advanced").UncertaintyArea;
pub const DetectedBias = @import("advanced").DetectedBias;
pub const ReasoningQuality = @import("advanced").ReasoningQuality;
pub const SelfReflectionEngine = @import("advanced").SelfReflectionEngine;
