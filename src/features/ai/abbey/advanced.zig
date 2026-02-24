//! Advanced Cognition Wrapper
//!
//! Provides a flattened import path for the Abbey advanced cognition module.
//! The detailed implementation remains in `advanced/mod.zig` and its
//! sub-modules.

pub const AdvancedCognition = @import("advanced/mod.zig").AdvancedCognition;
pub const CognitiveResult = @import("advanced/mod.zig").CognitiveResult;
pub const CognitiveState = @import("advanced/mod.zig").CognitiveState;
// Re-export other public symbols as needed
pub const TaskProfile = @import("advanced/mod.zig").TaskProfile;
pub const TaskDomain = @import("advanced/mod.zig").TaskDomain;
pub const LearningStrategy = @import("advanced/mod.zig").LearningStrategy;
pub const MetaLearner = @import("advanced/mod.zig").MetaLearner;
pub const FewShotLearner = @import("advanced/mod.zig").FewShotLearner;
pub const CurriculumScheduler = @import("advanced/mod.zig").CurriculumScheduler;
pub const MentalModel = @import("advanced/mod.zig").MentalModel;
pub const BeliefSystem = @import("advanced/mod.zig").BeliefSystem;
pub const KnowledgeState = @import("advanced/mod.zig").KnowledgeState;
pub const IntentionTracker = @import("advanced/mod.zig").IntentionTracker;
pub const EmotionalModel = @import("advanced/mod.zig").EmotionalModel;
pub const TheoryOfMind = @import("advanced/mod.zig").TheoryOfMind;
pub const ProblemDecomposer = @import("advanced/mod.zig").ProblemDecomposer;
pub const SelfEvaluation = @import("advanced/mod.zig").SelfEvaluation;
pub const UncertaintyArea = @import("advanced/mod.zig").UncertaintyArea;
pub const DetectedBias = @import("advanced/mod.zig").DetectedBias;
pub const ReasoningQuality = @import("advanced/mod.zig").ReasoningQuality;
pub const SelfReflectionEngine = @import("advanced/mod.zig").SelfReflectionEngine;
const std = @import("std");
