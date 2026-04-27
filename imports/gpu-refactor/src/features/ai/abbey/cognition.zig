pub const reasoning = @import("reasoning.zig");
pub const emotions = @import("emotions.zig");
pub const context = @import("context.zig");
pub const calibration = @import("calibration.zig");
pub const advanced = @import("advanced.zig");

// Re-exports
pub const ReasoningChain = reasoning.ReasoningChain;
pub const ReasoningStep = reasoning.ReasoningStep;
pub const StepType = reasoning.StepType;
pub const ConversationContext = context.ConversationContext;
pub const TopicTracker = context.TopicTracker;
pub const ContextWindow = context.ContextWindow;

pub const Evidence = calibration.Evidence;
pub const CalibrationResult = calibration.CalibrationResult;
pub const ConfidenceCalibrator = calibration.ConfidenceCalibrator;
pub const QueryAnalyzer = calibration.QueryAnalyzer;

pub const TaskProfile = advanced.TaskProfile;
pub const TaskDomain = advanced.TaskDomain;
pub const LearningStrategy = advanced.LearningStrategy;
pub const MetaLearner = advanced.MetaLearner;
pub const FewShotLearner = advanced.FewShotLearner;
pub const CurriculumScheduler = advanced.CurriculumScheduler;

pub const MentalModel = advanced.MentalModel;
pub const BeliefSystem = advanced.BeliefSystem;
pub const KnowledgeState = advanced.KnowledgeState;
pub const IntentionTracker = advanced.IntentionTracker;
pub const EmotionalModel = advanced.EmotionalModel;
pub const TheoryOfMind = advanced.TheoryOfMind;

pub const ProblemDecomposition = advanced.ProblemDecomposition;
pub const SubProblem = advanced.SubProblem;
pub const ExecutionPlan = advanced.ExecutionPlan;
pub const ProblemDecomposer = advanced.ProblemDecomposer;
pub const CounterfactualReasoner = advanced.CounterfactualReasoner;

pub const SelfEvaluation = advanced.SelfEvaluation;
pub const UncertaintyArea = advanced.UncertaintyArea;
pub const DetectedBias = advanced.DetectedBias;
pub const ReasoningQuality = advanced.ReasoningQuality;
pub const SelfReflectionEngine = advanced.SelfReflectionEngine;

pub const AdvancedCognition = advanced.AdvancedCognition;
pub const CognitiveResult = advanced.CognitiveResult;
pub const CognitiveState = advanced.CognitiveState;
