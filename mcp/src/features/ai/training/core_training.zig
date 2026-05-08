pub const optimizer = @import("optimizer.zig");
pub const trainer = @import("trainer.zig");
pub const training_utils = @import("training_utils.zig");
pub const gradient = @import("gradient.zig");
pub const loss = @import("loss.zig");
pub const logging = @import("logging.zig");
pub const mixed_precision = @import("mixed_precision.zig");

// Re-exports
pub const Optimizer = optimizer.Optimizer;
pub const SgdOptimizer = optimizer.SgdOptimizer;
pub const AdamOptimizer = optimizer.AdamOptimizer;
pub const AdamWOptimizer = optimizer.AdamWOptimizer;

pub const TrainingResult = trainer.TrainingResult;
pub const train = trainer.train;
pub const trainAndReport = trainer.trainAndReport;
pub const trainWithResult = trainer.trainWithResult;
pub const calculateLearningRate = trainer.calculateLearningRate;
pub const clipGradients = trainer.clipGradients;
pub const saveModelToWdbx = trainer.saveModelToWdbx;

pub const GradientAccumulator = gradient.GradientAccumulator;
pub const GradientError = gradient.GradientError;

pub const CrossEntropyLoss = loss.CrossEntropyLoss;
pub const MSELoss = loss.MSELoss;
pub const FocalLoss = loss.FocalLoss;
pub const perplexity = loss.perplexity;
pub const klDivergence = loss.klDivergence;

pub const TrainingLogger = logging.TrainingLogger;
pub const TrainingLogConfig = logging.LoggerConfig;
pub const TrainingLogMetric = logging.Metric;

pub const MixedPrecisionConfig = mixed_precision.MixedPrecisionConfig;
pub const MixedPrecisionContext = mixed_precision.MixedPrecisionContext;
pub const LossScaler = mixed_precision.LossScaler;
pub const MasterWeights = mixed_precision.MasterWeights;
pub const fp32ToFp16 = mixed_precision.fp32ToFp16;
pub const fp16ToFp32 = mixed_precision.fp16ToFp32;
