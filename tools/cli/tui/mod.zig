//! TUI Module
//!
//! Terminal User Interface components for the ABI CLI.
//! Includes widgets, panels, themes, and dashboard views.

pub const events = @import("events.zig");
pub const terminal = @import("terminal.zig");
pub const widgets = @import("widgets.zig");
pub const themes = @import("themes.zig");
pub const gpu_monitor = @import("gpu_monitor.zig");
pub const agent_panel = @import("agent_panel.zig");
pub const training_panel = @import("training_panel.zig");
pub const training_metrics = @import("training_metrics.zig");
pub const ring_buffer = @import("ring_buffer.zig");
pub const percentile_tracker = @import("percentile_tracker.zig");
pub const model_panel = @import("model_panel.zig");
pub const streaming_dashboard = @import("streaming_dashboard.zig");
pub const async_loop = @import("async_loop.zig");

pub const Key = events.Key;
pub const KeyCode = events.KeyCode;
pub const Modifiers = events.Modifiers;
pub const Event = events.Event;
pub const Mouse = events.Mouse;
pub const MouseButton = events.MouseButton;

pub const Terminal = terminal.Terminal;
pub const TerminalSize = terminal.TerminalSize;
pub const PlatformCapabilities = terminal.PlatformCapabilities;

// Widgets
pub const ProgressIndicator = widgets.ProgressIndicator;
pub const ProgressBar = widgets.ProgressBar;
pub const Dialog = widgets.Dialog;
pub const DialogResult = widgets.DialogResult;
pub const CommandPreview = widgets.CommandPreview;
pub const Toast = widgets.Toast;
pub const SpinnerStyle = widgets.SpinnerStyle;
pub const SparklineChart = widgets.SparklineChart;
pub const ProgressGauge = widgets.ProgressGauge;

// GPU Monitor Widget
pub const GpuMonitor = gpu_monitor.GpuMonitor;
pub const GpuDeviceStatus = gpu_monitor.GpuDeviceStatus;
pub const SchedulerStats = gpu_monitor.SchedulerStats;
pub const MemoryHistory = gpu_monitor.MemoryHistory;
pub const BackendType = gpu_monitor.BackendType;

// Agent Panel Widget
pub const AgentPanel = agent_panel.AgentPanel;
pub const LearningPhase = agent_panel.LearningPhase;
pub const DecisionEntry = agent_panel.DecisionEntry;
pub const RewardHistory = agent_panel.RewardHistory;

// Training Panel Widget
pub const TrainingPanel = training_panel.TrainingPanel;
pub const TrainingPanelConfig = training_panel.PanelConfig;
pub const TrainingMetrics = training_metrics.TrainingMetrics;
pub const MetricEvent = training_metrics.MetricEvent;
pub const MetricsParser = training_metrics.MetricsParser;

// Themes
pub const Theme = themes.Theme;
pub const ThemeManager = themes.ThemeManager;
pub const builtinThemes = themes.themes;

// Data Structures for Dashboards
pub const RingBuffer = ring_buffer.RingBuffer;
pub const PercentileTracker = percentile_tracker.PercentileTracker;

// Model Management Panel
pub const ModelManagementPanel = model_panel.ModelManagementPanel;

// Streaming Dashboard
pub const StreamingDashboard = streaming_dashboard.StreamingDashboard;

// Async Loop
pub const AsyncLoop = async_loop.AsyncLoop;
pub const AsyncEvent = async_loop.AsyncEvent;
pub const AsyncLoopConfig = async_loop.AsyncLoopConfig;
pub const MetricsTracker = async_loop.MetricsTracker;
