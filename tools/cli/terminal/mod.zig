//! TUI Module
//!
//! Terminal User Interface components for the ABI CLI.
//! Includes widgets, panels, themes, and dashboard views.

pub const layout = @import("layout");
pub const unicode = @import("unicode");
pub const events = @import("events");
pub const terminal = @import("terminal");
pub const widgets = @import("widgets");
pub const themes = @import("themes");
pub const render_utils = @import("render_utils");
pub const component = @import("component");
pub const panel = @import("panel");
pub const tab_bar = @import("tab_bar");
pub const help_overlay = @import("help_overlay");
pub const gpu_monitor = @import("gpu_monitor");
pub const agent_panel = @import("agent_panel");
pub const training_panel = @import("training_panel");
pub const training_metrics = @import("training_metrics");
pub const ring_buffer = @import("ring_buffer");
pub const percentile_tracker = @import("percentile_tracker");
pub const model_panel = @import("model_panel");
pub const bench_panel = @import("bench_panel");
pub const db_panel = @import("db_panel");
pub const network_panel = @import("network_panel");
pub const streaming_dashboard = @import("streaming_dashboard");
pub const brain_animation = @import("brain_animation");
pub const brain_panel = @import("brain_panel");
pub const metrics_file_reader = @import("metrics_file_reader");
pub const training_brain_mapper = @import("training_brain_mapper");
pub const async_loop = @import("async_loop");
pub const dashboard = @import("dashboard");
pub const keybindings = @import("keybindings");

pub const Key = events.Key;
pub const KeyCode = events.KeyCode;
pub const Modifiers = events.Modifiers;
pub const Event = events.Event;
pub const Mouse = events.Mouse;
pub const MouseButton = events.MouseButton;

pub const Terminal = terminal.Terminal;
pub const TerminalSize = terminal.TerminalSize;
pub const PlatformCapabilities = terminal.PlatformCapabilities;

// Layout
pub const Rect = layout.Rect;
pub const Constraint = layout.Constraint;

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
pub const BenchmarkPanel = bench_panel.BenchmarkPanel;
pub const DatabasePanel = db_panel.DatabasePanel;
pub const NetworkPanel = network_panel.NetworkPanel;

// Streaming Dashboard
pub const StreamingDashboard = streaming_dashboard.StreamingDashboard;

// Brain Dashboard
pub const BrainAnimation = brain_animation.BrainAnimation;
pub const BrainDashboardPanel = brain_panel.BrainDashboardPanel;
pub const BrainDashboardData = brain_panel.DashboardData;

// Training Brain Mapping
pub const MetricsFileReader = metrics_file_reader.MetricsFileReader;
pub const TrainingBrainMapper = training_brain_mapper.TrainingBrainMapper;

// Render Utilities
pub const BoxStyle = render_utils.BoxStyle;
pub const BoxChars = render_utils.BoxChars;

// Component System
pub const SubPanel = component.SubPanel;
pub const RenderFn = component.RenderFn;
pub const Panel = panel.Panel;
pub const TabBar = tab_bar.TabBar;
pub const HelpOverlay = help_overlay.HelpOverlay;

// Async Loop
pub const AsyncLoop = async_loop.AsyncLoop;
pub const AsyncEvent = async_loop.AsyncEvent;
pub const AsyncLoopConfig = async_loop.AsyncLoopConfig;
pub const MetricsTracker = async_loop.MetricsTracker;

// Generic Dashboard
pub const Dashboard = dashboard.Dashboard;

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
