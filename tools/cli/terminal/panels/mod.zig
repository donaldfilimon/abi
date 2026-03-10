//! Panel adapters and native panels for the unified TUI dashboard.
//!
//! Adapters wrap existing TUI panels (gpu_monitor, agent_panel, etc.)
//! in the Panel vtable interface. Native panels implement the vtable directly.

// Adapters (wrap existing panels)
// pub const gpu_adapter = @import("gpu_adapter");
pub const agent_adapter = @import("agent_adapter");
pub const training_adapter = @import("training_adapter");
pub const model_adapter = @import("model_adapter");
pub const streaming_adapter = @import("streaming_adapter");
// pub const brain_adapter = @import("brain_adapter");
pub const bench_adapter = @import("bench_adapter");
pub const db_adapter = @import("db_adapter");
pub const network_adapter = @import("network_adapter");
pub const chat_adapter = @import("chat_adapter");
pub const gpu_monitor = @import("../gpu_monitor");
pub const brain_panel = @import("../brain_panel");

// Native panels (implement Panel vtable directly)
pub const security_panel = @import("security_panel");
pub const connectors_panel = @import("connectors_panel");
pub const ralph_panel = @import("ralph_panel");
pub const memory_panel = @import("memory_panel");
pub const create_subagent_panel = @import("create_subagent_panel");
pub const registry = @import("registry");

// Convenience type aliases
pub const SecurityPanel = security_panel.SecurityPanel;
pub const ConnectorsPanel = connectors_panel.ConnectorsPanel;
pub const RalphPanel = ralph_panel.RalphPanel;
pub const MemoryPanel = memory_panel.MemoryPanel;
pub const CreateSubagentPanel = create_subagent_panel.CreateSubagentPanel;

test {
    @import("std").testing.refAllDecls(@This());
}
