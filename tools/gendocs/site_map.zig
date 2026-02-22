const model = @import("model.zig");

pub const guides = [_]model.GuideSpec{
    .{ .slug = "installation", .title = "Installation", .description = "System requirements, toolchain setup, and building ABI from source", .section = "Start", .order = 1, .permalink = "/installation/", .template_path = "tools/gendocs/templates/docs/installation.md.tpl" },
    .{ .slug = "getting-started", .title = "Getting Started", .description = "First build, first test, first example with ABI", .section = "Start", .order = 2, .permalink = "/getting-started/", .template_path = "tools/gendocs/templates/docs/getting-started.md.tpl" },

    .{ .slug = "architecture", .title = "Architecture", .description = "Module hierarchy, comptime feature gating, and framework lifecycle", .section = "Core", .order = 10, .permalink = "/architecture/", .template_path = "tools/gendocs/templates/docs/architecture.md.tpl" },
    .{ .slug = "configuration", .title = "Configuration", .description = "Feature flags, config builder, and environment variables", .section = "Core", .order = 11, .permalink = "/configuration/", .template_path = "tools/gendocs/templates/docs/configuration.md.tpl" },
    .{ .slug = "framework", .title = "Framework Lifecycle", .description = "Deep dive into Framework initialization, state machine, builder pattern, and feature registry", .section = "Core", .order = 12, .permalink = "/framework/", .template_path = "tools/gendocs/templates/docs/framework.md.tpl" },
    .{ .slug = "cli", .title = "CLI", .description = "29 commands and aliases for AI, GPU, database, and system management", .section = "Core", .order = 13, .permalink = "/cli/", .template_path = "tools/gendocs/templates/docs/cli.md.tpl" },

    .{ .slug = "ai-overview", .title = "AI Overview", .description = "Architecture overview of ABI AI modules", .section = "AI", .order = 20, .permalink = "/ai-overview/", .template_path = "tools/gendocs/templates/docs/ai-overview.md.tpl" },
    .{ .slug = "ai-core", .title = "AI Core", .description = "Agents, tools, prompts, personas, memory, and model discovery", .section = "AI", .order = 21, .permalink = "/ai-core/", .template_path = "tools/gendocs/templates/docs/ai-core.md.tpl" },
    .{ .slug = "ai-inference", .title = "Inference", .description = "LLM inference, embeddings, vision, streaming, and transformer architecture", .section = "AI", .order = 22, .permalink = "/ai-inference/", .template_path = "tools/gendocs/templates/docs/ai-inference.md.tpl" },
    .{ .slug = "ai-training", .title = "Training", .description = "Training pipelines, federated learning, and multimodal training", .section = "AI", .order = 23, .permalink = "/ai-training/", .template_path = "tools/gendocs/templates/docs/ai-training.md.tpl" },
    .{ .slug = "ai-reasoning", .title = "Reasoning", .description = "Abbey reasoning engine, RAG, evaluation, templates, and orchestration", .section = "AI", .order = 24, .permalink = "/ai-reasoning/", .template_path = "tools/gendocs/templates/docs/ai-reasoning.md.tpl" },

    .{ .slug = "gpu", .title = "GPU", .description = "Multi-backend GPU architecture and execution model", .section = "GPU", .order = 30, .permalink = "/gpu/", .template_path = "tools/gendocs/templates/docs/gpu.md.tpl" },
    .{ .slug = "gpu-backends", .title = "GPU Backends", .description = "Per-backend details and platform requirements", .section = "GPU", .order = 31, .permalink = "/gpu-backends/", .template_path = "tools/gendocs/templates/docs/gpu-backends.md.tpl" },

    .{ .slug = "database", .title = "Database (WDBX)", .description = "Vector database, indexing, and retrieval", .section = "Data", .order = 40, .permalink = "/database/", .template_path = "tools/gendocs/templates/docs/database.md.tpl" },
    .{ .slug = "cache", .title = "Cache", .description = "In-memory caching and eviction policy behavior", .section = "Data", .order = 41, .permalink = "/cache/", .template_path = "tools/gendocs/templates/docs/cache.md.tpl" },
    .{ .slug = "storage", .title = "Storage", .description = "Unified file/object storage backends", .section = "Data", .order = 42, .permalink = "/storage/", .template_path = "tools/gendocs/templates/docs/storage.md.tpl" },
    .{ .slug = "search", .title = "Search", .description = "Full-text and hybrid retrieval capabilities", .section = "Data", .order = 43, .permalink = "/search/", .template_path = "tools/gendocs/templates/docs/search.md.tpl" },

    .{ .slug = "network", .title = "Network", .description = "Distributed compute and node orchestration", .section = "Infrastructure", .order = 50, .permalink = "/network/", .template_path = "tools/gendocs/templates/docs/network.md.tpl" },
    .{ .slug = "gateway", .title = "Gateway", .description = "API gateway routing, rate limiting, and resilience", .section = "Infrastructure", .order = 51, .permalink = "/gateway/", .template_path = "tools/gendocs/templates/docs/gateway.md.tpl" },
    .{ .slug = "messaging", .title = "Messaging", .description = "Event bus, pub/sub, and queueing", .section = "Infrastructure", .order = 52, .permalink = "/messaging/", .template_path = "tools/gendocs/templates/docs/messaging.md.tpl" },
    .{ .slug = "pages", .title = "Pages", .description = "Dashboard UI pages and routing", .section = "Infrastructure", .order = 53, .permalink = "/pages/", .template_path = "tools/gendocs/templates/docs/pages.md.tpl" },
    .{ .slug = "web", .title = "Web", .description = "HTTP utilities and web middleware", .section = "Infrastructure", .order = 54, .permalink = "/web/", .template_path = "tools/gendocs/templates/docs/web.md.tpl" },
    .{ .slug = "cloud", .title = "Cloud", .description = "Cloud function adapters", .section = "Infrastructure", .order = 55, .permalink = "/cloud/", .template_path = "tools/gendocs/templates/docs/cloud.md.tpl" },
    .{ .slug = "mobile", .title = "Mobile", .description = "Mobile lifecycle, sensors, and notifications", .section = "Infrastructure", .order = 56, .permalink = "/mobile/", .template_path = "tools/gendocs/templates/docs/mobile.md.tpl" },

    .{ .slug = "auth", .title = "Auth & Security", .description = "Authentication, authorization, and security infrastructure", .section = "Operations", .order = 60, .permalink = "/auth/", .template_path = "tools/gendocs/templates/docs/auth.md.tpl" },
    .{ .slug = "analytics", .title = "Analytics", .description = "Event tracking and experiment framework", .section = "Operations", .order = 61, .permalink = "/analytics/", .template_path = "tools/gendocs/templates/docs/analytics.md.tpl" },
    .{ .slug = "observability", .title = "Observability", .description = "Metrics, tracing, profiling", .section = "Operations", .order = 62, .permalink = "/observability/", .template_path = "tools/gendocs/templates/docs/observability.md.tpl" },
    .{ .slug = "deployment", .title = "Deployment", .description = "Docker, Kubernetes, and production deployment", .section = "Operations", .order = 63, .permalink = "/deployment/", .template_path = "tools/gendocs/templates/docs/deployment.md.tpl" },
    .{ .slug = "benchmarks", .title = "Benchmarks", .description = "Built-in performance benchmark suite", .section = "Operations", .order = 64, .permalink = "/benchmarks/", .template_path = "tools/gendocs/templates/docs/benchmarks.md.tpl" },

    .{ .slug = "api", .title = "API Overview", .description = "Public API surface, import patterns, and HTTP endpoints", .section = "Reference", .order = 70, .permalink = "/api-overview/", .template_path = "tools/gendocs/templates/docs/api.md.tpl" },
    .{ .slug = "examples", .title = "Examples", .description = "Runnable examples across major modules", .section = "Reference", .order = 71, .permalink = "/examples/", .template_path = "tools/gendocs/templates/docs/examples.md.tpl" },
    .{ .slug = "c-bindings", .title = "C API Bindings", .description = "C-compatible ABI API exports and integration notes", .section = "Reference", .order = 72, .permalink = "/c-bindings/", .template_path = "tools/gendocs/templates/docs/c-bindings.md.tpl" },
    .{ .slug = "troubleshooting", .title = "Troubleshooting", .description = "Common errors and recovery guidance", .section = "Reference", .order = 73, .permalink = "/troubleshooting/", .template_path = "tools/gendocs/templates/docs/troubleshooting.md.tpl" },
    .{ .slug = "contributing", .title = "Contributing", .description = "Development workflow and contribution guidelines", .section = "Reference", .order = 74, .permalink = "/contributing/", .template_path = "tools/gendocs/templates/docs/contributing.md.tpl" },
    .{ .slug = "roadmap", .title = "Roadmap", .description = "Canonical now/next/later execution roadmap", .section = "Reference", .order = 75, .permalink = "/roadmap/", .template_path = "tools/gendocs/templates/docs/roadmap.md.tpl" },

    .{ .slug = "connectors", .title = "LLM Connectors", .description = "Provider and connector capability matrix", .section = "Services", .order = 80, .permalink = "/connectors/", .template_path = "tools/gendocs/templates/docs/connectors.md.tpl" },
    .{ .slug = "mcp", .title = "MCP Server", .description = "Model Context Protocol surface", .section = "Services", .order = 81, .permalink = "/mcp/", .template_path = "tools/gendocs/templates/docs/mcp.md.tpl" },
    .{ .slug = "acp", .title = "ACP Protocol", .description = "Agent Communication Protocol integration", .section = "Services", .order = 82, .permalink = "/acp/", .template_path = "tools/gendocs/templates/docs/acp.md.tpl" },

    .{ .slug = "llm-inference-guide", .title = "LLM Inference Guide", .description = "Provider router architecture, backend selection, fallback chains, and CLI usage for LLM inference", .section = "Guides", .order = 90, .permalink = "/llm-inference-guide/", .template_path = "tools/gendocs/templates/docs/llm-inference-guide.md.tpl" },
    .{ .slug = "ralph-guide", .title = "Ralph Orchestrator Guide", .description = "Autonomous AI code agent with iterative improvement loops, skill memory, and multi-agent coordination", .section = "Guides", .order = 91, .permalink = "/ralph-guide/", .template_path = "tools/gendocs/templates/docs/ralph-guide.md.tpl" },
    .{ .slug = "security-guide", .title = "Security Module Guide", .description = "JWT, RBAC, session management, encryption, TLS/mTLS, certificate management, API keys, secrets, and audit logging", .section = "Guides", .order = 92, .permalink = "/security-guide/", .template_path = "tools/gendocs/templates/docs/security-guide.md.tpl" },
    .{ .slug = "streaming-guide", .title = "Streaming Guide", .description = "AI streaming architecture with SSE, WebSocket, backpressure handling, and circuit breaker patterns", .section = "Guides", .order = 93, .permalink = "/streaming-guide/", .template_path = "tools/gendocs/templates/docs/streaming-guide.md.tpl" },
    .{ .slug = "connectors-guide", .title = "Connectors Guide", .description = "Connector configuration, environment variables, auto-discovery, and custom plugin connectors", .section = "Guides", .order = 94, .permalink = "/connectors-guide/", .template_path = "tools/gendocs/templates/docs/connectors-guide.md.tpl" },
};
