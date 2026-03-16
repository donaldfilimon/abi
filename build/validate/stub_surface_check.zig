//! Stub Surface Check
//!
//! Compiled under every flag combination in the validation matrix to ensure
//! that mod.zig and stub.zig expose the same public API surface.  This file
//! dereferences key public symbols from each feature namespace.  If a stub
//! is missing a symbol that the mod exports (or vice versa), the compile
//! will fail for the relevant flag combo, catching mod/stub drift early.
//!
//! This file is intentionally a library (no main/entry point).  The build
//! system compiles it as a static library under each flag combination.

const std = @import("std");
const abi = @import("abi");

// ============================================================================
// Core (always available, no feature gating)
// ============================================================================

comptime {
    _ = abi.App;
    _ = abi.AppBuilder;
    _ = abi.Config;
    _ = abi.Feature;
    _ = abi.FrameworkError;
    _ = abi.Registry;
    _ = abi.config;
    _ = abi.framework;
    _ = abi.errors;
    _ = abi.registry;
    _ = abi.feature_catalog;
    _ = abi.meta;
    _ = abi.version;
    _ = abi.appBuilder;
}

// ============================================================================
// Canonical top-level API (abi.<domain>)
// ============================================================================

comptime {
    // Core
    _ = abi.config;
    _ = abi.errors;
    _ = abi.registry;
    _ = abi.framework;

    // Services (non-feature-gated)
    _ = abi.foundation;
    _ = abi.runtime;
    _ = abi.platform;
    _ = abi.connectors;
    _ = abi.tasks;
    _ = abi.mcp;
    _ = abi.lsp;
    _ = abi.acp;
    _ = abi.ha;

    // Features (comptime-gated)
    _ = abi.gpu;
    _ = abi.ai;
    _ = abi.database;
    _ = abi.network;
    _ = abi.observability;
    _ = abi.web;
    _ = abi.pages;
    _ = abi.analytics;
    _ = abi.cloud;
    _ = abi.auth;
    _ = abi.messaging;
    _ = abi.cache;
    _ = abi.storage;
    _ = abi.search;
    _ = abi.mobile;
    _ = abi.gateway;
    _ = abi.benchmarks;
    _ = abi.compute;
    _ = abi.documents;
    _ = abi.desktop;
}

// ============================================================================
// Compat bridges (abi.services.*, abi.features.*)
// ============================================================================

comptime {
    _ = abi.services.runtime;
    _ = abi.services.platform;
    _ = abi.services.shared;
    _ = abi.services.connectors;
    _ = abi.services.ha;
    _ = abi.services.tasks;
    _ = abi.services.lsp;
    _ = abi.services.mcp;
    _ = abi.services.acp;
    _ = abi.services.simd;
}

// ============================================================================
// GPU Feature
// ============================================================================

comptime {
    _ = abi.features.gpu.Gpu;
    _ = abi.features.gpu.Backend;
    _ = abi.features.gpu.Device;
    _ = abi.features.gpu.Buffer;
    _ = abi.features.gpu.Stream;
    _ = abi.features.gpu.GpuConfig;
    _ = abi.features.gpu.Error;
    _ = abi.features.gpu.GpuDevice;
    _ = abi.features.gpu.Context;
    _ = abi.features.gpu.backends;
    _ = abi.features.gpu.diagnostics;
    _ = abi.features.gpu.platform;
    _ = abi.Gpu;
    _ = abi.GpuBackend;
}

// ============================================================================
// AI Feature
// ============================================================================

comptime {
    _ = abi.features.ai.Error;
    _ = abi.features.ai.Context;
    _ = abi.features.ai.core;
    _ = abi.features.ai.llm;
    _ = abi.features.ai.agents;
    _ = abi.features.ai.training;
    _ = abi.features.ai.streaming;
    _ = abi.features.ai.personas;
    _ = abi.features.ai.profiles;
    _ = abi.features.ai.coordination;
    _ = abi.features.ai.abbey;
    _ = abi.features.ai.explore;
    _ = abi.features.ai.rag;
    _ = abi.features.ai.eval;
    _ = abi.features.ai.constitution;
    _ = abi.features.ai.transformer;
    _ = abi.features.ai.tools;
    _ = abi.features.ai.models;
    _ = abi.features.ai.memory;
    _ = abi.features.ai.documents;
    _ = abi.features.ai.vision;
    _ = abi.features.ai.embeddings;
    _ = abi.features.ai.orchestration;
    _ = abi.features.ai.multi_agent;
    _ = abi.features.ai.templates;
    _ = abi.features.ai.federated;
    _ = abi.features.ai.prompts;
    _ = abi.features.ai.database;
    _ = abi.features.ai.isEnabled;
    _ = abi.features.ai.isInitialized;
}

// ============================================================================
// Database Feature
// ============================================================================

comptime {
    _ = abi.features.database.DatabaseHandle;
    _ = abi.features.database.SearchResult;
    _ = abi.features.database.Context;
    _ = abi.features.database.semantic_store;
    _ = abi.features.database.isEnabled;
    _ = abi.features.database.isInitialized;
    _ = abi.features.database.Stats;
}

// ============================================================================
// Network Feature
// ============================================================================

comptime {
    _ = abi.features.network.Error;
    _ = abi.features.network.Context;
    _ = abi.features.network.NetworkConfig;
    _ = abi.features.network.Node;
    _ = abi.features.network.RaftNode;
    _ = abi.features.network.CircuitBreaker;
    _ = abi.features.network.LoadBalancer;
    _ = abi.features.network.RetryConfig;
    _ = abi.features.network.ConnectionPool;
    _ = abi.features.network.isEnabled;
    _ = abi.features.network.isInitialized;
}

// ============================================================================
// Observability Feature
// ============================================================================

comptime {
    _ = abi.features.observability.Error;
    _ = abi.features.observability.Context;
    _ = abi.features.observability.MetricsCollector;
    _ = abi.features.observability.Counter;
    _ = abi.features.observability.Gauge;
    _ = abi.features.observability.Histogram;
    _ = abi.features.observability.Tracer;
    _ = abi.features.observability.Span;
    _ = abi.features.observability.AlertManager;
    _ = abi.features.observability.isEnabled;
    _ = abi.features.observability.isInitialized;
}

// ============================================================================
// Web Feature
// ============================================================================

comptime {
    _ = abi.features.web.Context;
    _ = abi.features.web.ChatHandler;
    _ = abi.features.web.HttpClient;
    _ = abi.features.web.Response;
    _ = abi.features.web.server;
    _ = abi.features.web.middleware;
    _ = abi.features.web.isEnabled;
    _ = abi.features.web.isInitialized;
}

// ============================================================================
// Analytics Feature
// ============================================================================

comptime {
    _ = abi.features.analytics.Engine;
    _ = abi.features.analytics.Event;
    _ = abi.features.analytics.Error;
    _ = abi.features.analytics.AnalyticsConfig;
    _ = abi.features.analytics.Context;
    _ = abi.features.analytics.Funnel;
    _ = abi.features.analytics.Experiment;
    _ = abi.features.analytics.isEnabled;
    _ = abi.features.analytics.isInitialized;
}

// ============================================================================
// Cloud Feature
// ============================================================================

comptime {
    _ = abi.features.cloud.Error;
    _ = abi.features.cloud.Context;
    _ = abi.features.cloud.CloudProvider;
    _ = abi.features.cloud.CloudConfig;
    _ = abi.features.cloud.CloudEvent;
    _ = abi.features.cloud.CloudResponse;
    _ = abi.features.cloud.ResponseBuilder;
    _ = abi.features.cloud.isEnabled;
    _ = abi.features.cloud.isInitialized;
}

// ============================================================================
// Auth Feature
// ============================================================================

comptime {
    _ = abi.features.auth.AuthError;
    _ = abi.features.auth.Error;
    _ = abi.features.auth.AuthConfig;
    _ = abi.features.auth.Token;
    _ = abi.features.auth.Session;
    _ = abi.features.auth.Permission;
    _ = abi.features.auth.Context;
    _ = abi.features.auth.isEnabled;
    _ = abi.features.auth.isInitialized;
}

// ============================================================================
// Messaging Feature
// ============================================================================

comptime {
    _ = abi.features.messaging.MessagingError;
    _ = abi.features.messaging.Error;
    _ = abi.features.messaging.MessagingConfig;
    _ = abi.features.messaging.Message;
    _ = abi.features.messaging.Channel;
    _ = abi.features.messaging.MessagingStats;
    _ = abi.features.messaging.Context;
    _ = abi.features.messaging.isEnabled;
    _ = abi.features.messaging.isInitialized;
}

// ============================================================================
// Cache Feature
// ============================================================================

comptime {
    _ = abi.features.cache.CacheError;
    _ = abi.features.cache.Error;
    _ = abi.features.cache.CacheConfig;
    _ = abi.features.cache.CacheEntry;
    _ = abi.features.cache.CacheStats;
    _ = abi.features.cache.Context;
    _ = abi.features.cache.isEnabled;
    _ = abi.features.cache.isInitialized;
}

// ============================================================================
// Storage Feature
// ============================================================================

comptime {
    _ = abi.features.storage.StorageError;
    _ = abi.features.storage.Error;
    _ = abi.features.storage.StorageConfig;
    _ = abi.features.storage.StorageObject;
    _ = abi.features.storage.ObjectMetadata;
    _ = abi.features.storage.StorageStats;
    _ = abi.features.storage.Context;
    _ = abi.features.storage.isEnabled;
    _ = abi.features.storage.isInitialized;
}

// ============================================================================
// Search Feature
// ============================================================================

comptime {
    _ = abi.features.search.SearchError;
    _ = abi.features.search.Error;
    _ = abi.features.search.SearchConfig;
    _ = abi.features.search.SearchResult;
    _ = abi.features.search.SearchIndex;
    _ = abi.features.search.SearchStats;
    _ = abi.features.search.Context;
    _ = abi.features.search.isEnabled;
    _ = abi.features.search.isInitialized;
}

// ============================================================================
// Mobile Feature
// ============================================================================

comptime {
    _ = abi.features.mobile.MobileError;
    _ = abi.features.mobile.Error;
    _ = abi.features.mobile.MobileConfig;
    _ = abi.features.mobile.SensorType;
    _ = abi.features.mobile.SensorData;
    _ = abi.features.mobile.Notification;
    _ = abi.features.mobile.Permission;
    _ = abi.features.mobile.PermissionStatus;
    _ = abi.features.mobile.DeviceInfo;
    _ = abi.features.mobile.Context;
    _ = abi.features.mobile.isEnabled;
    _ = abi.features.mobile.isInitialized;
}

// ============================================================================
// Gateway Feature
// ============================================================================

comptime {
    _ = abi.features.gateway.GatewayError;
    _ = abi.features.gateway.Error;
    _ = abi.features.gateway.GatewayConfig;
    _ = abi.features.gateway.Route;
    _ = abi.features.gateway.MatchResult;
    _ = abi.features.gateway.GatewayStats;
    _ = abi.features.gateway.Context;
    _ = abi.features.gateway.isEnabled;
    _ = abi.features.gateway.isInitialized;
}

// ============================================================================
// Pages Feature
// ============================================================================

comptime {
    _ = abi.features.pages.PagesError;
    _ = abi.features.pages.PagesConfig;
    _ = abi.features.pages.Page;
    _ = abi.features.pages.PageMatch;
    _ = abi.features.pages.RenderResult;
    _ = abi.features.pages.PagesStats;
    _ = abi.features.pages.Context;
    _ = abi.features.pages.isEnabled;
    _ = abi.features.pages.isInitialized;
}

// ============================================================================
// Benchmarks Feature
// ============================================================================

comptime {
    _ = abi.features.benchmarks.BenchmarksError;
    _ = abi.features.benchmarks.Error;
    _ = abi.features.benchmarks.Config;
    _ = abi.features.benchmarks.BenchmarkFn;
    _ = abi.features.benchmarks.BenchmarkState;
    _ = abi.features.benchmarks.BenchmarkResult;
    _ = abi.features.benchmarks.BenchmarkSuite;
    _ = abi.features.benchmarks.Context;
    _ = abi.features.benchmarks.isEnabled;
    _ = abi.features.benchmarks.isInitialized;
}

// ============================================================================
// Compute Feature
// ============================================================================

comptime {
    _ = abi.features.compute.mesh;
    _ = abi.features.compute.ComputeError;
    _ = abi.features.compute.Error;
    _ = abi.features.compute.Context;
    _ = abi.features.compute.isEnabled;
    _ = abi.features.compute.isInitialized;
}

// ============================================================================
// Documents Feature
// ============================================================================

comptime {
    _ = abi.features.documents.html;
    _ = abi.features.documents.pdf;
    _ = abi.features.documents.DocumentsError;
    _ = abi.features.documents.Error;
    _ = abi.features.documents.Context;
    _ = abi.features.documents.isEnabled;
    _ = abi.features.documents.isInitialized;
}

// ============================================================================
// Desktop Feature
// ============================================================================

comptime {
    _ = abi.features.desktop.macos_menu;
    _ = abi.features.desktop.DesktopError;
    _ = abi.features.desktop.Error;
    _ = abi.features.desktop.Context;
    _ = abi.features.desktop.isEnabled;
    _ = abi.features.desktop.isInitialized;
}
