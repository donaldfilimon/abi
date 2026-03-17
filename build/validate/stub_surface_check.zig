//! Stub Surface Check
//!
//! Compiled under every flag combination in the validation matrix to ensure
//! that mod.zig and stub.zig expose the same public API surface. This file
//! dereferences key public symbols from each canonical namespace. If a stub
//! is missing a symbol that the mod exports (or vice versa), the compile
//! will fail for the relevant flag combo, catching mod/stub drift early.
//!
//! This file is intentionally a library (no main/entry point). The build
//! system compiles it as a static library under each flag combination.

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
    _ = abi.config;
    _ = abi.errors;
    _ = abi.registry;
    _ = abi.framework;
    _ = abi.foundation;
    _ = abi.runtime;
    _ = abi.platform;
    _ = abi.connectors;
    _ = abi.tasks;
    _ = abi.mcp;
    _ = abi.lsp;
    _ = abi.acp;
    _ = abi.ha;
    _ = abi.gpu;
    _ = abi.ai;
    _ = abi.inference;
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
// Inference Surface
// ============================================================================

comptime {
    _ = abi.inference.Engine;
    _ = abi.inference.EngineConfig;
    _ = abi.inference.EngineResult;
    _ = abi.inference.EngineStats;
    _ = abi.inference.FinishReason;
    _ = abi.inference.Scheduler;
    _ = abi.inference.Request;
    _ = abi.inference.Sampler;
    _ = abi.inference.SamplingParams;
    _ = abi.inference.PagedKVCache;
    _ = abi.inference.PagedKVCacheConfig;
}

// ============================================================================
// GPU Feature
// ============================================================================

comptime {
    _ = abi.gpu.Gpu;
    _ = abi.gpu.Backend;
    _ = abi.gpu.Device;
    _ = abi.gpu.Buffer;
    _ = abi.gpu.Stream;
    _ = abi.gpu.GpuConfig;
    _ = abi.gpu.Error;
    _ = abi.gpu.GpuDevice;
    _ = abi.gpu.Context;
    _ = abi.gpu.backends;
    _ = abi.gpu.diagnostics;
    _ = abi.gpu.platform;
    _ = abi.Gpu;
    _ = abi.GpuBackend;
}

// ============================================================================
// AI Feature
// ============================================================================

comptime {
    _ = abi.ai.Error;
    _ = abi.ai.Context;
    _ = abi.ai.core;
    _ = abi.ai.llm;
    _ = abi.ai.agents;
    _ = abi.ai.training;
    _ = abi.ai.streaming;
    _ = abi.ai.profiles;
    _ = abi.ai.coordination;
    _ = abi.ai.abbey;
    _ = abi.ai.explore;
    _ = abi.ai.rag;
    _ = abi.ai.eval;
    _ = abi.ai.constitution;
    _ = abi.ai.transformer;
    _ = abi.ai.tools;
    _ = abi.ai.models;
    _ = abi.ai.memory;
    _ = abi.ai.documents;
    _ = abi.ai.vision;
    _ = abi.ai.embeddings;
    _ = abi.ai.orchestration;
    _ = abi.ai.multi_agent;
    _ = abi.ai.templates;
    _ = abi.ai.federated;
    _ = abi.ai.prompts;
    _ = abi.ai.database;
    _ = abi.ai.isEnabled;
    _ = abi.ai.isInitialized;
}

// ============================================================================
// Database Feature
// ============================================================================

comptime {
    _ = abi.database.DatabaseHandle;
    _ = abi.database.SearchResult;
    _ = abi.database.Context;
    _ = abi.database.semantic_store;
    _ = abi.database.isEnabled;
    _ = abi.database.isInitialized;
    _ = abi.database.Stats;
}

// ============================================================================
// Network Feature
// ============================================================================

comptime {
    _ = abi.network.Error;
    _ = abi.network.Context;
    _ = abi.network.NetworkConfig;
    _ = abi.network.Node;
    _ = abi.network.RaftNode;
    _ = abi.network.CircuitBreaker;
    _ = abi.network.LoadBalancer;
    _ = abi.network.RetryConfig;
    _ = abi.network.ConnectionPool;
    _ = abi.network.isEnabled;
    _ = abi.network.isInitialized;
}

// ============================================================================
// Observability Feature
// ============================================================================

comptime {
    _ = abi.observability.Error;
    _ = abi.observability.Context;
    _ = abi.observability.MetricsCollector;
    _ = abi.observability.Counter;
    _ = abi.observability.Gauge;
    _ = abi.observability.Histogram;
    _ = abi.observability.Tracer;
    _ = abi.observability.Span;
    _ = abi.observability.AlertManager;
    _ = abi.observability.isEnabled;
    _ = abi.observability.isInitialized;
}

// ============================================================================
// Web Feature
// ============================================================================

comptime {
    _ = abi.web.Context;
    _ = abi.web.ChatHandler;
    _ = abi.web.HttpClient;
    _ = abi.web.Response;
    _ = abi.web.server;
    _ = abi.web.middleware;
    _ = abi.web.isEnabled;
    _ = abi.web.isInitialized;
}

// ============================================================================
// Analytics Feature
// ============================================================================

comptime {
    _ = abi.analytics.Engine;
    _ = abi.analytics.Event;
    _ = abi.analytics.Error;
    _ = abi.analytics.AnalyticsConfig;
    _ = abi.analytics.Context;
    _ = abi.analytics.Funnel;
    _ = abi.analytics.Experiment;
    _ = abi.analytics.isEnabled;
    _ = abi.analytics.isInitialized;
}

// ============================================================================
// Cloud Feature
// ============================================================================

comptime {
    _ = abi.cloud.Error;
    _ = abi.cloud.Context;
    _ = abi.cloud.CloudProvider;
    _ = abi.cloud.CloudConfig;
    _ = abi.cloud.CloudEvent;
    _ = abi.cloud.CloudResponse;
    _ = abi.cloud.ResponseBuilder;
    _ = abi.cloud.isEnabled;
    _ = abi.cloud.isInitialized;
}

// ============================================================================
// Auth Feature
// ============================================================================

comptime {
    _ = abi.auth.AuthError;
    _ = abi.auth.Error;
    _ = abi.auth.AuthConfig;
    _ = abi.auth.Token;
    _ = abi.auth.Session;
    _ = abi.auth.Permission;
    _ = abi.auth.Context;
    _ = abi.auth.isEnabled;
    _ = abi.auth.isInitialized;
}

// ============================================================================
// Messaging Feature
// ============================================================================

comptime {
    _ = abi.messaging.MessagingError;
    _ = abi.messaging.Error;
    _ = abi.messaging.MessagingConfig;
    _ = abi.messaging.Message;
    _ = abi.messaging.Channel;
    _ = abi.messaging.MessagingStats;
    _ = abi.messaging.Context;
    _ = abi.messaging.isEnabled;
    _ = abi.messaging.isInitialized;
}

// ============================================================================
// Cache Feature
// ============================================================================

comptime {
    _ = abi.cache.CacheError;
    _ = abi.cache.Error;
    _ = abi.cache.CacheConfig;
    _ = abi.cache.CacheEntry;
    _ = abi.cache.CacheStats;
    _ = abi.cache.Context;
    _ = abi.cache.isEnabled;
    _ = abi.cache.isInitialized;
}

// ============================================================================
// Storage Feature
// ============================================================================

comptime {
    _ = abi.storage.StorageError;
    _ = abi.storage.Error;
    _ = abi.storage.StorageConfig;
    _ = abi.storage.StorageObject;
    _ = abi.storage.ObjectMetadata;
    _ = abi.storage.StorageStats;
    _ = abi.storage.Context;
    _ = abi.storage.isEnabled;
    _ = abi.storage.isInitialized;
}

// ============================================================================
// Search Feature
// ============================================================================

comptime {
    _ = abi.search.SearchError;
    _ = abi.search.Error;
    _ = abi.search.SearchConfig;
    _ = abi.search.SearchResult;
    _ = abi.search.SearchIndex;
    _ = abi.search.SearchStats;
    _ = abi.search.Context;
    _ = abi.search.isEnabled;
    _ = abi.search.isInitialized;
}

// ============================================================================
// Mobile Feature
// ============================================================================

comptime {
    _ = abi.mobile.MobileError;
    _ = abi.mobile.Error;
    _ = abi.mobile.MobileConfig;
    _ = abi.mobile.SensorType;
    _ = abi.mobile.SensorData;
    _ = abi.mobile.Notification;
    _ = abi.mobile.Permission;
    _ = abi.mobile.PermissionStatus;
    _ = abi.mobile.DeviceInfo;
    _ = abi.mobile.Context;
    _ = abi.mobile.isEnabled;
    _ = abi.mobile.isInitialized;
}

// ============================================================================
// Gateway Feature
// ============================================================================

comptime {
    _ = abi.gateway.GatewayError;
    _ = abi.gateway.Error;
    _ = abi.gateway.GatewayConfig;
    _ = abi.gateway.Route;
    _ = abi.gateway.MatchResult;
    _ = abi.gateway.GatewayStats;
    _ = abi.gateway.Context;
    _ = abi.gateway.isEnabled;
    _ = abi.gateway.isInitialized;
}

// ============================================================================
// Pages Feature
// ============================================================================

comptime {
    _ = abi.pages.PagesError;
    _ = abi.pages.PagesConfig;
    _ = abi.pages.Page;
    _ = abi.pages.PageMatch;
    _ = abi.pages.RenderResult;
    _ = abi.pages.PagesStats;
    _ = abi.pages.Context;
    _ = abi.pages.isEnabled;
    _ = abi.pages.isInitialized;
}

// ============================================================================
// Benchmarks Feature
// ============================================================================

comptime {
    _ = abi.benchmarks.BenchmarksError;
    _ = abi.benchmarks.Error;
    _ = abi.benchmarks.Config;
    _ = abi.benchmarks.BenchmarkFn;
    _ = abi.benchmarks.BenchmarkState;
    _ = abi.benchmarks.BenchmarkResult;
    _ = abi.benchmarks.BenchmarkSuite;
    _ = abi.benchmarks.Context;
    _ = abi.benchmarks.isEnabled;
    _ = abi.benchmarks.isInitialized;
}

// ============================================================================
// Compute Feature
// ============================================================================

comptime {
    _ = abi.compute.mesh;
    _ = abi.compute.ComputeError;
    _ = abi.compute.Error;
    _ = abi.compute.Context;
    _ = abi.compute.isEnabled;
    _ = abi.compute.isInitialized;
}

// ============================================================================
// Documents Feature
// ============================================================================

comptime {
    _ = abi.documents.html;
    _ = abi.documents.pdf;
    _ = abi.documents.DocumentsError;
    _ = abi.documents.Error;
    _ = abi.documents.Context;
    _ = abi.documents.isEnabled;
    _ = abi.documents.isInitialized;
}

// ============================================================================
// Desktop Feature
// ============================================================================

comptime {
    _ = abi.desktop.macos_menu;
    _ = abi.desktop.DesktopError;
    _ = abi.desktop.Error;
    _ = abi.desktop.Context;
    _ = abi.desktop.isEnabled;
    _ = abi.desktop.isInitialized;
}
