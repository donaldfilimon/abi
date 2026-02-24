# framework

> Framework orchestration with builder pattern.

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-framework"></a>`pub const Framework`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L148)

Framework orchestration handle.

The Framework struct is the central coordinator for the ABI framework. It manages
the lifecycle of all enabled feature modules, provides access to their contexts,
and maintains the framework's runtime state.

## Thread Safety

The Framework itself is not thread-safe. If you need to access the framework from
multiple threads, you should use external synchronization or ensure each thread
has its own Framework instance.

## Memory Management

The Framework allocates memory for feature contexts during initialization. All
allocated memory is released when `deinit()` is called. The caller must ensure
the provided allocator remains valid for the lifetime of the Framework.

## Example

```zig
var fw = try Framework.init(allocator, Config.defaults());
defer fw.deinit();

// Check state
if (fw.isRunning()) {
// Access features
if (fw.gpu) |gpu_ctx| {
// Use GPU...
}
}
```

### <a id="pub-const-state"></a>`pub const State`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L216)

Framework lifecycle states.

### <a id="pub-const-error"></a>`pub const Error`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L220)

Composable framework error set.
See `core/errors.zig` for the full hierarchy.

### <a id="pub-fn-init-allocator-std-mem-allocator-cfg-config-error-framework"></a>`pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L253)

Initialize the framework with the given configuration.

This is the primary initialization method for the Framework. It validates the
configuration, initializes all enabled feature modules, and transitions the
framework to the `running` state.

## Parameters

- `allocator`: Memory allocator for framework resources. Must remain valid for
the lifetime of the Framework.
- `cfg`: Configuration specifying which features to enable and their settings.

## Returns

A fully initialized Framework instance in the `running` state.

## Errors

- `ConfigError.FeatureDisabled`: A feature is enabled in config but disabled at compile time
- `error.OutOfMemory`: Memory allocation failed
- `error.FeatureInitFailed`: A feature module failed to initialize

## Example

```zig
var fw = try Framework.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.database = .{ .path = "./data" },
});
defer fw.deinit();
```

### <a id="pub-fn-initwithio-allocator-std-mem-allocator-cfg-config-io-std-io-error-framework"></a>`pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L259)

Initialize the framework with the given configuration **and** an I/O backend.
This method is used by the builder when `withIo` is supplied.

### <a id="pub-fn-initdefault-allocator-std-mem-allocator-error-framework"></a>`pub fn initDefault(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L282)

Create a framework with default configuration.

This is a convenience method that creates a framework with all compile-time
enabled features also enabled at runtime with their default settings.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A Framework instance with default configuration.

## Example

```zig
var fw = try Framework.initDefault(allocator);
defer fw.deinit();
```

### <a id="pub-fn-initminimal-allocator-std-mem-allocator-error-framework"></a>`pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L310)

Create a framework with minimal configuration (no features enabled).

This creates a framework with no optional features enabled. Only the
runtime context is initialized. Useful for testing or when you want
to explicitly enable specific features.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A Framework instance with minimal configuration.

## Example

```zig
var fw = try Framework.initMinimal(allocator);
defer fw.deinit();

// Only runtime is available, no features enabled
try std.testing.expect(fw.gpu == null);
try std.testing.expect(fw.ai == null);
```

### <a id="pub-fn-builder-allocator-std-mem-allocator-frameworkbuilder"></a>`pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L336)

Start building a framework configuration.

Returns a FrameworkBuilder that provides a fluent API for configuring
and initializing the framework.

## Parameters

- `allocator`: Memory allocator for framework resources

## Returns

A FrameworkBuilder instance for configuring the framework.

## Example

```zig
var fw = try Framework.builder(allocator)
.withGpuDefaults()
.withAi(.{ .llm = .{} })
.build();
defer fw.deinit();
```

### <a id="pub-fn-deinit-self-framework-void"></a>`pub fn deinit(self: *Framework) void`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L358)

Shutdown and cleanup the framework.

This method transitions the framework to the `stopping` state, deinitializes
all feature contexts in reverse order of initialization, cleans up the registry,
and finally transitions to `stopped`.

After calling `deinit()`, the framework instance should not be used. Any
pointers to feature contexts become invalid.

This method is idempotent - calling it multiple times is safe.

## Example

```zig
var fw = try Framework.initDefault(allocator);
// ... use framework ...
fw.deinit();  // Clean up all resources
```

### <a id="pub-fn-shutdownwithtimeout-self-framework-timeout-ms-u64-bool"></a>`pub fn shutdownWithTimeout(self: *Framework, timeout_ms: u64) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L364)

Shutdown with timeout. Currently synchronous (timeout reserved for
future async cleanup). Returns true if clean shutdown completed.

### <a id="pub-fn-isrunning-self-const-framework-bool"></a>`pub fn isRunning(self: *const Framework) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L373)

Check if the framework is running.

### <a id="pub-fn-isenabled-self-const-framework-feature-feature-bool"></a>`pub fn isEnabled(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L378)

Check if a feature is enabled.

### <a id="pub-fn-getstate-self-const-framework-state"></a>`pub fn getState(self: *const Framework) State`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L383)

Get the current framework state.

### <a id="pub-fn-getgpu-self-framework-error-gpu-mod-context"></a>`pub fn getGpu(self: *Framework) Error!*gpu_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L392)

Get GPU context (returns error if not enabled).

### <a id="pub-fn-getai-self-framework-error-ai-mod-context"></a>`pub fn getAi(self: *Framework) Error!*ai_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L397)

Get AI context (returns error if not enabled).

### <a id="pub-fn-getdatabase-self-framework-error-database-mod-context"></a>`pub fn getDatabase(self: *Framework) Error!*database_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L402)

Get database context (returns error if not enabled).

### <a id="pub-fn-getnetwork-self-framework-error-network-mod-context"></a>`pub fn getNetwork(self: *Framework) Error!*network_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L407)

Get network context (returns error if not enabled).

### <a id="pub-fn-getobservability-self-framework-error-observability-mod-context"></a>`pub fn getObservability(self: *Framework) Error!*observability_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L412)

Get observability context (returns error if not enabled).

### <a id="pub-fn-getweb-self-framework-error-web-mod-context"></a>`pub fn getWeb(self: *Framework) Error!*web_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L417)

Get web context (returns error if not enabled).

### <a id="pub-fn-getcloud-self-framework-error-cloud-mod-context"></a>`pub fn getCloud(self: *Framework) Error!*cloud_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L422)

Get cloud context (returns error if not enabled).

### <a id="pub-fn-getanalytics-self-framework-error-analytics-mod-context"></a>`pub fn getAnalytics(self: *Framework) Error!*analytics_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L427)

Get analytics context (returns error if not enabled).

### <a id="pub-fn-getauth-self-framework-error-auth-mod-context"></a>`pub fn getAuth(self: *Framework) Error!*auth_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L432)

Get auth context (returns error if not enabled).

### <a id="pub-fn-getmessaging-self-framework-error-messaging-mod-context"></a>`pub fn getMessaging(self: *Framework) Error!*messaging_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L437)

Get messaging context (returns error if not enabled).

### <a id="pub-fn-getcache-self-framework-error-cache-mod-context"></a>`pub fn getCache(self: *Framework) Error!*cache_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L442)

Get cache context (returns error if not enabled).

### <a id="pub-fn-getstorage-self-framework-error-storage-mod-context"></a>`pub fn getStorage(self: *Framework) Error!*storage_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L447)

Get storage context (returns error if not enabled).

### <a id="pub-fn-getsearch-self-framework-error-search-mod-context"></a>`pub fn getSearch(self: *Framework) Error!*search_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L452)

Get search context (returns error if not enabled).

### <a id="pub-fn-getgateway-self-framework-error-gateway-mod-context"></a>`pub fn getGateway(self: *Framework) Error!*gateway_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L457)

Get gateway context (returns error if not enabled).

### <a id="pub-fn-getpages-self-framework-error-pages-mod-context"></a>`pub fn getPages(self: *Framework) Error!*pages_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L462)

Get pages context (returns error if not enabled).

### <a id="pub-fn-getbenchmarks-self-framework-error-benchmarks-mod-context"></a>`pub fn getBenchmarks(self: *Framework) Error!*benchmarks_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L467)

Get benchmarks context (returns error if not enabled).

### <a id="pub-fn-getmobile-self-framework-error-mobile-mod-context"></a>`pub fn getMobile(self: *Framework) Error!*mobile_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L472)

Get mobile context (returns error if not enabled).

### <a id="pub-fn-getaicore-self-framework-error-ai-core-mod-context"></a>`pub fn getAiCore(self: *Framework) Error!*ai_core_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L477)

Get AI core context (agents, tools, prompts).

### <a id="pub-fn-getaiinference-self-framework-error-ai-inference-mod-context"></a>`pub fn getAiInference(self: *Framework) Error!*ai_inference_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L482)

Get AI inference context (LLM, embeddings, vision).

### <a id="pub-fn-getaitraining-self-framework-error-ai-training-mod-context"></a>`pub fn getAiTraining(self: *Framework) Error!*ai_training_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L487)

Get AI training context (pipelines, federated).

### <a id="pub-fn-getaireasoning-self-framework-error-ai-reasoning-mod-context"></a>`pub fn getAiReasoning(self: *Framework) Error!*ai_reasoning_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L492)

Get AI reasoning context (Abbey, RAG, eval).

### <a id="pub-fn-getruntime-self-framework-runtime-mod-context"></a>`pub fn getRuntime(self: *Framework) *runtime_mod.Context`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L497)

Get runtime context (always available).

### <a id="pub-fn-getregistry-self-framework-registry"></a>`pub fn getRegistry(self: *Framework) *Registry`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L502)

Get the feature registry for runtime feature management.

### <a id="pub-fn-isfeatureregistered-self-const-framework-feature-feature-bool"></a>`pub fn isFeatureRegistered(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L507)

Check if a feature is registered in the registry.

### <a id="pub-fn-listregisteredfeatures-self-const-framework-allocator-std-mem-allocator-registryerror-feature"></a>`pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) RegistryError![]Feature`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L512)

List all registered features.

### <a id="pub-const-frameworkbuilder"></a>`pub const FrameworkBuilder`

<sup>**const**</sup> | [source](../../src/core/framework.zig#L518)

Fluent builder for Framework initialization.

### <a id="pub-fn-withdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L530)

Start with default configuration.

### <a id="pub-fn-withgpu-self-frameworkbuilder-gpu-config-config-module-gpuconfig-frameworkbuilder"></a>`pub fn withGpu(self: *FrameworkBuilder, gpu_config: config_module.GpuConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L535)

Enable GPU with configuration.

### <a id="pub-fn-withgpudefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withGpuDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L540)

Enable GPU with defaults.

### <a id="pub-fn-withio-self-frameworkbuilder-io-std-io-frameworkbuilder"></a>`pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L546)

Provide a shared I/O backend for the framework.
Pass the `std.Io` obtained from `IoBackend.init`.

### <a id="pub-fn-withai-self-frameworkbuilder-ai-config-config-module-aiconfig-frameworkbuilder"></a>`pub fn withAi(self: *FrameworkBuilder, ai_config: config_module.AiConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L551)

Enable AI with configuration.

### <a id="pub-fn-withaidefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withAiDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L556)

Enable AI with defaults.

### <a id="pub-fn-withllm-self-frameworkbuilder-llm-config-config-module-llmconfig-frameworkbuilder"></a>`pub fn withLlm(self: *FrameworkBuilder, llm_config: config_module.LlmConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L561)

Enable LLM only.

### <a id="pub-fn-withdatabase-self-frameworkbuilder-db-config-config-module-databaseconfig-frameworkbuilder"></a>`pub fn withDatabase(self: *FrameworkBuilder, db_config: config_module.DatabaseConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L566)

Enable database with configuration.

### <a id="pub-fn-withdatabasedefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withDatabaseDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L571)

Enable database with defaults.

### <a id="pub-fn-withnetwork-self-frameworkbuilder-net-config-config-module-networkconfig-frameworkbuilder"></a>`pub fn withNetwork(self: *FrameworkBuilder, net_config: config_module.NetworkConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L576)

Enable network with configuration.

### <a id="pub-fn-withnetworkdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withNetworkDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L581)

Enable network with defaults.

### <a id="pub-fn-withobservability-self-frameworkbuilder-obs-config-config-module-observabilityconfig-frameworkbuilder"></a>`pub fn withObservability(self: *FrameworkBuilder, obs_config: config_module.ObservabilityConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L586)

Enable observability with configuration.

### <a id="pub-fn-withobservabilitydefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withObservabilityDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L591)

Enable observability with defaults.

### <a id="pub-fn-withweb-self-frameworkbuilder-web-config-config-module-webconfig-frameworkbuilder"></a>`pub fn withWeb(self: *FrameworkBuilder, web_config: config_module.WebConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L596)

Enable web with configuration.

### <a id="pub-fn-withwebdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withWebDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L601)

Enable web with defaults.

### <a id="pub-fn-withanalytics-self-frameworkbuilder-analytics-cfg-config-module-analyticsconfig-frameworkbuilder"></a>`pub fn withAnalytics(self: *FrameworkBuilder, analytics_cfg: config_module.AnalyticsConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L606)

Enable analytics with configuration.

### <a id="pub-fn-withanalyticsdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withAnalyticsDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L611)

Enable analytics with defaults.

### <a id="pub-fn-withcloud-self-frameworkbuilder-cloud-config-config-module-cloudconfig-frameworkbuilder"></a>`pub fn withCloud(self: *FrameworkBuilder, cloud_config: config_module.CloudConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L616)

Enable cloud with configuration.

### <a id="pub-fn-withclouddefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withCloudDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L621)

Enable cloud with defaults.

### <a id="pub-fn-withauth-self-frameworkbuilder-auth-config-config-module-authconfig-frameworkbuilder"></a>`pub fn withAuth(self: *FrameworkBuilder, auth_config: config_module.AuthConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L626)

Enable auth with configuration.

### <a id="pub-fn-withauthdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withAuthDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L631)

Enable auth with defaults.

### <a id="pub-fn-withmessaging-self-frameworkbuilder-msg-config-config-module-messagingconfig-frameworkbuilder"></a>`pub fn withMessaging(self: *FrameworkBuilder, msg_config: config_module.MessagingConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L636)

Enable messaging with configuration.

### <a id="pub-fn-withmessagingdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withMessagingDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L641)

Enable messaging with defaults.

### <a id="pub-fn-withcache-self-frameworkbuilder-cache-config-config-module-cacheconfig-frameworkbuilder"></a>`pub fn withCache(self: *FrameworkBuilder, cache_config: config_module.CacheConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L646)

Enable cache with configuration.

### <a id="pub-fn-withcachedefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withCacheDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L651)

Enable cache with defaults.

### <a id="pub-fn-withstorage-self-frameworkbuilder-storage-config-config-module-storageconfig-frameworkbuilder"></a>`pub fn withStorage(self: *FrameworkBuilder, storage_config: config_module.StorageConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L656)

Enable storage with configuration.

### <a id="pub-fn-withstoragedefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withStorageDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L661)

Enable storage with defaults.

### <a id="pub-fn-withsearch-self-frameworkbuilder-search-config-config-module-searchconfig-frameworkbuilder"></a>`pub fn withSearch(self: *FrameworkBuilder, search_config: config_module.SearchConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L666)

Enable search with configuration.

### <a id="pub-fn-withsearchdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withSearchDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L671)

Enable search with defaults.

### <a id="pub-fn-withgateway-self-frameworkbuilder-gateway-cfg-config-module-gatewayconfig-frameworkbuilder"></a>`pub fn withGateway(self: *FrameworkBuilder, gateway_cfg: config_module.GatewayConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L676)

Enable gateway with configuration.

### <a id="pub-fn-withgatewaydefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withGatewayDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L681)

Enable gateway with defaults.

### <a id="pub-fn-withpages-self-frameworkbuilder-pages-cfg-config-module-pagesconfig-frameworkbuilder"></a>`pub fn withPages(self: *FrameworkBuilder, pages_cfg: config_module.PagesConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L686)

Enable pages with configuration.

### <a id="pub-fn-withpagesdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withPagesDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L691)

Enable pages with defaults.

### <a id="pub-fn-withbenchmarks-self-frameworkbuilder-benchmarks-cfg-config-module-benchmarksconfig-frameworkbuilder"></a>`pub fn withBenchmarks(self: *FrameworkBuilder, benchmarks_cfg: config_module.BenchmarksConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L696)

Enable benchmarks with configuration.

### <a id="pub-fn-withbenchmarksdefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withBenchmarksDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L701)

Enable benchmarks with defaults.

### <a id="pub-fn-withmobile-self-frameworkbuilder-mobile-cfg-config-module-mobileconfig-frameworkbuilder"></a>`pub fn withMobile(self: *FrameworkBuilder, mobile_cfg: config_module.MobileConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L706)

Enable mobile with configuration.

### <a id="pub-fn-withmobiledefaults-self-frameworkbuilder-frameworkbuilder"></a>`pub fn withMobileDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L711)

Enable mobile with defaults.

### <a id="pub-fn-withplugins-self-frameworkbuilder-plugin-config-config-module-pluginconfig-frameworkbuilder"></a>`pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L716)

Configure plugins.

### <a id="pub-fn-build-self-frameworkbuilder-framework-error-framework"></a>`pub fn build(self: *FrameworkBuilder) Framework.Error!Framework`

<sup>**fn**</sup> | [source](../../src/core/framework.zig#L723)

Build and initialize the framework.
If an I/O backend was supplied via `withIo`, it will be stored in the
resulting `Framework` instance.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
