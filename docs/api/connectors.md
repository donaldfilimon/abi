# connectors

> External service connectors (OpenAI, Anthropic, Ollama, etc.).

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-fn-init-std-mem-allocator-void"></a>`pub fn init(_: std.mem.Allocator) !void`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L60)

Initialize the connectors subsystem (idempotent; no-op if already initialized).

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L65)

Tear down the connectors subsystem; safe to call multiple times.

### <a id="pub-fn-isenabled-bool"></a>`pub fn isEnabled() bool`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L70)

Returns true; connectors are always available when this module is compiled in.

### <a id="pub-fn-isinitialized-bool"></a>`pub fn isInitialized() bool`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L75)

Returns true after `init()` has been called.

### <a id="pub-fn-getenvowned-allocator-std-mem-allocator-name-const-u8-u8"></a>`pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L85)

Read environment variable by name; returns owned slice or null if unset. Caller must free.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
