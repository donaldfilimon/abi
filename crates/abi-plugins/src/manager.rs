//! The plugin manager shared by `abi plugin` and the MCP `plugin_*` tools.
//!
//! Ported from `src/plugins/plugin_manager.zig`.
//!
//! ## Synchronization
//!
//! Zig guarded the manager with an `RwLock` because a single long-lived manager
//! was shared. This port takes `&mut self` for mutations instead and leaves
//! sharing to the caller, matching [`abi_core::registry::Registry`]. Both current
//! call sites — the CLI handler and the MCP `plugin_*` tools — construct a manager
//! per invocation, exactly as the Zig code did, so nothing is shared today.

use std::fmt;
use std::fmt::Write as _;

use abi_foundation::plugin_manifest::{
    self, ManifestCommand, ManifestContextProvider, ManifestError,
};

use crate::{BUNDLED, PluginError};

/// A loaded plugin's metadata.
///
/// The owned counterpart of [`crate::BundledPlugin`]: a plugin loaded from disk
/// has no compiled-in `run`, so this carries metadata only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,
    /// Semantic version.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// The feature gate this plugin targets.
    pub target_feature: String,
    /// Entry-point file name, relative to `path`.
    pub entry_point: String,
    /// The directory this plugin was loaded from.
    pub path: String,
    /// Always `true` for a plugin held by the manager; retained because the Zig
    /// `PluginInfo` exposed it and both surfaces read it.
    pub loaded: bool,
    /// Declared slash-commands.
    pub commands: Vec<ManifestCommand>,
    /// Declared context providers.
    pub context_providers: Vec<ManifestContextProvider>,
}

/// Why loading or unloading a plugin failed.
///
/// The variants Zig declared in `PluginLoadError`, less `OutOfMemory` (Rust
/// aborts on allocation failure rather than surfacing it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginLoadError {
    /// The plugin directory or its `abi-plugin.json` was not found.
    FileNotFound,
    /// The manifest was unparseable, incomplete, or named a missing/unsafe
    /// entry point.
    InvalidManifest(ManifestError),
    /// A plugin with this name is already loaded.
    AlreadyLoaded,
    /// No plugin with this name is loaded.
    NotLoaded,
}

impl fmt::Display for PluginLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound => f.write_str("plugin manifest not found"),
            Self::InvalidManifest(error) => write!(f, "invalid plugin manifest: {error}"),
            Self::AlreadyLoaded => f.write_str("plugin is already loaded"),
            Self::NotLoaded => f.write_str("plugin is not loaded"),
        }
    }
}

impl std::error::Error for PluginLoadError {}

/// Why running a plugin failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginRunError {
    /// No plugin with the requested name is loaded. Zig's
    /// `error.PluginNotFound`.
    PluginNotFound,
    /// The plugin ran but reported a failure.
    Failed(PluginError),
}

impl fmt::Display for PluginRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PluginNotFound => f.write_str("plugin not found"),
            Self::Failed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PluginRunError {}

impl From<PluginError> for PluginRunError {
    fn from(error: PluginError) -> Self {
        Self::Failed(error)
    }
}

/// Loads, lists, and dispatches plugins.
#[derive(Debug, Default)]
pub struct PluginManager {
    /// Insertion-ordered, mirroring Zig's `StringArrayHashMapUnmanaged`. The
    /// order is contract: MCP `plugin_list` emits it.
    plugins: Vec<PluginInfo>,
}

impl PluginManager {
    /// Create an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Load every bundled plugin, skipping any already loaded.
    ///
    /// Reads the compiled-in [`BUNDLED`] table rather than the disk, so the
    /// frozen 16-plugin listing does not depend on the process's working
    /// directory — see the crate docs for why this deviates from Zig.
    pub fn load_bundled(&mut self) {
        for plugin in BUNDLED {
            if self.contains(plugin.name) {
                continue;
            }
            self.plugins.push(PluginInfo {
                name: plugin.name.to_string(),
                version: plugin.version.to_string(),
                description: plugin.description.to_string(),
                target_feature: plugin.target_feature.to_string(),
                entry_point: plugin.entry_point.to_string(),
                path: plugin.path.to_string(),
                loaded: true,
                commands: plugin
                    .commands
                    .iter()
                    .map(|command| ManifestCommand {
                        name: command.name.to_string(),
                        summary: command.summary.to_string(),
                        aliases: command.aliases.iter().map(|&a| a.to_string()).collect(),
                    })
                    .collect(),
                context_providers: plugin
                    .context_providers
                    .iter()
                    .map(|provider| ManifestContextProvider {
                        name: provider.name.to_string(),
                        summary: provider.summary.to_string(),
                    })
                    .collect(),
            });
            abi_telemetry::record("plugin.loaded");
        }
    }

    /// Load an external plugin from a directory containing `abi-plugin.json`,
    /// `mod.rs`, and `stub.rs`.
    pub fn load_plugin(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<PluginInfo, PluginLoadError> {
        let path = path.as_ref();
        let manifest = plugin_manifest::validate(path).map_err(|error| match error {
            ManifestError::MissingDirectory | ManifestError::MissingManifest => {
                PluginLoadError::FileNotFound
            }
            other => PluginLoadError::InvalidManifest(other),
        })?;

        if self.contains(&manifest.name) {
            return Err(PluginLoadError::AlreadyLoaded);
        }

        let info = PluginInfo {
            name: manifest.name,
            version: manifest.version,
            description: manifest.description,
            target_feature: manifest.target_feature,
            entry_point: manifest.entry_point,
            path: path.to_string_lossy().into_owned(),
            loaded: true,
            commands: manifest.commands,
            context_providers: manifest.context_providers,
        };

        self.plugins.push(info.clone());
        abi_telemetry::record("plugin.loaded");
        Ok(info)
    }

    /// Unload the plugin named `name`.
    ///
    /// Uses swap-removal, matching Zig's `swapRemove` — so unloading reorders the
    /// remaining plugins rather than preserving insertion order.
    pub fn unload_plugin(&mut self, name: &str) -> Result<(), PluginLoadError> {
        let index = self.index_of(name).ok_or(PluginLoadError::NotLoaded)?;
        self.plugins.swap_remove(index);
        abi_telemetry::record("plugin.unloaded");
        Ok(())
    }

    /// The plugin named `name`, if loaded.
    pub fn plugin(&self, name: &str) -> Result<&PluginInfo, PluginLoadError> {
        self.index_of(name)
            .map(|index| &self.plugins[index])
            .ok_or(PluginLoadError::NotLoaded)
    }

    /// Every loaded plugin, in load order.
    #[must_use]
    pub fn list(&self) -> &[PluginInfo] {
        &self.plugins
    }

    /// How many plugins are loaded.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Whether any plugin is loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Run the plugin named `name` against `input`.
    ///
    /// A plugin must be loaded to run, even if it is bundled — that check is what
    /// makes `plugin_run` on an unknown name a clean error rather than a panic.
    /// A loaded plugin with no compiled-in implementation (an external directory)
    /// gets Zig's contract-honored placeholder response.
    pub fn run(&self, name: &str, input: &str) -> Result<String, PluginRunError> {
        if !self.contains(name) {
            return Err(PluginRunError::PluginNotFound);
        }
        abi_telemetry::record("plugin.run");

        if let Some(plugin) = crate::bundled(name) {
            return (plugin.run)(input).map_err(PluginRunError::Failed);
        }

        Ok(format!(
            "plugin '{name}' executed (contract honored; input len={}).",
            input.len()
        ))
    }

    /// Collect context snippets from every loaded plugin that declares context
    /// providers.
    ///
    /// Each provider is invoked as `run("__context__:<name>")` and rendered as
    /// `[context:<plugin>:<provider>]\n<snippet>\n[/context]\n`. A provider that
    /// fails is skipped, matching Zig's log-and-continue.
    #[must_use]
    pub fn collect_context_snippets(&self) -> String {
        // Collect the (plugin, provider) pairs before dispatching: `run` borrows
        // `self`, so it cannot be called while iterating `self.plugins`.
        let requests: Vec<(&str, &str)> = self
            .plugins
            .iter()
            .flat_map(|plugin| {
                plugin
                    .context_providers
                    .iter()
                    .map(move |provider| (plugin.name.as_str(), provider.name.as_str()))
            })
            .collect();

        let mut out = String::new();
        for (plugin_name, provider_name) in requests {
            let input = format!("__context__:{provider_name}");
            match self.run(plugin_name, &input) {
                Ok(snippet) => {
                    let _ = writeln!(
                        out,
                        "[context:{plugin_name}:{provider_name}]\n{snippet}\n[/context]"
                    );
                }
                Err(error) => {
                    // Zig used `std.log.warn`, which writes to stderr. There is no
                    // process-global logger in `abi_foundation` (its `Logger` is an
                    // owned, buffered instance), so this goes to stderr directly to
                    // keep the same observable behavior.
                    eprintln!(
                        "warning: context provider {plugin_name}/{provider_name} failed: {error}"
                    );
                }
            }
        }
        out
    }

    fn contains(&self, name: &str) -> bool {
        self.index_of(name).is_some()
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.plugins.iter().position(|plugin| plugin.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_bundled_loads_sixteen_in_declaration_order() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        assert_eq!(manager.plugin_count(), 16);
        assert_eq!(manager.list()[0].name, "example-plugin");
        assert_eq!(manager.list()[2].name, "telemetry-exporter");
    }

    #[test]
    fn load_bundled_is_idempotent() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        manager.load_bundled();
        assert_eq!(
            manager.plugin_count(),
            16,
            "a second call must not duplicate entries"
        );
    }

    #[test]
    fn load_bundled_does_not_depend_on_the_working_directory() {
        // The deliberate deviation from Zig: the listing is compiled in, so a
        // process started outside the repo still reports all sixteen plugins.
        let original = std::env::current_dir().expect("a working directory");
        std::env::set_current_dir(std::env::temp_dir()).expect("chdir to temp");
        let mut manager = PluginManager::new();
        manager.load_bundled();
        std::env::set_current_dir(original).expect("restore the working directory");
        assert_eq!(manager.plugin_count(), 16);
    }

    #[test]
    fn unload_then_reload_restores_the_plugin() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        manager.unload_plugin("tui-plugin").expect("loaded");
        assert_eq!(manager.plugin_count(), 15);
        assert_eq!(
            manager.plugin("tui-plugin"),
            Err(PluginLoadError::NotLoaded)
        );

        manager.load_bundled();
        assert_eq!(manager.plugin_count(), 16);
        assert!(manager.plugin("tui-plugin").is_ok());
    }

    #[test]
    fn unload_uses_swap_removal_like_zig() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        let last = manager.list()[15].name.clone();
        manager.unload_plugin("example-plugin").expect("loaded");
        // Zig's swapRemove moved the final entry into the freed slot; preserving
        // that keeps the two ports' post-unload ordering identical.
        assert_eq!(manager.list()[0].name, last);
    }

    #[test]
    fn unload_unknown_plugin_reports_not_loaded() {
        let mut manager = PluginManager::new();
        assert_eq!(
            manager.unload_plugin("ghost"),
            Err(PluginLoadError::NotLoaded)
        );
    }

    #[test]
    fn run_requires_the_plugin_to_be_loaded() {
        let manager = PluginManager::new();
        assert_eq!(
            manager.run("example-plugin", ""),
            Err(PluginRunError::PluginNotFound),
            "bundled but unloaded is still not runnable"
        );
    }

    #[test]
    fn run_dispatches_to_the_bundled_implementation() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        // The exact string pinned by tests/golden/mcp-tool-calls-args.jsonl.
        assert_eq!(
            manager.run("example-plugin", "").unwrap(),
            "example-plugin received input (len=0)"
        );
        assert_eq!(
            manager.run("tui-plugin", "__cmd__:ping").unwrap(),
            "tui-plugin: pong"
        );
    }

    #[test]
    fn run_on_unknown_name_reports_plugin_not_found() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        assert_eq!(
            manager.run("no-such-plugin", ""),
            Err(PluginRunError::PluginNotFound)
        );
    }

    #[test]
    fn collect_context_snippets_renders_the_declared_provider() {
        let mut manager = PluginManager::new();
        manager.load_bundled();
        // example-plugin is the only bundled plugin declaring a provider.
        assert_eq!(
            manager.collect_context_snippets(),
            "[context:example-plugin:example-info]\nexample-plugin v0.1.0 (target: plugins)\n[/context]\n"
        );
    }

    #[test]
    fn collect_context_snippets_is_empty_without_providers() {
        let manager = PluginManager::new();
        assert_eq!(manager.collect_context_snippets(), "");
    }

    #[test]
    fn load_plugin_reads_a_real_bundled_directory() {
        // Exercises the on-disk path against a manifest that ships in-tree.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/plugins/example-plugin");
        let mut manager = PluginManager::new();
        let info = manager.load_plugin(dir).expect("valid plugin directory");
        assert_eq!(info.name, "example-plugin");
        assert_eq!(info.entry_point, "mod.rs");
        assert_eq!(info.commands.len(), 1);
        assert_eq!(info.commands[0].name, "echo");
        assert_eq!(info.commands[0].aliases, vec!["say".to_string()]);
        assert_eq!(info.context_providers.len(), 1);
        assert_eq!(info.context_providers[0].name, "example-info");

        assert_eq!(
            manager.load_plugin(dir),
            Err(PluginLoadError::AlreadyLoaded)
        );
    }

    #[test]
    fn load_plugin_reports_a_missing_directory_as_file_not_found() {
        let mut manager = PluginManager::new();
        assert_eq!(
            manager.load_plugin("/nonexistent/plugin/directory"),
            Err(PluginLoadError::FileNotFound)
        );
    }
}
