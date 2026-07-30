//! The plugin registry.
//!
//! Ported from `src/core/registry.zig`. Plugins declare a descriptor plus
//! optional slash-commands and context providers; the registry enforces that no
//! two plugins claim the same command token or context-provider name.
//!
//! Two structural notes:
//!
//! - **Insertion order is preserved.** Zig used a `StringArrayHashMapUnmanaged`
//!   precisely for that, and `abi plugin list` output order depends on it. A
//!   `HashMap` here would scramble the listing between runs.
//! - Zig needed ~90 lines of `dupeDescriptor`/`freeDescriptor` to deep-copy every
//!   nested slice and unwind partial copies on allocation failure. Owned `String`
//!   and `Vec` make that whole layer disappear, along with its leak paths.

use std::fmt::Write as _;

/// A slash-command contributed by a plugin.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PluginCommand {
    /// Primary token.
    pub name: String,
    /// One-line description.
    pub summary: String,
    /// Alternative tokens that invoke the same command.
    pub aliases: Vec<String>,
}

impl PluginCommand {
    /// A command with just a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            summary: String::new(),
            aliases: Vec::new(),
        }
    }

    /// Set the summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Set the aliases.
    #[must_use]
    pub fn with_aliases<I, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    /// Whether `token` names this command, as its primary name or an alias.
    #[must_use]
    pub fn matches(&self, token: &str) -> bool {
        self.name == token || self.aliases.iter().any(|alias| alias == token)
    }

    /// Every token that invokes this command.
    pub fn tokens(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

/// A context provider that augments the REPL prompt context at startup.
///
/// Invoked through the plugin's entry point as `__context__:<name>`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContextProvider {
    /// Provider name, as used in the `__context__:` token.
    pub name: String,
    /// One-line description.
    pub summary: String,
}

impl ContextProvider {
    /// A provider with just a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            summary: String::new(),
        }
    }
}

/// Everything the registry knows about one plugin.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PluginDescriptor {
    /// Unique plugin name.
    pub name: String,
    /// Semantic version, or empty if unknown.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// The feature gate this plugin targets.
    pub target_feature: String,
    /// Entry point path, relative to the plugin directory.
    pub entry_point: String,
    /// Slash-commands this plugin registers.
    pub commands: Vec<PluginCommand>,
    /// Context providers this plugin registers.
    pub context_providers: Vec<ContextProvider>,
}

impl PluginDescriptor {
    /// A descriptor with a name and description, matching Zig's `register`.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            ..Self::default()
        }
    }

    /// Set the version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the target feature gate.
    #[must_use]
    pub fn with_target_feature(mut self, target_feature: impl Into<String>) -> Self {
        self.target_feature = target_feature.into();
        self
    }

    /// Set the entry point.
    #[must_use]
    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    /// Set the commands.
    #[must_use]
    pub fn with_commands(mut self, commands: Vec<PluginCommand>) -> Self {
        self.commands = commands;
        self
    }

    /// Set the context providers.
    #[must_use]
    pub fn with_context_providers(mut self, providers: Vec<ContextProvider>) -> Self {
        self.context_providers = providers;
        self
    }
}

/// Why a registration was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Another plugin already claims this command token.
    DuplicateCommand {
        /// The contested token.
        token: String,
        /// The plugin that already claims it.
        owned_by: String,
    },
    /// Another plugin already claims this context-provider name.
    DuplicateContextProvider {
        /// The contested provider name.
        name: String,
        /// The plugin that already claims it.
        owned_by: String,
    },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCommand { token, owned_by } => {
                write!(f, "command {token:?} is already registered by {owned_by:?}")
            }
            Self::DuplicateContextProvider { name, owned_by } => {
                write!(
                    f,
                    "context provider {name:?} is already registered by {owned_by:?}"
                )
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// A command found by lookup, together with its owning plugin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    /// The plugin that registered the command.
    pub plugin: String,
    /// The command itself.
    pub command: PluginCommand,
}

/// The plugin registry.
///
/// Insertion-ordered. Not internally synchronized: Zig guarded it with an
/// `RwLock` because the registry was a process-global, but here ownership is
/// explicit — wrap it in `Arc<RwLock<Registry>>` if it must be shared.
#[derive(Debug, Default, Clone)]
pub struct Registry {
    plugins: Vec<PluginDescriptor>,
}

impl Registry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin by name and description only.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<(), RegistryError> {
        self.register_plugin(PluginDescriptor::new(name, description))
    }

    /// Register a plugin descriptor.
    ///
    /// Re-registering the same *name* replaces the existing entry in place,
    /// keeping its position in the listing — and is not treated as a collision
    /// with itself. Collisions are only checked against *other* plugins.
    pub fn register_plugin(&mut self, descriptor: PluginDescriptor) -> Result<(), RegistryError> {
        self.check_no_collision(&descriptor)?;

        if let Some(existing) = self
            .plugins
            .iter_mut()
            .find(|plugin| plugin.name == descriptor.name)
        {
            *existing = descriptor;
        } else {
            self.plugins.push(descriptor);
        }
        Ok(())
    }

    /// Reject a descriptor whose commands or providers collide with another plugin's.
    fn check_no_collision(&self, descriptor: &PluginDescriptor) -> Result<(), RegistryError> {
        for existing in &self.plugins {
            // Re-registering the same plugin overwrites; not a self-collision.
            if existing.name == descriptor.name {
                continue;
            }

            for new_command in &descriptor.commands {
                for token in new_command.tokens() {
                    if existing
                        .commands
                        .iter()
                        .any(|old_command| old_command.matches(token))
                    {
                        return Err(RegistryError::DuplicateCommand {
                            token: token.to_string(),
                            owned_by: existing.name.clone(),
                        });
                    }
                }
            }

            for new_provider in &descriptor.context_providers {
                if existing
                    .context_providers
                    .iter()
                    .any(|old| old.name == new_provider.name)
                {
                    return Err(RegistryError::DuplicateContextProvider {
                        name: new_provider.name.clone(),
                        owned_by: existing.name.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The number of registered plugins.
    #[must_use]
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Whether nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// The descriptor for `name`.
    #[must_use]
    pub fn plugin(&self, name: &str) -> Option<&PluginDescriptor> {
        self.plugins.iter().find(|plugin| plugin.name == name)
    }

    /// Every descriptor, in registration order.
    #[must_use]
    pub fn plugins(&self) -> &[PluginDescriptor] {
        &self.plugins
    }

    /// Plugin names, in registration order.
    #[must_use]
    pub fn plugin_names(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name.as_str()).collect()
    }

    /// Find the plugin owning the command invoked by `token`.
    ///
    /// Matches primary names and aliases alike.
    #[must_use]
    pub fn find_command(&self, token: &str) -> Option<ResolvedCommand> {
        for plugin in &self.plugins {
            if let Some(command) = plugin.commands.iter().find(|c| c.matches(token)) {
                return Some(ResolvedCommand {
                    plugin: plugin.name.clone(),
                    command: command.clone(),
                });
            }
        }
        None
    }

    /// Render the `abi plugin list` body.
    ///
    /// The three-way format is a contract, byte-compared against
    /// `tests/golden/plugin-list.txt`:
    ///
    /// - version, target feature and entry point all present:
    ///   `  - <name> v<version> [<feature>] (<entry>): <description>`
    /// - version and target feature only:
    ///   `  - <name> v<version> [<feature>]: <description>`
    /// - otherwise: `  - <name>: <description>`
    #[must_use]
    pub fn format_plugin_list(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Installed Plugins ({}):", self.plugins.len());
        for plugin in &self.plugins {
            let has_version = !plugin.version.is_empty();
            let has_feature = !plugin.target_feature.is_empty();
            let has_entry = !plugin.entry_point.is_empty();
            let _ = if has_version && has_feature && has_entry {
                writeln!(
                    out,
                    "  - {} v{} [{}] ({}): {}",
                    plugin.name,
                    plugin.version,
                    plugin.target_feature,
                    plugin.entry_point,
                    plugin.description
                )
            } else if has_version && has_feature {
                writeln!(
                    out,
                    "  - {} v{} [{}]: {}",
                    plugin.name, plugin.version, plugin.target_feature, plugin.description
                )
            } else {
                writeln!(out, "  - {}: {}", plugin.name, plugin.description)
            };
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_with_command(plugin: &str, command: &str, aliases: &[&str]) -> PluginDescriptor {
        PluginDescriptor::new(plugin, "d").with_commands(vec![
            PluginCommand::new(command).with_aliases(aliases.to_vec()),
        ])
    }

    #[test]
    fn a_new_registry_is_empty() {
        let registry = Registry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.plugin_count(), 0);
    }

    #[test]
    fn register_stores_name_and_description() {
        let mut registry = Registry::new();
        registry
            .register("demo", "a demo plugin")
            .expect("register");
        let plugin = registry.plugin("demo").expect("present");
        assert_eq!(plugin.name, "demo");
        assert_eq!(plugin.description, "a demo plugin");
        assert_eq!(registry.plugin_count(), 1);
    }

    #[test]
    fn an_unknown_plugin_is_none() {
        assert!(Registry::new().plugin("nope").is_none());
    }

    #[test]
    fn registration_order_is_preserved() {
        // `abi plugin list` order depends on this; a HashMap would scramble it.
        let mut registry = Registry::new();
        for name in ["zeta", "alpha", "mid"] {
            registry.register(name, "d").expect("register");
        }
        assert_eq!(registry.plugin_names(), ["zeta", "alpha", "mid"]);
    }

    #[test]
    fn re_registering_a_name_replaces_in_place() {
        let mut registry = Registry::new();
        registry.register("a", "first").expect("register");
        registry.register("b", "other").expect("register");
        registry.register("a", "second").expect("re-register");

        assert_eq!(registry.plugin_count(), 2);
        assert_eq!(registry.plugin("a").expect("present").description, "second");
        // Position is kept, not moved to the end.
        assert_eq!(registry.plugin_names(), ["a", "b"]);
    }

    #[test]
    fn re_registering_a_plugin_does_not_collide_with_itself() {
        let mut registry = Registry::new();
        let descriptor = descriptor_with_command("p", "run", &["r"]);
        registry
            .register_plugin(descriptor.clone())
            .expect("first registration");
        registry
            .register_plugin(descriptor)
            .expect("re-registering the same plugin must be allowed");
    }

    #[test]
    fn two_plugins_cannot_claim_the_same_command_name() {
        let mut registry = Registry::new();
        registry
            .register_plugin(descriptor_with_command("first", "deploy", &[]))
            .expect("register");
        let err = registry
            .register_plugin(descriptor_with_command("second", "deploy", &[]))
            .expect_err("collision");
        assert_eq!(
            err,
            RegistryError::DuplicateCommand {
                token: "deploy".to_string(),
                owned_by: "first".to_string()
            }
        );
        assert_eq!(registry.plugin_count(), 1, "rejected plugin must not land");
    }

    #[test]
    fn a_new_alias_cannot_shadow_an_existing_command_name() {
        let mut registry = Registry::new();
        registry
            .register_plugin(descriptor_with_command("first", "deploy", &[]))
            .expect("register");
        let err = registry
            .register_plugin(descriptor_with_command("second", "ship", &["deploy"]))
            .expect_err("alias collides with an existing primary name");
        assert!(matches!(err, RegistryError::DuplicateCommand { .. }));
    }

    #[test]
    fn a_new_command_name_cannot_shadow_an_existing_alias() {
        let mut registry = Registry::new();
        registry
            .register_plugin(descriptor_with_command("first", "ship", &["deploy"]))
            .expect("register");
        let err = registry
            .register_plugin(descriptor_with_command("second", "deploy", &[]))
            .expect_err("primary name collides with an existing alias");
        assert!(matches!(err, RegistryError::DuplicateCommand { .. }));
    }

    #[test]
    fn two_plugins_cannot_share_an_alias() {
        let mut registry = Registry::new();
        registry
            .register_plugin(descriptor_with_command("first", "alpha", &["shared"]))
            .expect("register");
        let err = registry
            .register_plugin(descriptor_with_command("second", "beta", &["shared"]))
            .expect_err("alias collision");
        assert!(matches!(err, RegistryError::DuplicateCommand { .. }));
    }

    #[test]
    fn two_plugins_cannot_claim_the_same_context_provider() {
        let mut registry = Registry::new();
        registry
            .register_plugin(
                PluginDescriptor::new("first", "d")
                    .with_context_providers(vec![ContextProvider::new("git")]),
            )
            .expect("register");
        let err = registry
            .register_plugin(
                PluginDescriptor::new("second", "d")
                    .with_context_providers(vec![ContextProvider::new("git")]),
            )
            .expect_err("collision");
        assert_eq!(
            err,
            RegistryError::DuplicateContextProvider {
                name: "git".to_string(),
                owned_by: "first".to_string()
            }
        );
    }

    #[test]
    fn find_command_resolves_names_and_aliases() {
        let mut registry = Registry::new();
        registry
            .register_plugin(descriptor_with_command("owner", "deploy", &["d", "ship"]))
            .expect("register");

        for token in ["deploy", "d", "ship"] {
            let resolved = registry.find_command(token).expect("resolves");
            assert_eq!(resolved.plugin, "owner");
            assert_eq!(resolved.command.name, "deploy");
        }
        assert!(registry.find_command("unknown").is_none());
    }

    #[test]
    fn command_tokens_lists_name_then_aliases() {
        let command = PluginCommand::new("deploy").with_aliases(["d", "ship"]);
        assert_eq!(
            command.tokens().collect::<Vec<_>>(),
            ["deploy", "d", "ship"]
        );
    }

    #[test]
    fn plugin_list_uses_the_full_form_when_all_fields_are_present() {
        let mut registry = Registry::new();
        registry
            .register_plugin(
                PluginDescriptor::new("gpu-plugin", "Example reference plugin.")
                    .with_version("0.1.0")
                    .with_target_feature("gpu")
                    .with_entry_point("mod.rs"),
            )
            .expect("register");
        assert_eq!(
            registry.format_plugin_list(),
            "Installed Plugins (1):\n  - gpu-plugin v0.1.0 [gpu] (mod.rs): Example reference plugin.\n"
        );
    }

    #[test]
    fn plugin_list_drops_the_entry_point_when_absent() {
        let mut registry = Registry::new();
        registry
            .register_plugin(
                PluginDescriptor::new("p", "desc")
                    .with_version("1.2.3")
                    .with_target_feature("feat"),
            )
            .expect("register");
        assert_eq!(
            registry.format_plugin_list(),
            "Installed Plugins (1):\n  - p v1.2.3 [feat]: desc\n"
        );
    }

    #[test]
    fn plugin_list_falls_back_to_name_and_description() {
        let mut registry = Registry::new();
        registry
            .register("bare", "just a description")
            .expect("register");
        assert_eq!(
            registry.format_plugin_list(),
            "Installed Plugins (1):\n  - bare: just a description\n"
        );
    }

    #[test]
    fn plugin_list_of_an_empty_registry_is_just_the_header() {
        assert_eq!(
            Registry::new().format_plugin_list(),
            "Installed Plugins (0):\n"
        );
    }

    #[test]
    fn plugin_list_matches_the_golden_header_shape() {
        // tests/golden/plugin-list.txt starts with "Installed Plugins (16):"
        // and every body line uses the full three-part form.
        let mut registry = Registry::new();
        for (name, feature) in [("accelerator-plugin", "accelerator"), ("ai-plugin", "ai")] {
            registry
                .register_plugin(
                    PluginDescriptor::new(
                        name,
                        format!("Example reference plugin targeting the feat-{feature} gate."),
                    )
                    .with_version("0.1.0")
                    .with_target_feature(feature)
                    .with_entry_point("mod.rs"),
                )
                .expect("register");
        }
        let rendered = registry.format_plugin_list();
        assert!(rendered.starts_with("Installed Plugins (2):\n"));
        assert!(rendered.contains(
            "  - accelerator-plugin v0.1.0 [accelerator] (mod.rs): Example reference plugin targeting the feat-accelerator gate.\n"
        ));
    }
}
