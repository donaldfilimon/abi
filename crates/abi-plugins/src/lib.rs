//! ABI framework plugins: the sixteen bundled plugins and the plugin manager.
//!
//! The Rust successor to `src/plugins/` in the Zig tree.
//!
//! ## Two surfaces, one plugin set
//!
//! `abi plugin run` and the MCP `plugin_run` tool dispatch through the same
//! [`PluginManager`] over the same [`BUNDLED`] table, so a plugin runnable from
//! one surface is runnable from the other. That symmetry was explicit in the Zig
//! comments on `bundled_plugin_paths` and is preserved here.
//!
//! The two *listings* legitimately differ, and this is contract, not drift:
//!
//! - MCP `plugin_list` emits [`BUNDLED`] order, which is declaration order and
//!   is **not** alphabetical (`telemetry-exporter` comes third).
//! - `abi plugin list` renders [`abi_core::registry::Registry`], which Zig
//!   populated from a generated file that walked the plugin directory — so it is
//!   alphabetical. [`registry_descriptors`] reproduces that.
//!
//! ## mod/stub parity
//!
//! Zig enforced "a feature and its no-op stub expose identical APIs" with
//! `tools/check_parity.zig`, which compared declaration lists across 14 plugin
//! pairs. Rust gets the same guarantee for free from the type system: both
//! `mod.rs` and `stub.rs` implement [`Plugin`], so a missing or misnamed item is
//! a compile error rather than a lint. [`assert_plugin_parity`] adds the half
//! types alone do not cover — that the metadata constants agree — as a
//! `const` assertion. The script is therefore dropped deliberately, not silently.
//!
//! ## Deliberate design notes
//!
//! Asserted by tests in `tests/golden_plugins.rs`:
//!
//! 1. **`entry_point` is `mod.rs`.** Golden fixtures (`tests/golden/plugin-list.txt`,
//!    the `plugin_list` line of `tests/golden/mcp-tool-calls.jsonl`) match the
//!    Rust entry point byte-for-byte.
//! 2. **[`PluginManager::load_bundled`] does not read the disk.** The Zig port
//!    resolved 16 repo-root-relative paths at runtime, so running the binary from
//!    another directory silently produced `count=0`. The bundled set is compiled
//!    in here, which makes the contract output independent of cwd.
//!    [`PluginManager::load_plugin`] still reads a manifest from disk for
//!    external plugin directories, and a test parses all 16 on-disk
//!    `abi-plugin.json` files and asserts they match [`BUNDLED`] field by field,
//!    so the compiled-in copy cannot drift from the manifests.

pub mod manager;

pub use abi_foundation::plugin_manifest::{ManifestCommand, ManifestContextProvider};
pub use manager::{PluginInfo, PluginLoadError, PluginManager};

/// A bundled plugin implementation.
///
/// Implemented twice per plugin: by `Mod` in `mod.rs` (the real behavior) and by
/// `Stub` in `stub.rs` (which always fails with [`PluginError::FeatureDisabled`]).
pub trait Plugin {
    /// Plugin name, matching `abi-plugin.json`.
    const NAME: &'static str;
    /// Semantic version, matching `abi-plugin.json`.
    const VERSION: &'static str;
    /// Human-readable description, matching `abi-plugin.json`.
    const DESCRIPTION: &'static str;
    /// The feature gate this plugin targets, matching `abi-plugin.json`.
    const TARGET_FEATURE: &'static str;

    /// Run the plugin against `input`.
    ///
    /// `input` carries an optional prefix protocol, unchanged from Zig:
    /// `__context__:<provider>` requests a context snippet, and
    /// `__cmd__:<command>[\n<payload>]` invokes a declared slash-command.
    /// Anything else is treated as opaque text.
    fn run(input: &str) -> Result<String, PluginError>;
}

/// Why a plugin's `run` failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginError {
    /// The plugin's feature gate is off; this is the `stub.rs` behavior, and
    /// corresponds to Zig's `error.FeatureDisabled`.
    FeatureDisabled,
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FeatureDisabled => f.write_str("plugin feature is disabled"),
        }
    }
}

impl std::error::Error for PluginError {}

/// Split a `__cmd__:` payload into its command name and remaining payload.
///
/// Mirrors Zig's `splitScalar(u8, rest, '\n')` followed by one `next()` and a
/// `rest()`: the command is everything up to the first newline, and the payload
/// is everything after it — empty when there is no newline at all.
#[must_use]
pub fn split_command(rest: &str) -> (&str, &str) {
    rest.split_once('\n').unwrap_or((rest, ""))
}

/// Compile-time string equality, for [`assert_plugin_parity`].
///
/// `==` on `&str` is not usable in a `const` block, so this compares bytes.
#[must_use]
pub const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Assert at compile time that a plugin and its stub declare identical metadata.
///
/// The replacement for `tools/check_parity.zig`. That the two types expose the
/// same *functions* is already guaranteed by both implementing [`Plugin`]; this
/// covers the constants, which a trait impl alone would let diverge.
#[macro_export]
macro_rules! assert_plugin_parity {
    ($real:ty, $stub:ty) => {
        const _: () = {
            assert!($crate::str_eq(
                <$real as $crate::Plugin>::NAME,
                <$stub as $crate::Plugin>::NAME,
            ));
            assert!($crate::str_eq(
                <$real as $crate::Plugin>::VERSION,
                <$stub as $crate::Plugin>::VERSION,
            ));
            assert!($crate::str_eq(
                <$real as $crate::Plugin>::DESCRIPTION,
                <$stub as $crate::Plugin>::DESCRIPTION,
            ));
            assert!($crate::str_eq(
                <$real as $crate::Plugin>::TARGET_FEATURE,
                <$stub as $crate::Plugin>::TARGET_FEATURE,
            ));
        };
    };
}

/// Declare a plugin's real and stub modules and assert their parity.
///
/// The plugin sources live in `crates/abi-plugins/plugins/<dir>/` rather than
/// under `src/`, so that each plugin remains a self-contained directory holding
/// `abi-plugin.json`, `mod.rs`, and `stub.rs` — the layout the manifest validator
/// requires, and the one the Zig tree used.
///
/// Both paths are spelled out at each call site because `#[path]` takes a string
/// literal and will not accept a `concat!`.
///
/// The two file modules are declared at the top level of `lib.rs`, not inside the
/// public `$module` block, so that `#[path]` resolves relative to `src/`. Nesting
/// them would make the path relative to `src/$module/` — a directory that does
/// not exist, and `..` cannot be resolved through a missing directory.
macro_rules! declare_plugin {
    ($module:ident, $real:ident, $stub:ident, $mod_path:literal, $stub_path:literal) => {
        #[path = $mod_path]
        mod $real;
        #[path = $stub_path]
        mod $stub;

        /// A bundled plugin: its real implementation and its disabled stub.
        pub mod $module {
            pub use super::$real::Mod;
            pub use super::$stub::Stub;
        }

        $crate::assert_plugin_parity!($module::Mod, $module::Stub);
    };
}

declare_plugin!(
    example_plugin,
    example_plugin_real,
    example_plugin_stub,
    "../plugins/example-plugin/mod.rs",
    "../plugins/example-plugin/stub.rs"
);
declare_plugin!(
    example_wdbx_plugin,
    example_wdbx_plugin_real,
    example_wdbx_plugin_stub,
    "../plugins/example-wdbx-plugin/mod.rs",
    "../plugins/example-wdbx-plugin/stub.rs"
);
declare_plugin!(
    telemetry_exporter,
    telemetry_exporter_real,
    telemetry_exporter_stub,
    "../plugins/telemetry-exporter/mod.rs",
    "../plugins/telemetry-exporter/stub.rs"
);
declare_plugin!(
    ai_plugin,
    ai_plugin_real,
    ai_plugin_stub,
    "../plugins/ai-plugin/mod.rs",
    "../plugins/ai-plugin/stub.rs"
);
declare_plugin!(
    gpu_plugin,
    gpu_plugin_real,
    gpu_plugin_stub,
    "../plugins/gpu-plugin/mod.rs",
    "../plugins/gpu-plugin/stub.rs"
);
declare_plugin!(
    accelerator_plugin,
    accelerator_plugin_real,
    accelerator_plugin_stub,
    "../plugins/accelerator-plugin/mod.rs",
    "../plugins/accelerator-plugin/stub.rs"
);
declare_plugin!(
    shader_plugin,
    shader_plugin_real,
    shader_plugin_stub,
    "../plugins/shader-plugin/mod.rs",
    "../plugins/shader-plugin/stub.rs"
);
declare_plugin!(
    mlir_plugin,
    mlir_plugin_real,
    mlir_plugin_stub,
    "../plugins/mlir-plugin/mod.rs",
    "../plugins/mlir-plugin/stub.rs"
);
declare_plugin!(
    os_control_plugin,
    os_control_plugin_real,
    os_control_plugin_stub,
    "../plugins/os-control-plugin/mod.rs",
    "../plugins/os-control-plugin/stub.rs"
);
declare_plugin!(
    hash_plugin,
    hash_plugin_real,
    hash_plugin_stub,
    "../plugins/hash-plugin/mod.rs",
    "../plugins/hash-plugin/stub.rs"
);
declare_plugin!(
    tui_plugin,
    tui_plugin_real,
    tui_plugin_stub,
    "../plugins/tui-plugin/mod.rs",
    "../plugins/tui-plugin/stub.rs"
);
declare_plugin!(
    nn_plugin,
    nn_plugin_real,
    nn_plugin_stub,
    "../plugins/nn-plugin/mod.rs",
    "../plugins/nn-plugin/stub.rs"
);
declare_plugin!(
    metrics_plugin,
    metrics_plugin_real,
    metrics_plugin_stub,
    "../plugins/metrics-plugin/mod.rs",
    "../plugins/metrics-plugin/stub.rs"
);
declare_plugin!(
    sea_plugin,
    sea_plugin_real,
    sea_plugin_stub,
    "../plugins/sea-plugin/mod.rs",
    "../plugins/sea-plugin/stub.rs"
);
declare_plugin!(
    mobile_plugin,
    mobile_plugin_real,
    mobile_plugin_stub,
    "../plugins/mobile-plugin/mod.rs",
    "../plugins/mobile-plugin/stub.rs"
);
declare_plugin!(
    foundationmodels_plugin,
    foundationmodels_plugin_real,
    foundationmodels_plugin_stub,
    "../plugins/foundationmodels-plugin/mod.rs",
    "../plugins/foundationmodels-plugin/stub.rs"
);

/// A slash-command declared by a bundled plugin, in compiled-in form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundledCommand {
    /// Command name.
    pub name: &'static str,
    /// One-line summary.
    pub summary: &'static str,
    /// Alternate names.
    pub aliases: &'static [&'static str],
}

/// A context provider declared by a bundled plugin, in compiled-in form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundledContextProvider {
    /// Provider name.
    pub name: &'static str,
    /// One-line summary.
    pub summary: &'static str,
}

/// One entry of the compiled-in bundled plugin set.
#[derive(Debug, Clone, Copy)]
pub struct BundledPlugin {
    /// Plugin name.
    pub name: &'static str,
    /// Semantic version.
    pub version: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// The feature gate this plugin targets.
    pub target_feature: &'static str,
    /// Entry-point file name, relative to `path`.
    pub entry_point: &'static str,
    /// Repo-relative directory holding this plugin's manifest and sources.
    pub path: &'static str,
    /// Declared slash-commands.
    pub commands: &'static [BundledCommand],
    /// Declared context providers.
    pub context_providers: &'static [BundledContextProvider],
    /// The real `run` implementation.
    pub run: fn(&str) -> Result<String, PluginError>,
}

/// Build a [`BundledPlugin`] from a plugin type, reusing its [`Plugin`] constants
/// so the table cannot disagree with the implementation.
macro_rules! bundled_entry {
    ($ty:ty, $dir:literal, $commands:expr, $context_providers:expr) => {
        BundledPlugin {
            name: <$ty as Plugin>::NAME,
            version: <$ty as Plugin>::VERSION,
            description: <$ty as Plugin>::DESCRIPTION,
            target_feature: <$ty as Plugin>::TARGET_FEATURE,
            entry_point: "mod.rs",
            path: concat!("crates/abi-plugins/plugins/", $dir),
            commands: $commands,
            context_providers: $context_providers,
            run: <$ty as Plugin>::run,
        }
    };
}

/// The sixteen bundled plugins, **in declaration order**.
///
/// This order is contract: MCP `plugin_list` emits it verbatim, and
/// `tests/golden/mcp-tool-calls.jsonl` pins it. It is deliberately not
/// alphabetical — it matches Zig's `bundled_plugin_paths`.
pub static BUNDLED: &[BundledPlugin] = &[
    bundled_entry!(
        example_plugin::Mod,
        "example-plugin",
        &[BundledCommand {
            name: "echo",
            summary: "Echo back the input text",
            aliases: &["say"],
        }],
        &[BundledContextProvider {
            name: "example-info",
            summary: "Example plugin metadata for REPL context",
        }]
    ),
    bundled_entry!(example_wdbx_plugin::Mod, "example-wdbx-plugin", &[], &[]),
    bundled_entry!(telemetry_exporter::Mod, "telemetry-exporter", &[], &[]),
    bundled_entry!(
        ai_plugin::Mod,
        "ai-plugin",
        &[
            BundledCommand {
                name: "profile",
                summary: "Show the current AI profile and router status",
                aliases: &["ai-status"],
            },
            BundledCommand {
                name: "models",
                summary: "List available model backends",
                aliases: &[],
            },
        ],
        &[]
    ),
    bundled_entry!(
        gpu_plugin::Mod,
        "gpu-plugin",
        &[
            BundledCommand {
                name: "gpu-status",
                summary: "Report GPU backend and acceleration mode",
                aliases: &["gpu"],
            },
            BundledCommand {
                name: "gpu-info",
                summary: "Show detailed GPU capabilities",
                aliases: &[],
            },
        ],
        &[]
    ),
    bundled_entry!(accelerator_plugin::Mod, "accelerator-plugin", &[], &[]),
    bundled_entry!(shader_plugin::Mod, "shader-plugin", &[], &[]),
    bundled_entry!(mlir_plugin::Mod, "mlir-plugin", &[], &[]),
    bundled_entry!(os_control_plugin::Mod, "os-control-plugin", &[], &[]),
    bundled_entry!(hash_plugin::Mod, "hash-plugin", &[], &[]),
    bundled_entry!(
        tui_plugin::Mod,
        "tui-plugin",
        &[
            BundledCommand {
                name: "hello",
                summary: "Greet from the tui-plugin",
                aliases: &["hi"],
            },
            BundledCommand {
                name: "ping",
                summary: "Respond with pong and timing info",
                aliases: &[],
            },
        ],
        &[]
    ),
    bundled_entry!(
        nn_plugin::Mod,
        "nn-plugin",
        &[
            BundledCommand {
                name: "nn-train",
                summary: "Train the miniature character-level neural-net demo",
                aliases: &[],
            },
            BundledCommand {
                name: "nn-sample",
                summary: "Generate text from the trained neural-net model",
                aliases: &[],
            },
        ],
        &[]
    ),
    bundled_entry!(metrics_plugin::Mod, "metrics-plugin", &[], &[]),
    bundled_entry!(sea_plugin::Mod, "sea-plugin", &[], &[]),
    bundled_entry!(mobile_plugin::Mod, "mobile-plugin", &[], &[]),
    bundled_entry!(
        foundationmodels_plugin::Mod,
        "foundationmodels-plugin",
        &[],
        &[]
    ),
];

/// Look up a bundled plugin by name.
#[must_use]
pub fn bundled(name: &str) -> Option<&'static BundledPlugin> {
    BUNDLED.iter().find(|plugin| plugin.name == name)
}

/// The bundled plugins as [`abi_core::registry`] descriptors, **alphabetically by
/// name**.
///
/// This is what `abi plugin list` renders. Zig built the same list from a
/// generated `src/plugin_registry.zig` that walked the plugin directory, which is
/// why the ordering differs from [`BUNDLED`].
#[must_use]
pub fn registry_descriptors() -> Vec<abi_core::registry::PluginDescriptor> {
    use abi_core::registry::{ContextProvider, PluginCommand, PluginDescriptor};

    let mut sorted: Vec<&BundledPlugin> = BUNDLED.iter().collect();
    sorted.sort_unstable_by_key(|plugin| plugin.name);

    sorted
        .into_iter()
        .map(|plugin| {
            let commands = plugin
                .commands
                .iter()
                .map(|command| {
                    PluginCommand::new(command.name)
                        .with_summary(command.summary)
                        .with_aliases(command.aliases.iter().copied())
                })
                .collect();
            let providers = plugin
                .context_providers
                .iter()
                .map(|provider| ContextProvider::new(provider.name))
                .collect();

            PluginDescriptor::new(plugin.name, plugin.description)
                .with_version(plugin.version)
                .with_target_feature(plugin.target_feature)
                .with_entry_point(plugin.entry_point)
                .with_commands(commands)
                .with_context_providers(providers)
        })
        .collect()
}

/// Register every bundled plugin into `registry`.
///
/// The replacement for Zig's generated `plugin_registry.registerPlugins`.
pub fn register_all(
    registry: &mut abi_core::registry::Registry,
) -> Result<(), abi_core::registry::RegistryError> {
    for descriptor in registry_descriptors() {
        registry.register_plugin(descriptor)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_holds_sixteen_plugins() {
        assert_eq!(BUNDLED.len(), 16, "the plugin count is frozen contract");
    }

    #[test]
    fn bundled_order_is_declaration_order_not_alphabetical() {
        // The first three, verbatim from Zig's `bundled_plugin_paths`. If this
        // ever sorts itself, MCP `plugin_list` stops matching its fixture.
        let names: Vec<&str> = BUNDLED.iter().map(|plugin| plugin.name).collect();
        assert_eq!(names[0], "example-plugin");
        assert_eq!(names[1], "example-wdbx-plugin");
        assert_eq!(names[2], "telemetry-exporter");

        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_ne!(names, sorted, "declaration order must not be alphabetical");
    }

    #[test]
    fn bundled_names_are_unique() {
        let mut names: Vec<&str> = BUNDLED.iter().map(|plugin| plugin.name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn registry_descriptors_are_alphabetical() {
        let names: Vec<&str> = registry_descriptors()
            .iter()
            .map(|descriptor| descriptor.name.clone())
            .map(|name| Box::leak(name.into_boxed_str()) as &str)
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert_eq!(names.len(), 16);
    }

    #[test]
    fn register_all_populates_a_registry() {
        let mut registry = abi_core::registry::Registry::new();
        register_all(&mut registry).expect("no command collisions among bundled plugins");
        assert_eq!(registry.plugin_count(), 16);
    }

    #[test]
    fn split_command_matches_zig_iterator_semantics() {
        assert_eq!(split_command("echo\nhello"), ("echo", "hello"));
        // No newline: the whole string is the command and the payload is empty,
        // which is what `iter.rest()` returns after a non-matching split.
        assert_eq!(split_command("ping"), ("ping", ""));
        // Trailing newline yields an empty payload, not a missing one.
        assert_eq!(split_command("echo\n"), ("echo", ""));
        // Only the first newline splits; the rest stays in the payload.
        assert_eq!(split_command("echo\na\nb"), ("echo", "a\nb"));
        assert_eq!(split_command(""), ("", ""));
    }

    #[test]
    fn stubs_report_feature_disabled() {
        assert_eq!(
            example_plugin::Stub::run("anything"),
            Err(PluginError::FeatureDisabled)
        );
        assert_eq!(
            tui_plugin::Stub::run("__cmd__:ping"),
            Err(PluginError::FeatureDisabled)
        );
    }

    #[test]
    fn example_plugin_handles_the_prefix_protocol() {
        // These strings are contract; `mcp-tool-calls-args.jsonl` pins the last one.
        assert_eq!(
            example_plugin::Mod::run("__cmd__:echo\nhello").unwrap(),
            "example-plugin echo: hello"
        );
        assert_eq!(
            example_plugin::Mod::run("__cmd__:say\nhello").unwrap(),
            "example-plugin echo: hello"
        );
        assert_eq!(
            example_plugin::Mod::run("__cmd__:echo").unwrap(),
            "example-plugin echo: (empty)"
        );
        assert_eq!(
            example_plugin::Mod::run("__cmd__:nope").unwrap(),
            "example-plugin: unknown command 'nope'"
        );
        assert_eq!(
            example_plugin::Mod::run("__context__:example-info").unwrap(),
            "example-plugin v0.1.0 (target: plugins)"
        );
        assert_eq!(
            example_plugin::Mod::run("__context__:mystery").unwrap(),
            "unknown context provider: mystery"
        );
        assert_eq!(
            example_plugin::Mod::run("").unwrap(),
            "example-plugin received input (len=0)"
        );
    }

    #[test]
    fn tui_plugin_aliases_resolve() {
        assert_eq!(
            tui_plugin::Mod::run("__cmd__:hi").unwrap(),
            "tui-plugin: hello"
        );
        assert_eq!(
            tui_plugin::Mod::run("__cmd__:ping").unwrap(),
            "tui-plugin: pong"
        );
    }

    #[test]
    fn plugin_run_counts_bytes_not_characters() {
        // Zig's `input.len` is a byte count; a multi-byte character must not be
        // counted as one.
        let output = hash_plugin::Mod::run("é").unwrap();
        assert_eq!(output, "hash-plugin event (bytes=2)");
    }

    #[test]
    fn bundled_run_pointers_reach_the_real_implementations() {
        let plugin = bundled("tui-plugin").expect("tui-plugin is bundled");
        assert_eq!((plugin.run)("__cmd__:ping").unwrap(), "tui-plugin: pong");
        assert!(bundled("no-such-plugin").is_none());
    }

    #[test]
    fn str_eq_compares_content() {
        assert!(str_eq("abc", "abc"));
        assert!(!str_eq("abc", "abd"));
        assert!(!str_eq("abc", "ab"));
        assert!(str_eq("", ""));
    }
}
