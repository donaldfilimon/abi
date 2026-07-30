//! Model catalog — the single source of truth for recognized model ids, short
//! aliases, and provider routing.
//!
//! Ported from `src/features/ai/models.zig`.

/// The provider that ultimately serves a given model id.
///
/// The local persona router is the default; the remote providers map to the
/// matching connector under `abi-connectors` and are only reachable across its
/// explicit live-transport boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// The local persona router — the only provider currently reachable from
    /// this crate.
    Local,
    /// Anthropic's API.
    Anthropic,
    /// [`OpenAI`](https://openai.com)'s API.
    OpenAi,
    /// xAI's Grok API.
    Grok,
    /// Apple `FoundationModels`, on-device.
    Fm,
}

impl Provider {
    /// The provider's own label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Grok => "grok",
            Self::Fm => "fm",
        }
    }
}

/// Anthropic Fable 5 — the default model recorded when a caller does not
/// request one.
pub const FABLE5: &str = "claude-fable-5";

/// Default model label. Kept in sync with the MCP `ai_complete` default.
pub const DEFAULT_MODEL: &str = FABLE5;

/// One catalog entry: a canonical id, its provider, and its short aliases.
struct Entry {
    id: &'static str,
    provider: Provider,
    aliases: &'static [&'static str],
}

/// The recognized model catalog.
///
/// Freeform ids are still accepted by callers; this list drives recognition,
/// alias resolution, and provider routing.
const CATALOG: &[Entry] = &[
    Entry {
        id: "abi-local",
        provider: Provider::Local,
        aliases: &[],
    },
    Entry {
        id: FABLE5,
        provider: Provider::Anthropic,
        aliases: &["fable-5", "fable5"],
    },
    Entry {
        id: "claude-opus-4-8",
        provider: Provider::Anthropic,
        aliases: &[],
    },
    Entry {
        id: "claude-sonnet-4-6",
        provider: Provider::Anthropic,
        aliases: &[],
    },
    Entry {
        id: "claude-haiku-4-5",
        provider: Provider::Anthropic,
        aliases: &[],
    },
    Entry {
        id: "apple-fm",
        provider: Provider::Fm,
        aliases: &["fm-local", "fm"],
    },
];

/// Resolve a user-supplied id, including short aliases, to its canonical id.
///
/// `None` if `id` is not in the catalog under any name.
#[must_use]
pub fn resolve(id: &str) -> Option<&'static str> {
    CATALOG
        .iter()
        .find_map(|entry| (entry.id == id || entry.aliases.contains(&id)).then_some(entry.id))
}

/// Whether `id` names a catalog entry, by id or alias.
#[must_use]
pub fn is_known(id: &str) -> bool {
    resolve(id).is_some()
}

/// Canonicalize a user-supplied id at the CLI/MCP edge.
///
/// A known alias or id resolves to its canonical catalog id; a freeform id
/// passes through unchanged. This is what makes `--model fable-5` and
/// `--model claude-fable-5` both record `claude-fable-5`.
#[must_use]
pub fn canonical(id: &str) -> &str {
    resolve(id).unwrap_or(id)
}

/// Classify a model id to its serving provider.
///
/// Catalog ids are authoritative; an unknown id falls back to prefix
/// heuristics so freeform ids still route sensibly. Local/offline providers
/// (Ollama, LM Studio, llama.cpp, vLLM, MLX) are recognized by prefix so they
/// route explicitly to [`Provider::Local`] rather than falling through
/// silently — the local persona router serves them deterministically without
/// cloud credentials.
#[must_use]
pub fn provider_of(id: &str) -> Provider {
    if let Some(canonical_id) = resolve(id)
        && let Some(entry) = CATALOG.iter().find(|entry| entry.id == canonical_id)
    {
        return entry.provider;
    }
    if id.starts_with("claude-") {
        return Provider::Anthropic;
    }
    if id.starts_with("gpt-") {
        return Provider::OpenAi;
    }
    if id.starts_with("grok") {
        return Provider::Grok;
    }
    if id.starts_with("apple-") {
        return Provider::Fm;
    }
    if id.starts_with("ollama-")
        || id.starts_with("ollama/")
        || id.starts_with("lmstudio/")
        || id.starts_with("llama-cpp/")
        || id.starts_with("llama/")
        || id.starts_with("vllm/")
        || id.starts_with("mlx-")
        || id.starts_with("mlx/")
    {
        return Provider::Local;
    }
    Provider::Local
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fable5_is_recognized_and_routes_to_anthropic() {
        assert!(is_known(FABLE5));
        assert_eq!(resolve("fable-5"), Some(FABLE5));
        assert_eq!(resolve("fable5"), Some(FABLE5));
        assert_eq!(provider_of(FABLE5), Provider::Anthropic);
    }

    #[test]
    fn canonical_resolves_aliases_and_passes_freeform_ids_through() {
        assert_eq!(canonical("fable-5"), FABLE5);
        assert_eq!(canonical(FABLE5), FABLE5);
        assert_eq!(canonical(DEFAULT_MODEL), DEFAULT_MODEL);
        assert_eq!(canonical("gpt-5-mystery"), "gpt-5-mystery");
    }

    #[test]
    fn fable5_alias_routes_identically_to_the_canonical_id() {
        assert_eq!(canonical(FABLE5), canonical("fable-5"));
        assert_eq!(provider_of(FABLE5), provider_of("fable-5"));
        assert_eq!(provider_of("fable-5"), Provider::Anthropic);
    }

    #[test]
    fn apple_fm_is_recognized_and_routes_to_the_on_device_provider() {
        assert!(is_known("apple-fm"));
        assert_eq!(resolve("fm"), Some("apple-fm"));
        assert_eq!(resolve("fm-local"), Some("apple-fm"));
        assert_eq!(canonical("fm"), "apple-fm");
        assert_eq!(provider_of("apple-fm"), Provider::Fm);
        assert_eq!(provider_of("apple-future-model"), Provider::Fm);
        assert_eq!(Provider::Fm.label(), "fm");
    }

    #[test]
    fn default_model_is_recognized_and_routes_to_anthropic() {
        assert!(is_known(DEFAULT_MODEL));
        assert_eq!(provider_of(DEFAULT_MODEL), Provider::Anthropic);
    }

    #[test]
    fn unknown_ids_are_not_known_but_prefixes_still_classify() {
        assert!(!is_known("mystery"));
        assert_eq!(resolve("mystery"), None);
        assert_eq!(provider_of("mystery"), Provider::Local);
        assert_eq!(provider_of("claude-future-9"), Provider::Anthropic);
        assert_eq!(provider_of("gpt-5"), Provider::OpenAi);
        assert_eq!(provider_of("grok-3"), Provider::Grok);
    }

    #[test]
    fn local_offline_provider_prefixes_route_to_local_deterministically() {
        for id in [
            "ollama-llama3",
            "ollama/qwen2",
            "lmstudio/qwen2",
            "llama-cpp/mistral",
            "llama/phi3",
            "vllm/mixtral",
            "mlx-qwen",
            "mlx/llama3",
        ] {
            assert_eq!(provider_of(id), Provider::Local, "{id}");
        }
        assert!(!is_known("ollama-llama3"));
        assert_eq!(resolve("ollama-llama3"), None);
        assert_eq!(canonical("ollama-llama3"), "ollama-llama3");
    }
}
