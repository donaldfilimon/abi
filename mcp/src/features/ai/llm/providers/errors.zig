const std = @import("std");

pub const ProviderError = error{
    InvalidProvider,
    InvalidBackend,
    ModelRequired,
    PromptRequired,
    NotAvailable,
    NoProviderAvailable,
    /// Strict-mode failure: the pinned backend is unreachable and fallback
    /// is explicitly disabled via `--strict-backend`.
    StrictBackendUnavailable,
    PluginNotFound,
    PluginDisabled,
    InvalidPlugin,
    AbiVersionMismatch,
    SymbolMissing,
    GenerationFailed,
    MissingApiKey,
};

test {
    std.testing.refAllDecls(@This());
}
