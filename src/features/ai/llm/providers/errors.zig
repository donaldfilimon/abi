const std = @import("std");

pub const ProviderError = error{
    InvalidProvider,
    InvalidBackend,
    ModelRequired,
    PromptRequired,
    NotAvailable,
    NoProviderAvailable,
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
