//! LSP (ZLS) Service module switcher.

const build_options = @import("build_options");

const impl = if (build_options.feat_lsp)
    @import("real.zig")
else
    @import("stub.zig");

pub const Config = impl.Config;
pub const Client = impl.Client;
pub const Response = impl.Response;
pub const types = impl.types;
pub const jsonrpc = impl.jsonrpc;
pub const resolveWorkspaceRoot = impl.resolveWorkspaceRoot;
pub const resolvePath = impl.resolvePath;
pub const pathToUri = impl.pathToUri;
pub const EnvConfig = impl.EnvConfig;
pub const loadConfigFromEnv = impl.loadConfigFromEnv;

pub fn isEnabled() bool {
    return build_options.feat_lsp;
}

pub const Context = struct {
    pub fn isEnabled() bool {
        return build_options.feat_lsp;
    }
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
