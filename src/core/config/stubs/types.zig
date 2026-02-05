const std = @import("std");

pub const ConfigError = error{
    ConfigDisabled,
    FeatureDisabled,
    InvalidConfig,
    MissingRequired,
    ConflictingConfig,
};

pub const LoadError = error{
    ConfigDisabled,
    InvalidValue,
    MissingRequired,
    ParseError,
    OutOfMemory,
};

pub const Feature = enum {
    gpu,
    ai,
    llm,
    embeddings,
    agents,
    training,
    database,
    network,
    observability,
    web,
    personas,
    cloud,

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
    }

    pub fn description(self: Feature) []const u8 {
        _ = self;
        return "Feature disabled";
    }

    pub fn isCompileTimeEnabled(self: Feature) bool {
        _ = self;
        return false;
    }
};
