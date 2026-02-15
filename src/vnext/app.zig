//! vNext application handle.
//!
//! Wraps the existing Framework in a forward API that supports staged
//! compatibility while vNext surfaces are introduced.

const std = @import("std");
const core_config = @import("../core/config/mod.zig");
const framework_mod = @import("../core/framework.zig");
const capability_mod = @import("capability.zig");
const config_mod = @import("config.zig");

pub const App = struct {
    framework: framework_mod.Framework,

    pub const Error = framework_mod.Framework.Error || error{
        CapabilityUnavailable,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config_mod.AppConfig) Error!App {
        var framework = try framework_mod.Framework.init(allocator, cfg.framework);
        errdefer framework.deinit();

        if (cfg.strict_capability_check) {
            for (cfg.required_capabilities) |capability| {
                const feature = capability_mod.toFeature(core_config.Feature, capability);
                if (!framework.isEnabled(feature)) {
                    return error.CapabilityUnavailable;
                }
            }
        }

        return .{
            .framework = framework,
        };
    }

    pub fn initDefault(allocator: std.mem.Allocator) Error!App {
        return init(allocator, config_mod.AppConfig.defaults());
    }

    pub fn deinit(self: *App) void {
        self.framework.deinit();
    }

    pub fn getFramework(self: *App) *framework_mod.Framework {
        return &self.framework;
    }

    pub fn getFrameworkConst(self: *const App) *const framework_mod.Framework {
        return &self.framework;
    }
};
