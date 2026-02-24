pub const registry = @import("registry.zig");
pub const pool = @import("pool.zig");

// Canonical backend metadata/detection/listing surface.
pub const types = @import("../backend/types.zig");
pub const meta = @import("../backend/meta.zig");
pub const detect = @import("../backend/detect.zig");
pub const libs = @import("../backend/libs.zig");
pub const listing = @import("../backend/listing.zig");

pub const cuda = @import("cuda/mod.zig");
pub const vulkan = @import("vulkan.zig");
pub const vulkan_compute = @import("vulkan/compute.zig");
pub const vulkan_vtable = @import("vulkan/vtable.zig");
pub const metal = @import("metal.zig");
pub const webgpu = @import("webgpu.zig");
pub const opengl = @import("opengl.zig");
pub const opengles = @import("opengles.zig");
pub const gl = @import("gl/mod.zig");
pub const directml = @import("directml/mod.zig");
const std = @import("std");
