pub const registry = @import("registry");
pub const pool = @import("pool");

// Canonical backend metadata/detection/listing surface.
pub const types = @import("../backend/types");
pub const meta = @import("../backend/meta");
pub const detect = @import("../backend/detect");
pub const libs = @import("../backend/libs");
pub const listing = @import("../backend/listing");

pub const cuda = @import("cuda");
pub const vulkan = @import("vulkan");
pub const vulkan_compute = @import("vulkan/compute");
pub const vulkan_vtable = @import("vulkan/vtable");
pub const metal = @import("metal");
pub const webgpu = @import("webgpu");
pub const opengl = @import("opengl");
pub const opengles = @import("opengles");
pub const gl = @import("gl");
pub const directml = @import("directml");
const std = @import("std");
