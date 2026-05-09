const std = @import("std");
const types = @import("../../types.zig");
const expr = @import("../../expr.zig");
const stmt = @import("../../stmt.zig");
const kernel = @import("../../kernel.zig");
const backend = @import("../../codegen/backend.zig");
const common = @import("../../codegen/common.zig");
const gpu_backend = @import("../../../backend.zig");
const configs = @import("../configs/mod.zig");

const header = @import("header.zig");
const helpers = @import("helpers.zig");
const shared = @import("shared.zig");
const signature = @import("signature.zig");
const types_gen = @import("types_gen.zig");
const expressions = @import("expressions.zig");
const statements = @import("statements.zig");
const body = @import("body.zig");

pub fn CodeGenerator(comptime Config: type) type {
    return struct {
        writer: common.CodeWriter,
        allocator: std.mem.Allocator,
        config: *const configs.BackendConfig,

        const Self = @This();

        pub const backend_config: configs.BackendConfig = Config.config;

        pub const unary_fn_table: configs.UnaryFunctions.LookupTable = backend_config.unary_fns.buildTable();
        pub const binary_fn_table: configs.BinaryFunctions.LookupTable = backend_config.binary_fns.buildTable();
        pub const type_name_table: configs.TypeNames.LookupTable = backend_config.type_names.buildTable();
        pub const vector_prefix_table: configs.VectorNaming.LookupTable = backend_config.vector_naming.buildTable();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .writer = common.CodeWriter.init(allocator),
                .allocator = allocator,
                .config = &backend_config,
            };
        }

        pub fn deinit(self: *Self) void {
            self.writer.deinit();
        }

        pub fn generate(
            self: *Self,
            ir: *const kernel.KernelIR,
        ) backend.CodegenError!backend.GeneratedSource {
            try header.writeHeader(self, ir);
            try shared.writeSharedMemory(self, ir);
            try helpers.writeHelpers(self);
            try signature.writeKernelSignature(self, ir);
            try signature.writeBuiltinVars(self, ir);
            try body.writeBody(self, ir);
            try body.writeKernelClose(self);

            const code = try self.writer.getCode();
            errdefer self.allocator.free(code);
            const entry_point = try self.allocator.dupe(u8, ir.entry_point);

            return .{
                .code = code,
                .entry_point = entry_point,
                .backend = self.getGpuBackend(),
                .language = self.getLanguage(),
            };
        }

        fn getGpuBackend(self: *const Self) gpu_backend.Backend {
            return switch (self.config.language) {
                .glsl => .vulkan,
                .wgsl => .webgpu,
                .msl => .metal,
                .cuda => .cuda,
                .spirv => .vulkan,
                .hlsl => .vulkan,
            };
        }

        fn getLanguage(self: *const Self) backend.GeneratedSource.Language {
            return switch (self.config.language) {
                .glsl => .glsl,
                .wgsl => .wgsl,
                .msl => .msl,
                .cuda => .cuda,
                .spirv => .spirv,
                .hlsl => .hlsl,
            };
        }

        pub fn writeType(self: *Self, ty: types.Type) backend.CodegenError!void {
            return types_gen.writeType(self, ty);
        }

        pub fn writeExpr(self: *Self, e: *const expr.Expr) backend.CodegenError!void {
            return expressions.writeExpr(self, e);
        }

        pub fn writeStmt(self: *Self, s: *const stmt.Stmt) backend.CodegenError!void {
            return statements.writeStmt(self, s);
        }
    };
}
