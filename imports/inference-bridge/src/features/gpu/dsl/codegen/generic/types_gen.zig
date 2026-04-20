const types = @import("../../types.zig");
const backend = @import("../../codegen/backend.zig");

pub fn writeType(self: anytype, ty: types.Type) backend.CodegenError!void {
    switch (ty) {
        .scalar => |s| try self.writer.write(@TypeOf(self.*).type_name_table[@intFromEnum(s)]),
        .vector => |v| try writeVectorType(self, v),
        .matrix => |m| try writeMatrixType(self, m),
        .array => |a| try writeArrayType(self, a),
        .ptr => |p| {
            try self.writeType(p.pointee.*);
            try self.writer.write("*");
        },
        .void_ => try self.writer.write(self.config.type_names.void_),
    }
}

fn writeVectorType(self: anytype, v: types.VectorType) !void {
    switch (self.config.vector_naming.style) {
        .prefix => {
            const prefix = @TypeOf(self.*).vector_prefix_table[@intFromEnum(v.element)];
            try self.writer.writeFmt("{s}{d}", .{ prefix, v.size });
        },
        .type_suffix => {
            const prefix = @TypeOf(self.*).vector_prefix_table[@intFromEnum(v.element)];
            try self.writer.writeFmt("{s}{d}", .{ prefix, v.size });
        },
        .generic => {
            // WGSL style: vec3<f32>
            try self.writer.writeFmt("vec{d}<", .{v.size});
            try self.writer.write(@TypeOf(self.*).type_name_table[@intFromEnum(v.element)]);
            try self.writer.write(">");
        },
    }
}

fn writeMatrixType(self: anytype, m: types.MatrixType) !void {
    switch (self.config.language) {
        .glsl => try self.writer.writeFmt("mat{d}x{d}", .{ m.cols, m.rows }),
        .wgsl => {
            try self.writer.writeFmt("mat{d}x{d}<", .{ m.cols, m.rows });
            try self.writer.write(@TypeOf(self.*).type_name_table[@intFromEnum(m.element)]);
            try self.writer.write(">");
        },
        .msl => try self.writer.writeFmt("float{d}x{d}", .{ m.cols, m.rows }),
        .cuda => {
            // CUDA doesn't have native matrix types
            const base = @TypeOf(self.*).type_name_table[@intFromEnum(m.element)];
            try self.writer.writeFmt("{s}[{d}][{d}]", .{ base, m.rows, m.cols });
        },
        else => try self.writer.writeFmt("mat{d}x{d}", .{ m.cols, m.rows }),
    }
}

fn writeArrayType(self: anytype, a: types.ArrayType) !void {
    switch (self.config.language) {
        .wgsl => {
            try self.writer.write("array<");
            try self.writeType(a.element.*);
            if (a.size) |size| {
                try self.writer.writeFmt(", {d}>", .{size});
            } else {
                try self.writer.write(">");
            }
        },
        else => {
            try self.writeType(a.element.*);
            if (a.size) |size| {
                try self.writer.writeFmt("[{d}]", .{size});
            } else {
                try self.writer.write("*");
            }
        },
    }
}
