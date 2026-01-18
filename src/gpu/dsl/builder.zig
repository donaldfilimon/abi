//! GPU Kernel DSL Builder API
//!
//! Provides a fluent API for constructing kernel IR programmatically.
//! This is the primary interface for creating portable GPU kernels.

const std = @import("std");
const types = @import("types.zig");
const expr = @import("expr.zig");
const stmt = @import("stmt.zig");
const kernel = @import("kernel.zig");

/// Builder for constructing kernel IR.
pub const KernelBuilder = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    entry_point: []const u8,
    workgroup_size: [3]u32,

    buffers: std.ArrayListUnmanaged(kernel.BufferBinding),
    uniforms: std.ArrayListUnmanaged(kernel.UniformBinding),
    push_constants: std.ArrayListUnmanaged(kernel.PushConstant),
    shared_memory: std.ArrayListUnmanaged(kernel.SharedMemory),
    statements: std.ArrayListUnmanaged(*const stmt.Stmt),
    functions: std.ArrayListUnmanaged(kernel.HelperFunction),

    // ID generation for values
    next_value_id: u32,
    next_binding_id: u32,

    // Feature tracking
    required_features: kernel.FeatureFlags,

    const Self = @This();

    /// Initialize a new kernel builder.
    pub fn init(allocator: std.mem.Allocator, kernel_name: []const u8) Self {
        return .{
            .allocator = allocator,
            .name = kernel_name,
            .entry_point = "main",
            .workgroup_size = .{ 256, 1, 1 },
            .buffers = .empty,
            .uniforms = .empty,
            .push_constants = .empty,
            .shared_memory = .empty,
            .statements = .empty,
            .functions = .empty,
            .next_value_id = 0,
            .next_binding_id = 0,
            .required_features = .{},
        };
    }

    /// Deinitialize the builder (does not free built IR).
    pub fn deinit(self: *Self) void {
        self.buffers.deinit(self.allocator);
        self.uniforms.deinit(self.allocator);
        self.push_constants.deinit(self.allocator);
        self.shared_memory.deinit(self.allocator);
        self.statements.deinit(self.allocator);
        self.functions.deinit(self.allocator);
    }

    // =========================================================================
    // Configuration Methods
    // =========================================================================

    /// Set the workgroup size.
    pub fn setWorkgroupSize(self: *Self, x: u32, y: u32, z: u32) *Self {
        self.workgroup_size = .{ x, y, z };
        return self;
    }

    /// Set the entry point name (default is "main").
    pub fn setEntryPoint(self: *Self, entry_point_name: []const u8) *Self {
        self.entry_point = entry_point_name;
        return self;
    }

    // =========================================================================
    // Binding Methods
    // =========================================================================

    /// Add a storage buffer binding.
    pub fn addBuffer(
        self: *Self,
        buf_name: []const u8,
        element_type: types.Type,
        access: types.AccessMode,
    ) !Value {
        const binding_id = self.next_binding_id;
        self.next_binding_id += 1;

        try self.buffers.append(self.allocator, .{
            .name = buf_name,
            .binding = binding_id,
            .group = 0,
            .element_type = element_type,
            .access = access,
        });

        return self.newBufferValue(buf_name, element_type);
    }

    /// Add a storage buffer binding with explicit binding index.
    pub fn addBufferAt(
        self: *Self,
        buf_name: []const u8,
        binding: u32,
        group: u32,
        element_type: types.Type,
        access: types.AccessMode,
    ) !Value {
        try self.buffers.append(self.allocator, .{
            .name = buf_name,
            .binding = binding,
            .group = group,
            .element_type = element_type,
            .access = access,
        });

        return self.newBufferValue(buf_name, element_type);
    }

    /// Add a uniform binding.
    pub fn addUniform(
        self: *Self,
        uni_name: []const u8,
        ty: types.Type,
    ) !Value {
        const binding_id = self.next_binding_id;
        self.next_binding_id += 1;

        try self.uniforms.append(self.allocator, .{
            .name = uni_name,
            .binding = binding_id,
            .group = 0,
            .ty = ty,
        });

        return self.newValue(ty, uni_name);
    }

    /// Add a uniform binding with explicit binding index.
    pub fn addUniformAt(
        self: *Self,
        uni_name: []const u8,
        binding: u32,
        group: u32,
        ty: types.Type,
    ) !Value {
        try self.uniforms.append(self.allocator, .{
            .name = uni_name,
            .binding = binding,
            .group = group,
            .ty = ty,
        });

        return self.newValue(ty, uni_name);
    }

    /// Add shared/workgroup memory.
    pub fn addSharedMemory(
        self: *Self,
        var_name: []const u8,
        element_type: types.Type,
        size: ?usize,
    ) !Value {
        try self.shared_memory.append(self.allocator, .{
            .name = var_name,
            .element_type = element_type,
            .size = size,
        });

        if (size == null) {
            self.required_features.dynamic_shared_memory = true;
        }

        const ptr_type = try self.allocType(.{
            .ptr = .{
                .pointee = try self.allocType(element_type),
                .address_space = .workgroup,
            },
        });
        _ = ptr_type;

        return self.newValue(element_type, var_name);
    }

    // =========================================================================
    // Built-in Variables
    // =========================================================================

    /// Get the global invocation ID (vec3<u32>).
    pub fn globalInvocationId(self: *Self) Value {
        return self.builtinValue(.global_invocation_id);
    }

    /// Get the local invocation ID (vec3<u32>).
    pub fn localInvocationId(self: *Self) Value {
        return self.builtinValue(.local_invocation_id);
    }

    /// Get the workgroup ID (vec3<u32>).
    pub fn workgroupId(self: *Self) Value {
        return self.builtinValue(.workgroup_id);
    }

    /// Get the local invocation index (u32).
    pub fn localInvocationIndex(self: *Self) Value {
        return self.builtinValue(.local_invocation_index);
    }

    /// Get the number of workgroups (vec3<u32>).
    pub fn numWorkgroups(self: *Self) Value {
        return self.builtinValue(.num_workgroups);
    }

    // =========================================================================
    // Expression Building
    // =========================================================================

    /// Create a literal expression.
    pub fn literal(self: *Self, comptime T: type, value: T) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .literal = switch (@typeInfo(T)) {
                .bool => .{ .bool_ = value },
                .int => |info| blk: {
                    if (info.bits <= 32) {
                        if (info.signedness == .signed) {
                            break :blk .{ .i32_ = @intCast(value) };
                        } else {
                            break :blk .{ .u32_ = @intCast(value) };
                        }
                    } else {
                        if (info.signedness == .signed) {
                            break :blk .{ .i64_ = @intCast(value) };
                        } else {
                            break :blk .{ .u64_ = @intCast(value) };
                        }
                    }
                },
                .float => |info| blk: {
                    if (info.bits <= 32) {
                        break :blk .{ .f32_ = @floatCast(value) };
                    } else {
                        break :blk .{ .f64_ = @floatCast(value) };
                    }
                },
                .comptime_int => .{ .i32_ = @intCast(value) },
                .comptime_float => .{ .f32_ = @floatCast(value) },
                else => @compileError("Unsupported literal type: " ++ @typeName(T)),
            },
        };
        return e;
    }

    /// Create a float literal.
    pub fn f32Lit(self: *Self, value: f32) !*const expr.Expr {
        return self.literal(f32, value);
    }

    /// Create an integer literal.
    pub fn i32Lit(self: *Self, value: i32) !*const expr.Expr {
        return self.literal(i32, value);
    }

    /// Create an unsigned integer literal.
    pub fn u32Lit(self: *Self, value: u32) !*const expr.Expr {
        return self.literal(u32, value);
    }

    /// Create a boolean literal.
    pub fn boolLit(self: *Self, value: bool) !*const expr.Expr {
        return self.literal(bool, value);
    }

    /// Binary operation.
    pub fn binOp(
        self: *Self,
        op: expr.BinaryOp,
        left: *const expr.Expr,
        right: *const expr.Expr,
    ) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .binary = .{
                .op = op,
                .left = left,
                .right = right,
            },
        };
        return e;
    }

    /// Unary operation.
    pub fn unaryOp(
        self: *Self,
        op: expr.UnaryOp,
        operand: *const expr.Expr,
    ) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .unary = .{
                .op = op,
                .operand = operand,
            },
        };
        return e;
    }

    // Convenience binary operations
    pub fn add(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.add, a, b);
    }

    pub fn sub(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.sub, a, b);
    }

    pub fn mul(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.mul, a, b);
    }

    pub fn div(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.div, a, b);
    }

    pub fn mod(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.mod, a, b);
    }

    pub fn lt(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.lt, a, b);
    }

    pub fn le(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.le, a, b);
    }

    pub fn gt(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.gt, a, b);
    }

    pub fn ge(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.ge, a, b);
    }

    /// Alias for ge (greater than or equal).
    pub fn gte(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.ge, a, b);
    }

    pub fn eq(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.eq, a, b);
    }

    pub fn ne(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.ne, a, b);
    }

    pub fn logicalAnd(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.and_, a, b);
    }

    pub fn logicalOr(self: *Self, a: *const expr.Expr, b: *const expr.Expr) !*const expr.Expr {
        return self.binOp(.or_, a, b);
    }

    // Convenience unary operations
    pub fn neg(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.neg, a);
    }

    pub fn logicalNot(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.not, a);
    }

    pub fn sqrt(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.sqrt, a);
    }

    pub fn abs(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.abs, a);
    }

    // Neural network activation functions
    pub fn exp(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.exp, a);
    }

    pub fn tanh(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.tanh, a);
    }

    pub fn sin(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.sin, a);
    }

    pub fn cos(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.cos, a);
    }

    pub fn log(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.log, a);
    }

    pub fn floor(self: *Self, a: *const expr.Expr) !*const expr.Expr {
        return self.unaryOp(.floor, a);
    }

    /// Array/buffer index access: base[index]
    pub fn index(self: *Self, base: *const expr.Expr, idx: *const expr.Expr) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .index = .{
                .base = base,
                .index = idx,
            },
        };
        return e;
    }

    /// Vector component access: v.x, v.y, v.z, v.w
    pub fn component(self: *Self, base: *const expr.Expr, field: []const u8) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .field = .{
                .base = base,
                .field = field,
            },
        };
        return e;
    }

    /// Type cast expression.
    pub fn cast(self: *Self, target_type: types.Type, operand: *const expr.Expr) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .cast = .{
                .target_type = target_type,
                .operand = operand,
            },
        };
        return e;
    }

    /// Cast to f32 type (convenience method).
    pub fn castToF32(self: *Self, operand: *const expr.Expr) !*const expr.Expr {
        return self.cast(types.Type.f32Type(), operand);
    }

    /// Cast to i32 type (convenience method).
    pub fn castToI32(self: *Self, operand: *const expr.Expr) !*const expr.Expr {
        return self.cast(types.Type.i32Type(), operand);
    }

    /// Cast to u32 type (convenience method).
    pub fn castToU32(self: *Self, operand: *const expr.Expr) !*const expr.Expr {
        return self.cast(types.Type.u32Type(), operand);
    }

    /// Ternary select expression (condition ? true_val : false_val).
    pub fn select(
        self: *Self,
        condition: *const expr.Expr,
        true_value: *const expr.Expr,
        false_value: *const expr.Expr,
    ) !*const expr.Expr {
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .select = .{
                .condition = condition,
                .true_value = true_value,
                .false_value = false_value,
            },
        };
        return e;
    }

    /// Built-in function call.
    pub fn call(
        self: *Self,
        function: expr.BuiltinFn,
        args: []const *const expr.Expr,
    ) !*const expr.Expr {
        const args_copy = try self.allocator.dupe(*const expr.Expr, args);
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .call = .{
                .function = function,
                .args = args_copy,
            },
        };
        return e;
    }

    /// Vector construction.
    pub fn vec(
        self: *Self,
        element_type: types.ScalarType,
        components: []const *const expr.Expr,
    ) !*const expr.Expr {
        std.debug.assert(components.len >= 2 and components.len <= 4);
        const components_copy = try self.allocator.dupe(*const expr.Expr, components);
        const e = try self.allocator.create(expr.Expr);
        e.* = .{
            .vector_construct = .{
                .element_type = element_type,
                .size = @intCast(components.len),
                .components = components_copy,
            },
        };
        return e;
    }

    // =========================================================================
    // Statement Building
    // =========================================================================

    /// Add a statement to the kernel body.
    pub fn addStmt(self: *Self, s: *const stmt.Stmt) !void {
        try self.statements.append(self.allocator, s);
    }

    /// Declare a variable.
    pub fn declareVar(
        self: *Self,
        var_name: []const u8,
        ty: types.Type,
        init_expr: ?*const expr.Expr,
    ) !Value {
        const s = try stmt.varDecl(self.allocator, var_name, ty, init_expr);
        try self.statements.append(self.allocator, s);
        return self.newValue(ty, var_name);
    }

    /// Declare a const variable.
    pub fn declareConst(
        self: *Self,
        const_name: []const u8,
        ty: types.Type,
        init_expr: *const expr.Expr,
    ) !Value {
        const s = try stmt.constDecl(self.allocator, const_name, ty, init_expr);
        try self.statements.append(self.allocator, s);
        return self.newValue(ty, const_name);
    }

    /// Assignment statement.
    pub fn assign(
        self: *Self,
        target: *const expr.Expr,
        value: *const expr.Expr,
    ) !void {
        const s = try stmt.assign(self.allocator, target, value);
        try self.statements.append(self.allocator, s);
    }

    /// If statement.
    pub fn ifStmt(
        self: *Self,
        condition: *const expr.Expr,
        then_body: []const *const stmt.Stmt,
        else_body: ?[]const *const stmt.Stmt,
    ) !void {
        const then_copy = try self.allocator.dupe(*const stmt.Stmt, then_body);
        const else_copy = if (else_body) |eb|
            try self.allocator.dupe(*const stmt.Stmt, eb)
        else
            null;

        const s = try stmt.ifStmt(self.allocator, condition, then_copy, else_copy);
        try self.statements.append(self.allocator, s);
    }

    /// Create an assignment statement (helper for building if bodies).
    pub fn assignStmt(
        self: *Self,
        target: *const expr.Expr,
        value: *const expr.Expr,
    ) !*const stmt.Stmt {
        return stmt.assign(self.allocator, target, value);
    }

    /// For loop.
    pub fn forLoop(
        self: *Self,
        init_s: ?*const stmt.Stmt,
        condition: ?*const expr.Expr,
        update: ?*const stmt.Stmt,
        body: []const *const stmt.Stmt,
    ) !void {
        const body_copy = try self.allocator.dupe(*const stmt.Stmt, body);
        const s = try stmt.forLoop(self.allocator, init_s, condition, update, body_copy);
        try self.statements.append(self.allocator, s);
    }

    /// While loop.
    pub fn whileLoop(
        self: *Self,
        condition: *const expr.Expr,
        body: []const *const stmt.Stmt,
    ) !void {
        const body_copy = try self.allocator.dupe(*const stmt.Stmt, body);
        const s = try stmt.whileLoop(self.allocator, condition, body_copy);
        try self.statements.append(self.allocator, s);
    }

    /// Workgroup barrier.
    pub fn barrier(self: *Self) !void {
        const call_expr = try self.call(.barrier, &.{});
        const s = try stmt.exprStmt(self.allocator, call_expr);
        try self.statements.append(self.allocator, s);
    }

    /// Memory barrier.
    pub fn memoryBarrier(self: *Self) !void {
        const call_expr = try self.call(.memory_barrier, &.{});
        const s = try stmt.exprStmt(self.allocator, call_expr);
        try self.statements.append(self.allocator, s);
    }

    // =========================================================================
    // Build Final IR
    // =========================================================================

    /// Build the final kernel IR.
    pub fn build(self: *Self) !kernel.KernelIR {
        return .{
            .name = self.name,
            .entry_point = self.entry_point,
            .workgroup_size = self.workgroup_size,
            .buffers = try self.buffers.toOwnedSlice(self.allocator),
            .uniforms = try self.uniforms.toOwnedSlice(self.allocator),
            .push_constants = try self.push_constants.toOwnedSlice(self.allocator),
            .shared_memory = try self.shared_memory.toOwnedSlice(self.allocator),
            .body = try self.statements.toOwnedSlice(self.allocator),
            .functions = try self.functions.toOwnedSlice(self.allocator),
            .required_features = self.required_features,
        };
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    fn newValue(self: *Self, ty: types.Type, var_name: ?[]const u8) Value {
        const id = self.next_value_id;
        self.next_value_id += 1;
        return .{
            .id = id,
            .ty = ty,
            .var_name = var_name,
            .builder = self,
        };
    }

    fn newBufferValue(self: *Self, buf_name: []const u8, element_type: types.Type) Value {
        const id = self.next_value_id;
        self.next_value_id += 1;
        return .{
            .id = id,
            .ty = element_type,
            .var_name = buf_name,
            .is_buffer = true,
            .builder = self,
        };
    }

    fn builtinValue(self: *Self, builtin: expr.BuiltinVar) Value {
        const id = self.next_value_id;
        self.next_value_id += 1;
        return .{
            .id = id,
            .ty = builtin.getType(),
            .var_name = builtin.name(),
            .builtin = builtin,
            .builder = self,
        };
    }

    fn allocType(self: *Self, ty: types.Type) !*const types.Type {
        const ptr = try self.allocator.create(types.Type);
        ptr.* = ty;
        return ptr;
    }
};

/// Wrapper for tracked values in the builder.
pub const Value = struct {
    id: u32,
    ty: types.Type,
    var_name: ?[]const u8 = null,
    is_buffer: bool = false,
    builtin: ?expr.BuiltinVar = null,
    builder: *KernelBuilder,

    /// Convert this value to an expression.
    pub fn toExpr(self: Value) !*const expr.Expr {
        const e = try self.builder.allocator.create(expr.Expr);
        e.* = .{
            .ref = .{
                .id = self.id,
                .ty = self.ty,
                .name = self.var_name,
            },
        };
        return e;
    }

    /// Get the x component of a vector value.
    pub fn x(self: Value) !*const expr.Expr {
        return self.builder.component(try self.toExpr(), "x");
    }

    /// Get the y component of a vector value.
    pub fn y(self: Value) !*const expr.Expr {
        return self.builder.component(try self.toExpr(), "y");
    }

    /// Get the z component of a vector value.
    pub fn z(self: Value) !*const expr.Expr {
        return self.builder.component(try self.toExpr(), "z");
    }

    /// Get the w component of a vector value.
    pub fn w(self: Value) !*const expr.Expr {
        return self.builder.component(try self.toExpr(), "w");
    }

    /// Index into this buffer.
    pub fn at(self: Value, idx: *const expr.Expr) !*const expr.Expr {
        return self.builder.index(try self.toExpr(), idx);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "KernelBuilder basic usage" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = KernelBuilder.init(allocator, "test_kernel");
    defer builder.deinit();

    _ = builder.setWorkgroupSize(128, 1, 1);

    const ir = try builder.build();
    try std.testing.expectEqualStrings("test_kernel", ir.name);
    try std.testing.expectEqual(@as(u32, 128), ir.workgroup_size[0]);
}

test "KernelBuilder add buffer" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = KernelBuilder.init(allocator, "test");
    defer builder.deinit();

    _ = try builder.addBuffer("input", types.Type.f32Type(), .read_only);
    _ = try builder.addBuffer("output", types.Type.f32Type(), .write_only);

    const ir = try builder.build();
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len);
    try std.testing.expectEqualStrings("input", ir.buffers[0].name);
    try std.testing.expectEqualStrings("output", ir.buffers[1].name);
}

test "KernelBuilder expression building" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = KernelBuilder.init(allocator, "test");
    defer builder.deinit();

    const a = try builder.f32Lit(1.0);
    const b = try builder.f32Lit(2.0);
    const sum = try builder.add(a, b);

    try std.testing.expect(sum.* == .binary);
    try std.testing.expectEqual(expr.BinaryOp.add, sum.binary.op);
}

test "KernelBuilder vector_add kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var builder = KernelBuilder.init(allocator, "vector_add");
    defer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Add buffer bindings
    const a = try builder.addBuffer("a", types.Type.f32Type(), .read_only);
    const b = try builder.addBuffer("b", types.Type.f32Type(), .read_only);
    const c = try builder.addBuffer("c", types.Type.f32Type(), .write_only);
    const n = try builder.addUniform("n", types.Type.u32Type());

    // Get thread index
    const gid = builder.globalInvocationId();
    const idx = try gid.x();

    // Bounds check: if (idx < n)
    const condition = try builder.lt(idx, try n.toExpr());

    // c[idx] = a[idx] + b[idx]
    const a_val = try a.at(idx);
    const b_val = try b.at(idx);
    const sum = try builder.add(a_val, b_val);
    const c_idx = try c.at(idx);

    const assign_stmt = try builder.assignStmt(c_idx, sum);
    try builder.ifStmt(condition, &[_]*const stmt.Stmt{assign_stmt}, null);

    const ir = try builder.build();

    try std.testing.expectEqualStrings("vector_add", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len);
    try std.testing.expectEqual(@as(usize, 1), ir.uniforms.len);
    try std.testing.expectEqual(@as(usize, 1), ir.body.len);
}
