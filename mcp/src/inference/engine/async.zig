const std = @import("std");
const scheduler_mod = @import("../scheduler.zig");
const types = @import("types.zig");

pub const AsyncCallback = *const fn (types.Result) void;

pub fn beginAsyncJob(self: anytype) !void {
    self.async_mu.lock();
    defer self.async_mu.unlock();
    if (self.closing_async) return error.ShutdownInProgress;
    self.in_flight_async += 1;
}

pub fn finishAsyncJob(self: anytype) void {
    self.async_mu.lock();
    defer self.async_mu.unlock();
    std.debug.assert(self.in_flight_async > 0);
    self.in_flight_async -= 1;
}

pub fn generateAsync(
    self: anytype,
    request: scheduler_mod.Request,
    callback: AsyncCallback,
) !void {
    const EnginePtr = @TypeOf(self);

    try beginAsyncJob(self);
    errdefer finishAsyncJob(self);

    const prompt_copy = try self.allocator.dupe(u8, request.prompt);
    errdefer self.allocator.free(prompt_copy);

    const profile_copy = try self.allocator.dupe(u8, request.profile);
    errdefer self.allocator.free(profile_copy);

    const AsyncContext = struct {
        engine: EnginePtr,
        request: scheduler_mod.Request,
        callback: AsyncCallback,

        fn run(ctx: @This()) void {
            defer finishAsyncJob(ctx.engine);
            defer ctx.engine.allocator.free(ctx.request.profile);
            defer ctx.engine.allocator.free(ctx.request.prompt);

            var res = ctx.engine.generate(ctx.request) catch
                types.makeErrorResult(ctx.request.id, "Error: internal generation error");
            defer res.deinit(ctx.engine.allocator);
            ctx.callback(res);
        }
    };

    const ctx = AsyncContext{
        .engine = self,
        .request = .{
            .id = request.id,
            .prompt = prompt_copy,
            .max_tokens = request.max_tokens,
            .temperature = request.temperature,
            .top_p = request.top_p,
            .top_k = request.top_k,
            .profile = profile_copy,
            .priority = request.priority,
            .created_at = request.created_at,
            .stream = request.stream,
        },
        .callback = callback,
    };

    const thread = try std.Thread.spawn(.{}, AsyncContext.run, .{ctx});
    thread.detach();
}

pub fn generateAsyncWithTimeout(
    self: anytype,
    request: scheduler_mod.Request,
) !*types.AsyncResult {
    const EnginePtr = @TypeOf(self);

    try beginAsyncJob(self);
    errdefer finishAsyncJob(self);

    const prompt_copy = try self.allocator.dupe(u8, request.prompt);
    errdefer self.allocator.free(prompt_copy);

    const profile_copy = try self.allocator.dupe(u8, request.profile);
    errdefer self.allocator.free(profile_copy);

    const ar = try self.allocator.create(types.AsyncResult);
    errdefer self.allocator.destroy(ar);

    ar.* = .{
        .allocator = self.allocator,
        .state = std.atomic.Value(u8).init(@intFromEnum(types.AsyncResult.State.pending)),
        .result = undefined,
    };

    const TimeoutContext = struct {
        engine: EnginePtr,
        request: scheduler_mod.Request,
        ar: *types.AsyncResult,

        fn run(ctx: @This()) void {
            defer finishAsyncJob(ctx.engine);
            defer ctx.engine.allocator.free(ctx.request.profile);
            defer ctx.engine.allocator.free(ctx.request.prompt);

            const res = ctx.engine.generate(ctx.request) catch
                types.makeErrorResult(ctx.request.id, "Error: internal generation error");
            ctx.ar.result = res;

            const prev = ctx.ar.state.cmpxchgStrong(
                @intFromEnum(types.AsyncResult.State.pending),
                @intFromEnum(types.AsyncResult.State.ready),
                .acq_rel,
                .acquire,
            );
            if (prev != null) {
                if (ctx.ar.result.text_owned) ctx.engine.allocator.free(ctx.ar.result.text);
                if (ctx.ar.result.tokens_owned) ctx.engine.allocator.free(ctx.ar.result.tokens);
                ctx.engine.allocator.destroy(ctx.ar);
                return;
            }
        }
    };

    const ctx = TimeoutContext{
        .engine = self,
        .request = .{
            .id = request.id,
            .prompt = prompt_copy,
            .max_tokens = request.max_tokens,
            .temperature = request.temperature,
            .top_p = request.top_p,
            .top_k = request.top_k,
            .profile = profile_copy,
            .priority = request.priority,
            .created_at = request.created_at,
            .stream = request.stream,
        },
        .ar = ar,
    };

    const thread = try std.Thread.spawn(.{}, TimeoutContext.run, .{ctx});
    thread.detach();

    return ar;
}

pub fn isClosing(self: anytype) bool {
    self.async_mu.lock();
    defer self.async_mu.unlock();
    return self.closing_async;
}

test {
    std.testing.refAllDecls(@This());
}
