const std = @import("std");
const utils = @import("../../../../foundation/mod.zig").utils;
const provider_router = @import("../../llm/providers/router.zig");
const types = @import("../types.zig");
const selection = @import("selection.zig");

pub const DispatchFn = *const fn (
    response_allocator: std.mem.Allocator,
    model: *types.ModelEntry,
    prompt: []const u8,
) types.OrchestrationError![]u8;

fn unixMs() i64 {
    return utils.unixMs();
}

pub fn execute(
    self: anytype,
    prompt: []const u8,
    task_type: ?types.TaskType,
    response_allocator: std.mem.Allocator,
) types.OrchestrationError![]u8 {
    const route_result = try self.route(prompt, task_type);

    if (self.config.enable_fallback) {
        return executeWithFallback(self, route_result, response_allocator);
    }

    return executeSingle(self, route_result.model_id, prompt, response_allocator);
}

pub fn executeEnsemble(
    self: anytype,
    prompt: []const u8,
    task_type: ?types.TaskType,
    response_allocator: std.mem.Allocator,
) types.OrchestrationError!types.EnsembleResult {
    if (!self.config.enable_ensemble) {
        return types.OrchestrationError.InvalidConfig;
    }

    const ens = self.ensemble_instance orelse return types.OrchestrationError.InvalidConfig;
    _ = ens;

    var available = std.ArrayListUnmanaged(*types.ModelEntry).empty;
    defer available.deinit(self.allocator);

    self.mutex.lock();
    var it = self.models.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.isAvailable()) {
            if (task_type) |tt| {
                if (selection.modelSupportsTask(self, entry.value_ptr, tt)) {
                    available.append(self.allocator, entry.value_ptr) catch
                        return types.OrchestrationError.OutOfMemory;
                }
            } else {
                available.append(self.allocator, entry.value_ptr) catch
                    return types.OrchestrationError.OutOfMemory;
            }
        }
    }
    self.mutex.unlock();

    if (available.items.len < self.config.min_ensemble_models) {
        return types.OrchestrationError.InsufficientModelsForEnsemble;
    }

    var responses = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (responses.items) |resp| {
            response_allocator.free(resp);
        }
        responses.deinit(self.allocator);
    }

    for (available.items) |model| {
        const resp = executeSingle(self, model.config.id, prompt, response_allocator) catch continue;
        responses.append(self.allocator, resp) catch {
            response_allocator.free(resp);
            continue;
        };
    }

    if (responses.items.len == 0) {
        return types.OrchestrationError.AllModelsFailed;
    }

    var combined = std.ArrayList(u8).empty;
    defer combined.deinit(response_allocator);

    for (responses.items, 0..) |resp, i| {
        if (i > 0) {
            combined.appendSlice(response_allocator, "\n\n--- Ensemble Response ---\n\n") catch
                return types.OrchestrationError.OutOfMemory;
        }
        combined.appendSlice(response_allocator, resp) catch
            return types.OrchestrationError.OutOfMemory;
    }

    const final_response = combined.toOwnedSlice(response_allocator) catch
        return types.OrchestrationError.OutOfMemory;

    return types.EnsembleResult{
        .response = final_response,
        .model_count = responses.items.len,
        .confidence = calculateEnsembleConfidence(responses.items.len, available.items.len),
    };
}

pub fn executeWithFallback(
    self: anytype,
    route_result: types.RouteResult,
    response_allocator: std.mem.Allocator,
) types.OrchestrationError![]u8 {
    return executeWithDispatch(self, route_result, response_allocator, providerDispatch);
}

pub fn executeWithDispatch(
    self: anytype,
    route_result: types.RouteResult,
    response_allocator: std.mem.Allocator,
    dispatch: DispatchFn,
) types.OrchestrationError![]u8 {
    if (executeSingleWithDispatch(self, route_result.model_id, route_result.prompt, response_allocator, dispatch)) |resp| {
        return resp;
    } else |_| {
        if (self.getModel(route_result.model_id)) |model| {
            model.consecutive_failures += 1;
            if (model.consecutive_failures >= self.config.max_retries) {
                model.status = .degraded;
            }
        }
    }

    self.mutex.lock();
    var fallbacks = std.ArrayListUnmanaged(*types.ModelEntry).empty;
    defer fallbacks.deinit(self.allocator);

    var it = self.models.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.isAvailable() and
            !std.mem.eql(u8, entry.value_ptr.config.id, route_result.model_id))
        {
            fallbacks.append(self.allocator, entry.value_ptr) catch continue;
        }
    }
    self.mutex.unlock();

    std.mem.sort(*types.ModelEntry, fallbacks.items, {}, struct {
        fn lessThan(_: void, a: *types.ModelEntry, b: *types.ModelEntry) bool {
            return a.config.priority < b.config.priority;
        }
    }.lessThan);

    for (fallbacks.items) |model| {
        if (executeSingleWithDispatch(self, model.config.id, route_result.prompt, response_allocator, dispatch)) |resp| {
            return resp;
        } else |_| {
            model.consecutive_failures += 1;
        }
    }

    return types.OrchestrationError.AllModelsFailed;
}

pub fn executeSingle(
    self: anytype,
    model_id: []const u8,
    prompt: []const u8,
    response_allocator: std.mem.Allocator,
) types.OrchestrationError![]u8 {
    return executeSingleWithDispatch(self, model_id, prompt, response_allocator, providerDispatch);
}

pub fn executeSingleWithDispatch(
    self: anytype,
    model_id: []const u8,
    prompt: []const u8,
    response_allocator: std.mem.Allocator,
    dispatch: DispatchFn,
) types.OrchestrationError![]u8 {
    self.mutex.lock();
    const model = self.models.getPtr(model_id) orelse {
        self.mutex.unlock();
        return types.OrchestrationError.ModelNotFound;
    };

    if (!model.config.enabled) {
        self.mutex.unlock();
        return types.OrchestrationError.ModelDisabled;
    }

    model.active_requests += 1;
    model.total_requests += 1;
    model.last_request_time = unixMs();
    self.mutex.unlock();

    defer {
        self.mutex.lock();
        model.active_requests -= 1;
        self.mutex.unlock();
    }

    const response = dispatch(response_allocator, model, prompt) catch {
        self.mutex.lock();
        model.consecutive_failures += 1;
        model.total_failures += 1;
        model.last_failure_time = unixMs();
        if (model.consecutive_failures >= 3) {
            model.status = .unhealthy;
        }
        self.mutex.unlock();
        return types.OrchestrationError.AllModelsFailed;
    };

    self.mutex.lock();
    model.consecutive_failures = 0;
    self.mutex.unlock();

    return response;
}

fn providerDispatch(
    response_allocator: std.mem.Allocator,
    model: *types.ModelEntry,
    prompt: []const u8,
) types.OrchestrationError![]u8 {
    const provider_id = model.config.backend.toProviderId();
    const model_name = if (model.config.model_name.len > 0)
        model.config.model_name
    else
        model.config.id;

    var result = provider_router.generate(response_allocator, .{
        .model = model_name,
        .prompt = prompt,
        .backend = provider_id,
        .fallback = &.{},
        .strict_backend = false,
        .max_tokens = model.config.max_tokens,
    }) catch return types.OrchestrationError.AllModelsFailed;
    defer result.deinit(response_allocator);

    return response_allocator.dupe(u8, result.content) catch
        return types.OrchestrationError.OutOfMemory;
}

fn calculateEnsembleConfidence(successful: usize, total: usize) f64 {
    if (total == 0) return 0.0;
    return @as(f64, @floatFromInt(successful)) / @as(f64, @floatFromInt(total));
}


test {
    std.testing.refAllDecls(@This());
}
