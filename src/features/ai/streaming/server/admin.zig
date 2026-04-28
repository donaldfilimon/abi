//! Admin and Metrics Endpoint Handlers
//!
//! Handles admin model hot-reload, models list, and metrics endpoints.

const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const build_options = @import("build_options");
const observability = if (build_options.feat_observability) @import("../../../observability/mod.zig") else @import("../../../observability/stub.zig");
const handlers = @import("handlers.zig");
const routing = @import("routing.zig");
const config_mod = @import("config.zig");
const request_types = @import("../request_types.zig");
const server_types = @import("types.zig");

const extractJsonString = request_types.extractJsonString;
const histogramPercentile = handlers.histogramPercentile;

/// Handle metrics endpoint.
pub fn handleMetrics(server: anytype, request: *std.http.Server.Request) !void {
    const metrics = server.getMetrics() orelse {
        return routing.respondJson(request, "{\"status\":\"disabled\"}", .service_unavailable);
    };

    const snap = metrics.snapshot();

    var active_streams_count: u32 = 0;
    for (snap.backend_active_streams) |count| {
        if (count > 0) {
            active_streams_count += @intCast(count);
        }
    }

    var primary_idx: usize = 0;
    var max_requests: u64 = 0;
    for (snap.backend_requests, 0..) |count, i| {
        if (count > max_requests) {
            max_requests = count;
            primary_idx = i;
        }
    }

    const ttft_p50 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.50);
    const ttft_p95 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.95);
    const ttft_p99 = histogramPercentile(&metrics.backend_metrics[primary_idx].token_latency_ms, 0.99);

    const now_ms = shared_utils.unixMs();
    const uptime_ms: u64 = if (now_ms >= server.start_time_ms)
        @intCast(now_ms - server.start_time_ms)
    else
        0;

    const throughput_tps: f64 = if (uptime_ms > 0)
        @as(f64, @floatFromInt(snap.total_tokens)) / (@as(f64, @floatFromInt(uptime_ms)) / 1000.0)
    else
        0.0;

    var json = std.ArrayListUnmanaged(u8).empty;
    defer json.deinit(server.allocator);

    try json.print(
        server.allocator,
        "{{\"status\":\"ok\",\"uptime_ms\":{d},\"active_streams\":{d},\"max_streams\":{d}," ++
            "\"queue_depth\":{d},\"total_tokens\":{d},\"total_requests\":{d},\"total_errors\":{d}," ++
            "\"ttft_ms_p50\":{d},\"ttft_ms_p95\":{d},\"ttft_ms_p99\":{d},\"throughput_tps\":{d:.2}}}",
        .{
            uptime_ms,
            active_streams_count,
            server.config.max_concurrent_streams,
            0,
            snap.total_tokens,
            snap.total_streams,
            snap.total_errors,
            ttft_p50,
            ttft_p95,
            ttft_p99,
            throughput_tps,
        },
    );

    return routing.respondJson(request, json.items, .ok);
}

/// Handle models list (OpenAI-compatible).
pub fn handleModelsList(server: anytype, request: *std.http.Server.Request) !void {
    const models_json = try server.backend_router.listModelsJson(server.allocator);
    defer server.allocator.free(models_json);
    return routing.respondJson(request, models_json, .ok);
}

/// Handle admin model hot-reload.
pub fn handleAdminReload(server: anytype, request: *std.http.Server.Request) !void {
    if (request.head.method != .POST) {
        return routing.respondJson(
            request,
            "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}",
            .method_not_allowed,
        );
    }

    const body = try routing.readRequestBody(server.allocator, request);
    defer server.allocator.free(body);

    const model_path = extractJsonString(body, "model_path") orelse {
        return routing.respondJson(
            request,
            "{\"error\":{\"message\":\"missing model_path field\",\"type\":\"invalid_request_error\"}}",
            .bad_request,
        );
    };

    var timer = time.Timer.start() catch {
        return performModelReload(server, request, model_path);
    };

    while (server.active_streams.load(.seq_cst) > 0) {
        if (timer.read() >= config_mod.ADMIN_RELOAD_DRAIN_TIMEOUT_NS) {
            return routing.respondJson(
                request,
                "{\"error\":{\"message\":\"timeout waiting for active streams to drain\",\"type\":\"timeout_error\"}}",
                .request_timeout,
            );
        }
        const poll_timer = time.Timer.start() catch continue;
        var pt = poll_timer;
        while (pt.read() < config_mod.ADMIN_RELOAD_POLL_INTERVAL_NS) {
            std.atomic.spinLoopHint();
        }
    }

    return performModelReload(server, request, model_path);
}

/// Perform the actual model reload on the local backend.
fn performModelReload(server: anytype, request: *std.http.Server.Request, model_path: []const u8) !void {
    const backend = server.backend_router.getBackend(.local) catch {
        return routing.respondJson(
            request,
            "{\"error\":{\"message\":\"local backend unavailable\",\"type\":\"backend_error\"}}",
            .internal_server_error,
        );
    };

    backend.impl.local.loadModel(model_path) catch {
        return routing.respondJson(
            request,
            "{\"error\":{\"message\":\"model reload failed\",\"type\":\"model_error\"}}",
            .internal_server_error,
        );
    };

    return routing.respondJson(
        request,
        "{\"status\":\"ok\",\"message\":\"model reloaded successfully\"}",
        .ok,
    );
}

test {
    std.testing.refAllDecls(@This());
}
