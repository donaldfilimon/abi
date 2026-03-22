//! Integration Tests: Web Module
//!
//! Verifies web module type exports, handler types, route types,
//! and basic API contracts without making real HTTP requests.

const std = @import("std");
const abi = @import("abi");

const web = abi.web;

// ============================================================================
// Core type availability
// ============================================================================

test "web: Response type exists" {
    const R = web.Response;
    _ = R;
}

test "web: HttpClient type exists" {
    const HC = web.HttpClient;
    _ = HC;
}

test "web: RequestOptions type exists" {
    const RO = web.RequestOptions;
    _ = RO;
}

test "web: Context type exists" {
    const Ctx = web.Context;
    _ = Ctx;
}

test "web: WebError type exists" {
    const WE = web.WebError;
    _ = WE;
}

// ============================================================================
// Handler types
// ============================================================================

test "web: ChatHandler type exists" {
    const CH = web.ChatHandler;
    _ = CH;
}

test "web: ChatRequest type exists" {
    const CR = web.ChatRequest;
    _ = CR;
}

test "web: ChatResponse type exists" {
    const CR = web.ChatResponse;
    _ = CR;
}

test "web: ChatResult type exists" {
    const CR = web.ChatResult;
    _ = CR;
}

// ============================================================================
// Route types
// ============================================================================

test "web: ProfileRouter type exists" {
    const PR = web.ProfileRouter;
    _ = PR;
}

test "web: Route type exists" {
    const R = web.Route;
    _ = R;
}

test "web: RouteContext type exists" {
    const RC = web.RouteContext;
    _ = RC;
}

// ============================================================================
// Weather types
// ============================================================================

test "web: WeatherClient type exists" {
    const WC = web.WeatherClient;
    _ = WC;
}

test "web: WeatherConfig type exists" {
    const WC = web.WeatherConfig;
    _ = WC;
}

// ============================================================================
// Submodule availability
// ============================================================================

test "web: server submodule exists" {
    const S = web.server;
    _ = S;
}

test "web: middleware submodule exists" {
    const M = web.middleware;
    _ = M;
}

test "web: types submodule exists" {
    const T = web.types;
    _ = T;
}

// ============================================================================
// JSON types
// ============================================================================

test "web: JsonValue type exists" {
    const JV = web.JsonValue;
    _ = JV;
}

test "web: ParsedJson type exists" {
    const PJ = web.ParsedJson;
    _ = PJ;
}

test {
    std.testing.refAllDecls(@This());
}
