//! WDBX HTTP Server Module
//!
//! Provides HTTP server functionality for the WDBX vector database.
//! This module wraps the server/wdbx_http.zig implementation.

const std = @import("std");
const wdbx_http = @import("../web/wdbx_http.zig");

// Re-export main types
pub const WdbxHttpServer = wdbx_http.WdbxHttpServer;
pub const ServerConfig = wdbx_http.ServerConfig;

// Re-export functions
pub const createServer = wdbx_http.WdbxHttpServer.init;
pub const HttpError = wdbx_http.HttpError;
pub const Response = wdbx_http.Response;
