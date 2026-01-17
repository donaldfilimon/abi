//! # Plugin System
//!
//! Lightweight plugin architecture for runtime-loadable modules.
//!
//! ## Features
//!
//! - **Runtime Loading**: Load plugins dynamically
//! - **Interface Compliance**: Type-safe plugin contracts
//! - **Lifecycle Management**: Init/deinit coordination
//! - **Registry**: Centralized plugin discovery
//!
//! ## Plugin Interface
//!
//! Plugins must implement the `Plugin` interface:
//!
//! ```zig
//! pub const Plugin = struct {
//!     name: []const u8,
//!     version: []const u8,
//!     init: *const fn (allocator: std.mem.Allocator) anyerror!void,
//!     deinit: *const fn () void,
//!     execute: ?*const fn (ctx: *anyopaque) anyerror!void,
//! };
//! ```
//!
//! ## Creating a Plugin
//!
//! ```zig
//! // my_plugin.zig
//! const std = @import("std");
//! const Plugin = @import("shared").plugins.Plugin;
//!
//! var state: ?*MyState = null;
//!
//! pub const plugin = Plugin{
//!     .name = "my_plugin",
//!     .version = "1.0.0",
//!     .init = init,
//!     .deinit = deinit,
//!     .execute = execute,
//! };
//!
//! fn init(allocator: std.mem.Allocator) !void {
//!     state = try allocator.create(MyState);
//! }
//!
//! fn deinit() void {
//!     if (state) |s| allocator.destroy(s);
//! }
//!
//! fn execute(ctx: *anyopaque) !void {
//!     // Plugin logic
//! }
//! ```
//!
//! ## Registration
//!
//! ```zig
//! var registry = abi.plugins.Registry.init(allocator);
//! defer registry.deinit();
//!
//! try registry.register(&my_plugin.plugin);
//! try registry.initAll();
//! ```
//!
//! ## See Also
//!
//! - [Framework Module](../../framework/README.md)

