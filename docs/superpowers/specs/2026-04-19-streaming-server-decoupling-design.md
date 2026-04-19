# Streaming Server Decoupling Design

## Overview
This specification details the refactor of `src/features/ai/streaming/server/mod.zig` into a decoupled, registry-based endpoint dispatch system. The goal is to move from a hard-coded central router to a compile-time discovery mechanism, improving extensibility and maintainability.

## Architectural Changes

### 1. Registry-Based Dispatch
We will replace the existing `routing_mod.dispatchRequest` function with a registry-based approach. 
- **Endpoint Definition**: A new `types.zig` in the streaming server directory will define an `Endpoint` struct.
- **Static Registry**: A central `registry.zig` will leverage Zig's `comptime` to discover endpoints exported by sub-modules (`openai.zig`, `admin.zig`, etc.).
- **Dispatch Logic**: The main server will iterate over this registry at runtime to dispatch incoming HTTP requests, eliminating the need to modify the central router for new endpoints.

### 2. File Restructuring
- `mod.zig`: Retained as a thin facade.
- `registry.zig`: New module responsible for compile-time discovery of registered endpoints.
- `routing.zig`: Refactored to focus on protocol dispatching (SSE, WebSocket) instead of hard-coded routing.
- `types.zig`: Central location for `Endpoint` and other server-specific shared types.

## Safety and Parity
- **Mod/Stub Consistency**: All new sub-modules must provide both a `mod.zig` and `stub.zig` if they encapsulate a new feature.
- **Type Safety**: Endpoint handlers will use strongly typed `ConnectionContext` and `Request` wrappers.

## Proposed Components
- `Endpoint`: `{ .path: []const u8, .method: std.http.Method, .handler: *const fn(...) anyerror!void }`
- `Router`: A static map or sorted array of `Endpoint` instances.
- `Context`: Refactored to decouple request lifecycle from server state.

---
*Review and Approval: Please let me know if this architectural direction meets your requirements before I create the implementation plan.*
