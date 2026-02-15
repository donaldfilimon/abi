//! Generic Radix Tree for Path Matching
//!
//! A comptime-generic radix (prefix) tree for O(path_segments) URL matching
//! with support for path parameters (`{name}`) and wildcards (`*`).
//!
//! Used by both `gateway` and `pages` modules to avoid duplicating ~300 lines
//! of identical routing logic.
//!
//! ## Usage
//!
//! ```zig
//! const tree = RadixTree(u32);
//! var root = try tree.createRoot(allocator);
//! defer tree.destroyRoot(root, allocator);
//!
//! try tree.insert(root, allocator, "/users/{id}", 0);
//! var result = tree.MatchResult{};
//! if (tree.match(root, "/users/42", &result)) {
//!     // result.terminal_idx == 0, result.params[0] == .{ "id", "42" }
//! }
//! ```

const std = @import("std");

/// A radix tree parameterized on the terminal index type.
///
/// `TerminalIndex` is the integer type stored at terminal nodes to identify
/// which entry (route, page, etc.) matched. Typically `u32`.
pub fn RadixTree(comptime TerminalIndex: type) type {
    return struct {
        const Self = @This();

        /// Maximum number of extracted path parameters per match.
        pub const max_params = 8;

        /// A single extracted path parameter from a match.
        pub const Param = struct {
            name: []const u8 = "",
            value: []const u8 = "",
        };

        /// Result of a path match operation.
        pub const MatchResult = struct {
            terminal_idx: ?TerminalIndex = null,
            params: [max_params]Param = [_]Param{.{}} ** max_params,
            param_count: u8 = 0,

            /// Look up a parameter value by name.
            pub fn getParam(self: *const MatchResult, name: []const u8) ?[]const u8 {
                for (self.params[0..self.param_count]) |p| {
                    if (std.mem.eql(u8, p.name, name)) return p.value;
                }
                return null;
            }
        };

        /// Internal radix tree node.
        pub const Node = struct {
            segment: []const u8 = "",
            is_param: bool = false,
            param_name: []const u8 = "",
            is_wildcard: bool = false,
            terminal_idx: ?TerminalIndex = null,
            children: std.ArrayListUnmanaged(*Node) = .empty,

            /// Recursively free all children and their child lists.
            pub fn deinitRecursive(self: *Node, allocator: std.mem.Allocator) void {
                for (self.children.items) |child| {
                    child.deinitRecursive(allocator);
                    allocator.destroy(child);
                }
                self.children.deinit(allocator);
            }

            /// Find a child matching a static segment exactly.
            pub fn findChild(self: *const Node, segment: []const u8) ?*Node {
                for (self.children.items) |child| {
                    if (!child.is_param and !child.is_wildcard and
                        std.mem.eql(u8, child.segment, segment))
                    {
                        return child;
                    }
                }
                return null;
            }

            /// Find the first child that is a path parameter (`{name}`).
            pub fn findParamChild(self: *const Node) ?*Node {
                for (self.children.items) |child| {
                    if (child.is_param) return child;
                }
                return null;
            }

            /// Find the first child that is a wildcard (`*`).
            pub fn findWildcardChild(self: *const Node) ?*Node {
                for (self.children.items) |child| {
                    if (child.is_wildcard) return child;
                }
                return null;
            }
        };

        // ── Tree Operations ───────────────────────────────────────────

        /// Create a heap-allocated root node.
        pub fn createRoot(allocator: std.mem.Allocator) !*Node {
            const root = try allocator.create(Node);
            root.* = .{};
            return root;
        }

        /// Destroy a root node and all of its descendants.
        pub fn destroyRoot(root: *Node, allocator: std.mem.Allocator) void {
            root.deinitRecursive(allocator);
            allocator.destroy(root);
        }

        /// Insert a path pattern into the tree with the given terminal index.
        ///
        /// Path segments are split on `/`. Segments like `{name}` become parameter
        /// nodes, and `*` becomes a wildcard terminal.
        pub fn insert(
            root: *Node,
            allocator: std.mem.Allocator,
            path: []const u8,
            idx: TerminalIndex,
        ) !void {
            var current = root;
            var segments = splitPath(path);

            while (segments.next()) |segment| {
                if (segment.len > 0 and segment[0] == '*') {
                    // Wildcard — terminal
                    const child = try allocator.create(Node);
                    child.* = .{
                        .is_wildcard = true,
                        .terminal_idx = idx,
                    };
                    try current.children.append(allocator, child);
                    return;
                }

                if (segment.len > 2 and segment[0] == '{' and segment[segment.len - 1] == '}') {
                    // Path parameter: {name}
                    const param_name = segment[1 .. segment.len - 1];
                    if (current.findParamChild()) |child| {
                        current = child;
                    } else {
                        const child = try allocator.create(Node);
                        child.* = .{
                            .is_param = true,
                            .param_name = param_name,
                        };
                        try current.children.append(allocator, child);
                        current = child;
                    }
                } else {
                    // Static segment
                    if (current.findChild(segment)) |child| {
                        current = child;
                    } else {
                        const child = try allocator.create(Node);
                        child.* = .{ .segment = segment };
                        try current.children.append(allocator, child);
                        current = child;
                    }
                }
            }

            current.terminal_idx = idx;
        }

        /// Match a request path against the tree, filling `result` with the
        /// terminal index and any extracted parameters.
        ///
        /// Returns `true` if a terminal node was reached.
        pub fn match(
            root: *const Node,
            path: []const u8,
            result: *MatchResult,
        ) bool {
            var segments = splitPath(path);
            return matchNode(root, &segments, result);
        }

        /// Split a URL path into segments, stripping the leading `/`.
        pub fn splitPath(path: []const u8) std.mem.SplitIterator(u8, .scalar) {
            const trimmed = if (path.len > 0 and path[0] == '/') path[1..] else path;
            return std.mem.splitScalar(u8, trimmed, '/');
        }

        // ── Internal ──────────────────────────────────────────────────

        fn matchNode(
            node: *const Node,
            segments: *std.mem.SplitIterator(u8, .scalar),
            result: *MatchResult,
        ) bool {
            const segment = segments.next() orelse {
                // No more segments — check if current node is terminal
                if (node.terminal_idx) |idx| {
                    result.terminal_idx = idx;
                    return true;
                }
                return false;
            };

            // Try exact match first
            if (node.findChild(segment)) |child| {
                var child_segments = segments.*;
                if (matchNode(child, &child_segments, result)) {
                    segments.* = child_segments;
                    return true;
                }
            }

            // Try param match
            if (node.findParamChild()) |child| {
                var child_segments = segments.*;
                if (matchNode(child, &child_segments, result)) {
                    if (result.param_count < max_params) {
                        result.params[result.param_count] = .{
                            .name = child.param_name,
                            .value = segment,
                        };
                        result.param_count += 1;
                    }
                    segments.* = child_segments;
                    return true;
                }
            }

            // Try wildcard match
            if (node.findWildcardChild()) |child| {
                if (child.terminal_idx) |idx| {
                    result.terminal_idx = idx;
                    return true;
                }
            }

            return false;
        }
    };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

test "radix_tree static route insertion and match" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/api/users", 0);
    try Tree.insert(root, allocator, "/api/orders", 1);

    var r1 = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/api/users", &r1));
    try std.testing.expectEqual(@as(?u32, 0), r1.terminal_idx);

    var r2 = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/api/orders", &r2));
    try std.testing.expectEqual(@as(?u32, 1), r2.terminal_idx);
}

test "radix_tree path parameter extraction" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/users/{id}", 0);

    var result = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/users/42", &result));
    try std.testing.expectEqual(@as(?u32, 0), result.terminal_idx);
    try std.testing.expectEqual(@as(u8, 1), result.param_count);

    const id_val = result.getParam("id");
    try std.testing.expect(id_val != null);
    try std.testing.expectEqualStrings("42", id_val.?);
}

test "radix_tree multi-param extraction" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/items/{category}/{id}", 0);

    var result = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/items/books/123", &result));
    try std.testing.expectEqual(@as(u8, 2), result.param_count);

    const cat = result.getParam("category");
    try std.testing.expect(cat != null);
    const id = result.getParam("id");
    try std.testing.expect(id != null);
}

test "radix_tree wildcard matching" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/api/*", 0);

    var result = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/api/anything", &result));
    try std.testing.expectEqual(@as(?u32, 0), result.terminal_idx);
}

test "radix_tree no match returns false" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/api/known", 0);

    var result = Tree.MatchResult{};
    try std.testing.expect(!Tree.match(root, "/api/unknown", &result));
    try std.testing.expectEqual(@as(?u32, null), result.terminal_idx);
}

test "radix_tree prefers exact over param" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    try Tree.insert(root, allocator, "/users/admin", 0);
    try Tree.insert(root, allocator, "/users/{id}", 1);

    // "admin" should match the exact static route (index 0), not the param route
    var result = Tree.MatchResult{};
    try std.testing.expect(Tree.match(root, "/users/admin", &result));
    try std.testing.expectEqual(@as(?u32, 0), result.terminal_idx);
    try std.testing.expectEqual(@as(u8, 0), result.param_count);
}

test "radix_tree empty root match fails" {
    const Tree = RadixTree(u32);
    const allocator = std.testing.allocator;

    const root = try Tree.createRoot(allocator);
    defer Tree.destroyRoot(root, allocator);

    var result = Tree.MatchResult{};
    try std.testing.expect(!Tree.match(root, "/anything", &result));
}

test "radix_tree splitPath strips leading slash" {
    const Tree = RadixTree(u32);
    var iter = Tree.splitPath("/api/v1/users");

    try std.testing.expectEqualStrings("api", iter.next().?);
    try std.testing.expectEqualStrings("v1", iter.next().?);
    try std.testing.expectEqualStrings("users", iter.next().?);
    try std.testing.expect(iter.next() == null);
}
