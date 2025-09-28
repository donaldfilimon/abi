//! Spatial Data Structures - Tree-based spatial indexing
//!
//! This module provides spatial data structures for:
//! - KD-tree for k-dimensional nearest neighbor search
//! - Quad-tree for 2D spatial indexing
//! - Ball-tree for hierarchical clustering
//! - LSH Forest for approximate nearest neighbor search

const std = @import("std");

/// Node in a KD-tree
const KDNode = struct {
    point: []f32,
    left: ?*KDNode,
    right: ?*KDNode,
    axis: usize,
};

/// KD-tree for efficient nearest neighbor search in k-dimensional space
pub fn KDTree(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Root node
        root: ?*KDNode,
        /// Dimensionality of the space
        dimensions: usize,
        /// Memory allocator
        allocator: std.mem.Allocator,
        /// All nodes for cleanup
        nodes: std.ArrayList(*KDNode),

        /// Initialize a new KD-tree
        pub fn init(allocator: std.mem.Allocator, dimensions: usize) !*Self {
            const tree = try allocator.create(Self);
            tree.* = Self{
                .root = null,
                .dimensions = dimensions,
                .allocator = allocator,
                .nodes = try std.ArrayList(*KDNode).initCapacity(allocator, 0),
            };
            return tree;
        }

        /// Deinitialize the tree
        pub fn deinit(self: *Self) void {
            // Free all points and nodes
            for (self.nodes.items) |node| {
                self.allocator.free(node.point);
                self.allocator.destroy(node);
            }
            self.nodes.deinit();
            self.allocator.destroy(self);
        }

        /// Insert a point into the tree
        pub fn insert(self: *Self, point: []const T) !void {
            if (point.len != self.dimensions) return error.InvalidDimensions;

            const point_copy = try self.allocator.dupe(T, point);
            const new_node = try self.allocator.create(KDNode);
            new_node.* = KDNode{
                .point = point_copy,
                .left = null,
                .right = null,
                .axis = 0,
            };

            try self.nodes.append(new_node);

            if (self.root == null) {
                self.root = new_node;
            } else {
                try self.insertRecursive(self.root.?, new_node, 0);
            }
        }

        fn insertRecursive(self: *Self, current: *KDNode, new_node: *KDNode, depth: usize) !void {
            const axis = depth % self.dimensions;
            new_node.axis = axis;

            const current_val = if (T == f32) current.point[axis] else @as(f32, @floatFromInt(current.point[axis]));
            const new_val = if (T == f32) new_node.point[axis] else @as(f32, @floatFromInt(new_node.point[axis]));

            if (new_val < current_val) {
                if (current.left == null) {
                    current.left = new_node;
                } else {
                    try self.insertRecursive(current.left.?, new_node, depth + 1);
                }
            } else {
                if (current.right == null) {
                    current.right = new_node;
                } else {
                    try self.insertRecursive(current.right.?, new_node, depth + 1);
                }
            }
        }

        /// Find nearest neighbor to a query point
        pub fn nearestNeighbor(self: *Self, query: []const T) !?[]T {
            if (query.len != self.dimensions or self.root == null) return null;

            var best_point: ?[]T = null;
            var best_distance: f32 = std.math.inf(f32);

            self.nearestNeighborRecursive(self.root.?, query, 0, &best_point, &best_distance);

            return best_point;
        }

        fn nearestNeighborRecursive(self: *Self, node: *KDNode, query: []const T, depth: usize, best_point: *?[]T, best_distance: *f32) void {
            const axis = depth % self.dimensions;
            const query_val = if (T == f32) query[axis] else @as(f32, @floatFromInt(query[axis]));
            const node_val = if (T == f32) node.point[axis] else @as(f32, @floatFromInt(node.point[axis]));

            const max_distance = self.distance(query, node.point);
            if (max_distance < best_distance.*) {
                best_distance.* = max_distance;
                best_point.* = node.point;
            }

            const go_left = if (query_val < node_val) node.left else node.right;
            const go_right = if (query_val < node_val) node.right else node.left;

            if (go_left) |left| {
                self.nearestNeighborRecursive(left, query, depth + 1, best_point, best_distance);
            }

            if (go_right) |right| {
                self.nearestNeighborRecursive(right, query, depth + 1, best_point, best_distance);
            }
        }

        fn distance(self: *Self, a: []const T, b: []const T) f32 {
            _ = self;
            var sum: f32 = 0.0;
            for (0..a.len) |i| {
                const diff = if (T == f32) a[i] - b[i] else @as(f32, @floatFromInt(a[i] - b[i]));
                sum += diff * diff;
            }
            return @sqrt(sum);
        }
    };
}

/// Quad-tree for 2D spatial indexing
pub const QuadTree = struct {
    const Self = @This();

    /// Boundary of this quad tree node
    boundary: struct {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    /// Points in this node
    points: std.ArrayList(struct { x: f32, y: f32 }),
    /// Capacity before subdividing
    capacity: usize,
    /// Child nodes
    children: ?[4]*QuadTree,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new quad tree
    pub fn init(allocator: std.mem.Allocator, x: f32, y: f32, width: f32, height: f32, capacity: usize) !*Self {
        const tree = try allocator.create(Self);
        tree.* = Self{
            .boundary = .{ .x = x, .y = y, .width = width, .height = height },
            .points = try std.ArrayList(struct { x: f32, y: f32 }).initCapacity(allocator, capacity),
            .capacity = capacity,
            .children = null,
            .allocator = allocator,
        };
        return tree;
    }

    /// Deinitialize the tree
    pub fn deinit(self: *Self) void {
        self.points.deinit();
        if (self.children) |children| {
            for (children) |child| {
                child.deinit();
            }
            self.allocator.free(children);
        }
        self.allocator.destroy(self);
    }

    /// Insert a point into the tree
    pub fn insert(self: *Self, x: f32, y: f32) !void {
        // Check if point is within boundary
        if (x < self.boundary.x or x > self.boundary.x + self.boundary.width or
            y < self.boundary.y or y > self.boundary.y + self.boundary.height)
        {
            return;
        }

        if (self.children == null) {
            // Try to add to this node
            try self.points.append(.{ .x = x, .y = y });

            // Subdivide if capacity exceeded
            if (self.points.items.len > self.capacity) {
                try self.subdivide();
            }
        } else {
            // Insert into appropriate child
            const child = self.getChildIndex(x, y);
            try self.children.?[child].insert(x, y);
        }
    }

    fn subdivide(self: *Self) !void {
        const x = self.boundary.x;
        const y = self.boundary.y;
        const w = self.boundary.width / 2;
        const h = self.boundary.height / 2;

        var children = try self.allocator.alloc(*QuadTree, 4);
        children[0] = try QuadTree.init(self.allocator, x, y, w, h, self.capacity);
        children[1] = try QuadTree.init(self.allocator, x + w, y, w, h, self.capacity);
        children[2] = try QuadTree.init(self.allocator, x, y + h, w, h, self.capacity);
        children[3] = try QuadTree.init(self.allocator, x + w, y + h, w, h, self.capacity);
        self.children = children;

        // Reinsert all points into children
        for (self.points.items) |point| {
            const child = self.getChildIndex(point.x, point.y);
            try self.children.?[child].insert(point.x, point.y);
        }

        // Clear points from this node
        self.points.clearAndFree();
    }

    fn getChildIndex(self: *Self, x: f32, y: f32) usize {
        const mid_x = self.boundary.x + self.boundary.width / 2;
        const mid_y = self.boundary.y + self.boundary.height / 2;

        if (x < mid_x) {
            return if (y < mid_y) 0 else 2;
        } else {
            return if (y < mid_y) 1 else 3;
        }
    }
};

/// Ball-tree for hierarchical clustering
pub const BallTree = struct {
    const Self = @This();

    /// Center of the ball
    center: []f32,
    /// Radius of the ball
    radius: f32,
    /// Left subtree
    left: ?*BallTree,
    /// Right subtree
    right: ?*BallTree,
    /// Points in this node (leaf nodes only)
    points: std.ArrayList([]f32),
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new ball tree
    pub fn init(allocator: std.mem.Allocator, points: []const []const f32) !*Self {
        const tree = try allocator.create(Self);
        tree.* = Self{
            .center = try allocator.alloc(f32, points[0].len),
            .radius = 0.0,
            .left = null,
            .right = null,
            .points = try std.ArrayList([]f32).initCapacity(allocator, points.len),
            .allocator = allocator,
        };

        // Copy points
        for (points) |point| {
            const copy = try allocator.dupe(f32, point);
            try tree.points.append(copy);
        }

        // Calculate center and radius
        try tree.calculateBoundingBall();

        return tree;
    }

    /// Deinitialize the tree
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.center);
        for (self.points.items) |point| {
            self.allocator.free(point);
        }
        self.points.deinit();

        if (self.left) |left| left.deinit();
        if (self.right) |right| right.deinit();

        self.allocator.destroy(self);
    }

    fn calculateBoundingBall(self: *Self) !void {
        if (self.points.items.len == 0) return;

        // Initialize center with first point
        @memcpy(self.center, self.points.items[0]);

        // Find minimum bounding ball
        for (self.points.items[1..]) |point| {
            // Update center and radius
            // This is a simplified implementation
            for (0..self.center.len) |i| {
                self.center[i] = (self.center[i] + point[i]) / 2.0;
            }
        }

        // Calculate radius
        var max_dist: f32 = 0.0;
        for (self.points.items) |point| {
            const point_distance = self.distance(self.center, point);
            max_dist = @max(max_dist, point_distance);
        }
        self.radius = max_dist;
    }

    fn distance(self: *Self, a: []const f32, b: []const f32) f32 {
        _ = self;
        var sum: f32 = 0.0;
        for (0..a.len) |i| {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return @sqrt(sum);
    }
};

/// LSH Forest for approximate nearest neighbor search
pub const LSHForest = struct {
    const Self = @This();

    /// Hash tables
    tables: std.ArrayList(std.AutoHashMap(u64, std.ArrayList([]f32))),
    /// Number of hash tables
    num_tables: usize,
    /// Number of hash functions per table
    num_hashes: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new LSH forest
    pub fn init(allocator: std.mem.Allocator, num_tables: usize, num_hashes: usize) !*Self {
        const forest = try allocator.create(Self);
        forest.* = Self{
            .tables = try std.ArrayList(std.AutoHashMap(u64, std.ArrayList([]f32))).initCapacity(allocator, num_tables),
            .num_tables = num_tables,
            .num_hashes = num_hashes,
            .allocator = allocator,
        };

        // Initialize hash tables
        for (0..num_tables) |_| {
            try forest.tables.append(std.AutoHashMap(u64, std.ArrayList([]f32)).init(allocator));
        }

        return forest;
    }

    /// Deinitialize the forest
    pub fn deinit(self: *Self) void {
        for (self.tables.items) |*table| {
            var it = table.iterator();
            while (it.next()) |entry| {
                for (entry.value_ptr.*.items) |point| {
                    self.allocator.free(point);
                }
                entry.value_ptr.*.deinit();
            }
            table.deinit();
        }
        self.tables.deinit();
        self.allocator.destroy(self);
    }

    /// Add a point to the forest
    pub fn add(self: *Self, point: []const f32) !void {
        const point_copy = try self.allocator.dupe(f32, point);

        for (self.tables.items) |*table| {
            const hash = self.hashPoint(point, table == &self.tables.items[0]);
            const gop = try table.getOrPut(hash);
            if (!gop.found_existing) {
                gop.value_ptr.* = std.ArrayList([]f32){};
            }
            try gop.value_ptr.*.append(point_copy);
        }
    }

    /// Query approximate nearest neighbors
    pub fn query(self: *Self, query_point: []const f32, k: usize) !std.ArrayList([]f32) {
        var candidates = std.ArrayList([]f32){};
        defer candidates.deinit(self.allocator);

        // Collect candidates from all tables
        for (self.tables.items) |*table| {
            const hash = self.hashPoint(query_point, table == &self.tables.items[0]);
            if (table.get(hash)) |bucket| {
                for (bucket.items) |point| {
                    const copy = try self.allocator.dupe(f32, point);
                    try candidates.append(self.allocator, copy);
                }
            }
        }

        // Sort by distance and return top k
        const result = try std.ArrayList([]f32).initCapacity(self.allocator, @min(k, candidates.items.len));
        // In a real implementation, you'd sort by distance here
        for (candidates.items[0..@min(k, candidates.items.len)]) |point| {
            const copy = try self.allocator.dupe(f32, point);
            try result.append(copy);
        }

        return result;
    }

    fn hashPoint(self: *Self, point: []const f32, seed: bool) u64 {
        _ = self;
        var hash: u64 = if (seed) 0 else 1;
        for (point) |val| {
            const val_bits = @as(u64, @bitCast(@as(f32, val)));
            hash = std.hash.Wyhash.hash(hash, std.mem.asBytes(&val_bits));
        }
        return hash;
    }
};
