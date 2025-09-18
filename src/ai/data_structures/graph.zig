//! Graph Data Structures - Graph algorithms and representations
//!
//! This module provides graph data structures for:
//! - Directed and undirected graphs
//! - Bipartite graphs
//! - Graph traversal algorithms
//! - Memory-efficient storage

const std = @import("std");

/// Generic graph implementation
pub const Graph = struct {
    const Self = @This();

    /// Adjacency list representation
    adjacency_list: std.ArrayList(std.ArrayList(usize)),
    /// Number of vertices
    vertices: usize,
    /// Number of edges
    edges: usize,
    /// Whether the graph is directed
    directed: bool,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new graph
    pub fn init(allocator: std.mem.Allocator, vertices: usize, directed: bool) !*Self {
        const graph = try allocator.create(Self);
        graph.* = Self{
            .adjacency_list = try std.ArrayList(std.ArrayList(usize)).initCapacity(allocator, vertices),
            .vertices = vertices,
            .edges = 0,
            .directed = directed,
            .allocator = allocator,
        };

        // Initialize adjacency lists
        for (0..vertices) |_| {
            try graph.adjacency_list.append(std.ArrayList(usize).init(allocator));
        }

        return graph;
    }

    /// Deinitialize the graph
    pub fn deinit(self: *Self) void {
        for (self.adjacency_list.items) |*list| {
            list.deinit();
        }
        self.adjacency_list.deinit();
        self.allocator.destroy(self);
    }

    /// Add an edge between two vertices
    pub fn addEdge(self: *Self, from: usize, to: usize) !void {
        if (from >= self.vertices or to >= self.vertices) return error.InvalidVertex;

        try self.adjacency_list.items[from].append(to);
        self.edges += 1;

        if (!self.directed and from != to) {
            try self.adjacency_list.items[to].append(from);
        }
    }

    /// Remove an edge between two vertices
    pub fn removeEdge(self: *Self, from: usize, to: usize) !void {
        if (from >= self.vertices or to >= self.vertices) return error.InvalidVertex;

        // Remove from -> to
        if (self.removeFromList(&self.adjacency_list.items[from], to)) {
            self.edges -= 1;
        }

        if (!self.directed) {
            // Remove to -> from
            _ = self.removeFromList(&self.adjacency_list.items[to], from);
        }
    }

    fn removeFromList(list: *std.ArrayList(usize), value: usize) bool {
        for (list.items, 0..) |item, i| {
            if (item == value) {
                _ = list.orderedRemove(i);
                return true;
            }
        }
        return false;
    }

    /// Get neighbors of a vertex
    pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
        if (vertex >= self.vertices) return null;
        return self.adjacency_list.items[vertex].items;
    }

    /// Perform breadth-first search
    pub fn bfs(self: *Self, start: usize, visitor: anytype) !void {
        if (start >= self.vertices) return error.InvalidVertex;

        var visited = try std.DynamicBitSet.initEmpty(self.allocator, self.vertices);
        defer visited.deinit();

        var queue = std.ArrayList(usize).init(self.allocator);
        defer queue.deinit();

        try queue.append(start);
        visited.set(start);

        while (queue.items.len > 0) {
            const current = queue.orderedRemove(0);
            try visitor.visit(current);

            for (self.adjacency_list.items[current].items) |neighbor| {
                if (!visited.isSet(neighbor)) {
                    visited.set(neighbor);
                    try queue.append(neighbor);
                }
            }
        }
    }

    /// Perform depth-first search
    pub fn dfs(self: *Self, start: usize, visitor: anytype) !void {
        if (start >= self.vertices) return error.InvalidVertex;

        var visited = try std.DynamicBitSet.initEmpty(self.allocator, self.vertices);
        defer visited.deinit();

        try self.dfsRecursive(start, &visited, visitor);
    }

    fn dfsRecursive(self: *Self, vertex: usize, visited: *std.DynamicBitSet, visitor: anytype) !void {
        visited.set(vertex);
        try visitor.visit(vertex);

        for (self.adjacency_list.items[vertex].items) |neighbor| {
            if (!visited.isSet(neighbor)) {
                try self.dfsRecursive(neighbor, visited, visitor);
            }
        }
    }
};

/// Directed graph (alias for Graph with directed=true)
pub const DirectedGraph = struct {
    const Self = @This();

    /// Underlying graph implementation
    graph: *Graph,

    /// Initialize a new directed graph
    pub fn init(allocator: std.mem.Allocator, vertices: usize) !*Self {
        const directed_graph = try allocator.create(Self);
        directed_graph.* = Self{
            .graph = try Graph.init(allocator, vertices, true),
        };
        return directed_graph;
    }

    /// Deinitialize the graph
    pub fn deinit(self: *Self) void {
        self.graph.deinit();
        self.allocator.destroy(self);
    }

    /// Add a directed edge
    pub fn addEdge(self: *Self, from: usize, to: usize) !void {
        try self.graph.addEdge(from, to);
    }

    /// Get neighbors (outgoing edges)
    pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
        return self.graph.getNeighbors(vertex);
    }

    /// Get reverse neighbors (incoming edges)
    pub fn getReverseNeighbors(self: *Self, vertex: usize) !std.ArrayList(usize) {
        var reverse = std.ArrayList(usize).init(self.graph.allocator);
        for (0..self.graph.vertices) |i| {
            if (self.graph.getNeighbors(i)) |neighbors| {
                for (neighbors) |neighbor| {
                    if (neighbor == vertex) {
                        try reverse.append(i);
                    }
                }
            }
        }
        return reverse;
    }
};

/// Bipartite graph implementation
pub const BipartiteGraph = struct {
    const Self = @This();

    /// Underlying graph
    graph: *Graph,
    /// Set A vertices
    set_a: std.DynamicBitSet,
    /// Set B vertices
    set_b: std.DynamicBitSet,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new bipartite graph
    pub fn init(allocator: std.mem.Allocator, size_a: usize, size_b: usize) !*Self {
        const total_vertices = size_a + size_b;
        const bipartite = try allocator.create(Self);
        bipartite.* = Self{
            .graph = try Graph.init(allocator, total_vertices, true),
            .set_a = try std.DynamicBitSet.initEmpty(allocator, total_vertices),
            .set_b = try std.DynamicBitSet.initEmpty(allocator, total_vertices),
            .allocator = allocator,
        };

        // Set up partitions
        for (0..size_a) |i| {
            bipartite.set_a.set(i);
        }
        for (size_a..total_vertices) |i| {
            bipartite.set_b.set(i);
        }

        return bipartite;
    }

    /// Deinitialize the graph
    pub fn deinit(self: *Self) void {
        self.graph.deinit();
        self.set_a.deinit();
        self.set_b.deinit();
        self.allocator.destroy(self);
    }

    /// Add an edge between sets (only allowed between different sets)
    pub fn addEdge(self: *Self, a_vertex: usize, b_vertex: usize) !void {
        const actual_b_vertex = self.graph.vertices / 2 + b_vertex;

        if (a_vertex >= self.graph.vertices / 2 or b_vertex >= self.graph.vertices / 2) {
            return error.InvalidVertex;
        }

        try self.graph.addEdge(a_vertex, actual_b_vertex);
    }

    /// Check if the graph is bipartite (validate no edges within same set)
    pub fn validateBipartite(self: *Self) bool {
        for (0..self.graph.vertices) |i| {
            if (self.graph.getNeighbors(i)) |neighbors| {
                const in_set_a = self.set_a.isSet(i);
                for (neighbors) |neighbor| {
                    const neighbor_in_set_a = self.set_a.isSet(neighbor);
                    if (in_set_a == neighbor_in_set_a) {
                        return false; // Edge within same set
                    }
                }
            }
        }
        return true;
    }

    /// Get vertices in set A
    pub fn getSetA(self: *Self) std.ArrayList(usize) {
        var result = std.ArrayList(usize).init(self.allocator);
        var i: usize = 0;
        while (i < self.graph.vertices) : (i += 1) {
            if (self.set_a.isSet(i)) {
                result.append(i) catch {};
            }
        }
        return result;
    }

    /// Get vertices in set B
    pub fn getSetB(self: *Self) std.ArrayList(usize) {
        var result = std.ArrayList(usize).init(self.allocator);
        var i: usize = 0;
        while (i < self.graph.vertices) : (i += 1) {
            if (self.set_b.isSet(i)) {
                result.append(i) catch {};
            }
        }
        return result;
    }
};
