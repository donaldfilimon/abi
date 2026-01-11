//! Metadata filtering for vector search.
//!
//! Provides configurable metadata filtering with various operators
//! for pre-filter and post-filter search strategies.

const std = @import("std");

/// Filter comparison operators.
pub const FilterOperator = enum {
    /// Equal to value.
    eq,
    /// Not equal to value.
    ne,
    /// Greater than value.
    gt,
    /// Greater than or equal to value.
    gte,
    /// Less than value.
    lt,
    /// Less than or equal to value.
    lte,
    /// Contains substring.
    contains,
    /// Starts with prefix.
    starts_with,
    /// Ends with suffix.
    ends_with,
    /// Value is in list.
    in_list,
    /// Value is not in list.
    not_in_list,
    /// Field exists.
    exists,
    /// Field does not exist.
    not_exists,
    /// Matches regex pattern.
    regex,
    /// Between two values (inclusive).
    between,
};

/// Metadata value types.
pub const MetadataValue = union(enum) {
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
    string_list: []const []const u8,
    integer_list: []const i64,
    null_value: void,

    /// Check equality with another value.
    pub fn eql(self: MetadataValue, other: MetadataValue) bool {
        return switch (self) {
            .string => |s| switch (other) {
                .string => |o| std.mem.eql(u8, s, o),
                else => false,
            },
            .integer => |i| switch (other) {
                .integer => |o| i == o,
                .float => |o| @as(f64, @floatFromInt(i)) == o,
                else => false,
            },
            .float => |f| switch (other) {
                .float => |o| f == o,
                .integer => |o| f == @as(f64, @floatFromInt(o)),
                else => false,
            },
            .boolean => |b| switch (other) {
                .boolean => |o| b == o,
                else => false,
            },
            .null_value => switch (other) {
                .null_value => true,
                else => false,
            },
            .string_list, .integer_list => false, // Lists not compared directly
        };
    }

    /// Compare values for ordering.
    pub fn compare(self: MetadataValue, other: MetadataValue) ?std.math.Order {
        return switch (self) {
            .integer => |i| switch (other) {
                .integer => |o| std.math.order(i, o),
                .float => |o| std.math.order(@as(f64, @floatFromInt(i)), o),
                else => null,
            },
            .float => |f| switch (other) {
                .float => |o| std.math.order(f, o),
                .integer => |o| std.math.order(f, @as(f64, @floatFromInt(o))),
                else => null,
            },
            .string => |s| switch (other) {
                .string => |o| std.mem.order(u8, s, o),
                else => null,
            },
            else => null,
        };
    }

    /// Check if value contains substring.
    pub fn containsSubstring(self: MetadataValue, substr: []const u8) bool {
        return switch (self) {
            .string => |s| std.mem.indexOf(u8, s, substr) != null,
            else => false,
        };
    }

    /// Check if value is in list.
    pub fn isInList(self: MetadataValue, list: MetadataValue) bool {
        return switch (list) {
            .string_list => |sl| switch (self) {
                .string => |s| blk: {
                    for (sl) |item| {
                        if (std.mem.eql(u8, s, item)) break :blk true;
                    }
                    break :blk false;
                },
                else => false,
            },
            .integer_list => |il| switch (self) {
                .integer => |i| blk: {
                    for (il) |item| {
                        if (i == item) break :blk true;
                    }
                    break :blk false;
                },
                else => false,
            },
            else => false,
        };
    }
};

/// Single filter condition.
pub const FilterCondition = struct {
    field: []const u8,
    operator: FilterOperator,
    value: MetadataValue,
    /// Secondary value for range operators.
    secondary_value: ?MetadataValue = null,

    /// Evaluate condition against metadata.
    pub fn evaluate(self: FilterCondition, metadata: *const std.StringHashMapUnmanaged(MetadataValue)) bool {
        const field_value = metadata.get(self.field);

        return switch (self.operator) {
            .exists => field_value != null,
            .not_exists => field_value == null,
            .eq => if (field_value) |v| v.eql(self.value) else false,
            .ne => if (field_value) |v| !v.eql(self.value) else true,
            .gt => if (field_value) |v|
                if (v.compare(self.value)) |ord| ord == .gt else false
            else
                false,
            .gte => if (field_value) |v|
                if (v.compare(self.value)) |ord| ord != .lt else false
            else
                false,
            .lt => if (field_value) |v|
                if (v.compare(self.value)) |ord| ord == .lt else false
            else
                false,
            .lte => if (field_value) |v|
                if (v.compare(self.value)) |ord| ord != .gt else false
            else
                false,
            .contains => if (field_value) |v| switch (self.value) {
                .string => |s| v.containsSubstring(s),
                else => false,
            } else false,
            .starts_with => if (field_value) |v| switch (v) {
                .string => |s| switch (self.value) {
                    .string => |prefix| std.mem.startsWith(u8, s, prefix),
                    else => false,
                },
                else => false,
            } else false,
            .ends_with => if (field_value) |v| switch (v) {
                .string => |s| switch (self.value) {
                    .string => |suffix| std.mem.endsWith(u8, s, suffix),
                    else => false,
                },
                else => false,
            } else false,
            .in_list => if (field_value) |v| v.isInList(self.value) else false,
            .not_in_list => if (field_value) |v| !v.isInList(self.value) else true,
            .between => if (field_value) |v| blk: {
                const lower = v.compare(self.value);
                const upper = if (self.secondary_value) |sv| v.compare(sv) else null;
                break :blk if (lower != null and upper != null)
                    lower.? != .lt and upper.? != .gt
                else
                    false;
            } else false,
            .regex => false, // Regex matching would require additional library
        };
    }
};

/// Logical operators for combining conditions.
pub const LogicalOperator = enum {
    and_op,
    or_op,
    not_op,
};

/// Composite filter expression.
pub const FilterExpression = union(enum) {
    condition: FilterCondition,
    logical: struct {
        operator: LogicalOperator,
        operands: []const FilterExpression,
    },

    /// Evaluate expression against metadata.
    pub fn evaluate(self: FilterExpression, metadata: *const std.StringHashMapUnmanaged(MetadataValue)) bool {
        return switch (self) {
            .condition => |c| c.evaluate(metadata),
            .logical => |l| switch (l.operator) {
                .and_op => blk: {
                    for (l.operands) |op| {
                        if (!op.evaluate(metadata)) break :blk false;
                    }
                    break :blk true;
                },
                .or_op => blk: {
                    for (l.operands) |op| {
                        if (op.evaluate(metadata)) break :blk true;
                    }
                    break :blk false;
                },
                .not_op => if (l.operands.len > 0)
                    !l.operands[0].evaluate(metadata)
                else
                    true,
            },
        };
    }
};

/// Filter builder for fluent API.
pub const FilterBuilder = struct {
    allocator: std.mem.Allocator,
    conditions: std.ArrayListUnmanaged(FilterExpression),

    pub fn init(allocator: std.mem.Allocator) FilterBuilder {
        return .{
            .allocator = allocator,
            .conditions = .{},
        };
    }

    pub fn deinit(self: *FilterBuilder) void {
        self.conditions.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add equality condition.
    pub fn whereEq(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .eq,
                .value = value,
            },
        });
        return self;
    }

    /// Add not equal condition.
    pub fn whereNe(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .ne,
                .value = value,
            },
        });
        return self;
    }

    /// Add greater than condition.
    pub fn whereGt(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .gt,
                .value = value,
            },
        });
        return self;
    }

    /// Add greater than or equal condition.
    pub fn whereGte(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .gte,
                .value = value,
            },
        });
        return self;
    }

    /// Add less than condition.
    pub fn whereLt(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .lt,
                .value = value,
            },
        });
        return self;
    }

    /// Add less than or equal condition.
    pub fn whereLte(self: *FilterBuilder, field: []const u8, value: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .lte,
                .value = value,
            },
        });
        return self;
    }

    /// Add contains condition.
    pub fn whereContains(self: *FilterBuilder, field: []const u8, substring: []const u8) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .contains,
                .value = .{ .string = substring },
            },
        });
        return self;
    }

    /// Add in list condition.
    pub fn whereIn(self: *FilterBuilder, field: []const u8, list: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .in_list,
                .value = list,
            },
        });
        return self;
    }

    /// Add exists condition.
    pub fn whereExists(self: *FilterBuilder, field: []const u8) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .exists,
                .value = .null_value,
            },
        });
        return self;
    }

    /// Add between condition.
    pub fn whereBetween(self: *FilterBuilder, field: []const u8, lower: MetadataValue, upper: MetadataValue) !*FilterBuilder {
        try self.conditions.append(self.allocator, .{
            .condition = .{
                .field = field,
                .operator = .between,
                .value = lower,
                .secondary_value = upper,
            },
        });
        return self;
    }

    /// Build AND expression from all conditions.
    pub fn buildAnd(self: *FilterBuilder) !FilterExpression {
        if (self.conditions.items.len == 0) {
            return .{ .condition = .{
                .field = "",
                .operator = .exists,
                .value = .null_value,
            } };
        }
        if (self.conditions.items.len == 1) {
            return self.conditions.items[0];
        }
        return .{ .logical = .{
            .operator = .and_op,
            .operands = try self.allocator.dupe(FilterExpression, self.conditions.items),
        } };
    }

    /// Build OR expression from all conditions.
    pub fn buildOr(self: *FilterBuilder) !FilterExpression {
        if (self.conditions.items.len == 0) {
            return .{ .condition = .{
                .field = "",
                .operator = .not_exists,
                .value = .null_value,
            } };
        }
        if (self.conditions.items.len == 1) {
            return self.conditions.items[0];
        }
        return .{ .logical = .{
            .operator = .or_op,
            .operands = try self.allocator.dupe(FilterExpression, self.conditions.items),
        } };
    }
};

/// Filtered search result.
pub const FilteredResult = struct {
    id: u64,
    score: f32,
    metadata: std.StringHashMapUnmanaged(MetadataValue),

    pub fn lessThan(_: void, a: FilteredResult, b: FilteredResult) bool {
        return a.score > b.score; // Higher score first
    }
};

/// Metadata store for indexed documents.
pub const MetadataStore = struct {
    allocator: std.mem.Allocator,
    documents: std.AutoHashMapUnmanaged(u64, std.StringHashMapUnmanaged(MetadataValue)),
    field_index: std.StringHashMapUnmanaged(std.AutoHashMapUnmanaged(u64, void)),

    pub fn init(allocator: std.mem.Allocator) MetadataStore {
        return .{
            .allocator = allocator,
            .documents = .{},
            .field_index = .{},
        };
    }

    pub fn deinit(self: *MetadataStore) void {
        var doc_iter = self.documents.iterator();
        while (doc_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.documents.deinit(self.allocator);

        var field_iter = self.field_index.iterator();
        while (field_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.field_index.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add metadata for a document.
    pub fn put(self: *MetadataStore, doc_id: u64, metadata: std.StringHashMapUnmanaged(MetadataValue)) !void {
        // Remove old metadata if exists
        if (self.documents.get(doc_id)) |*old| {
            var iter = old.iterator();
            while (iter.next()) |entry| {
                if (self.field_index.getPtr(entry.key_ptr.*)) |doc_set| {
                    _ = doc_set.remove(doc_id);
                }
            }
            old.deinit(self.allocator);
        }

        // Store new metadata
        try self.documents.put(self.allocator, doc_id, metadata);

        // Update field index
        var iter = metadata.iterator();
        while (iter.next()) |entry| {
            const field_entry = try self.field_index.getOrPut(self.allocator, entry.key_ptr.*);
            if (!field_entry.found_existing) {
                field_entry.key_ptr.* = try self.allocator.dupe(u8, entry.key_ptr.*);
                field_entry.value_ptr.* = .{};
            }
            try field_entry.value_ptr.put(self.allocator, doc_id, {});
        }
    }

    /// Get metadata for a document.
    pub fn get(self: *const MetadataStore, doc_id: u64) ?*const std.StringHashMapUnmanaged(MetadataValue) {
        return self.documents.getPtr(doc_id);
    }

    /// Remove metadata for a document.
    pub fn remove(self: *MetadataStore, doc_id: u64) void {
        if (self.documents.fetchRemove(doc_id)) |kv| {
            var meta = kv.value;
            var iter = meta.iterator();
            while (iter.next()) |entry| {
                if (self.field_index.getPtr(entry.key_ptr.*)) |doc_set| {
                    _ = doc_set.remove(doc_id);
                }
            }
            meta.deinit(self.allocator);
        }
    }

    /// Get all document IDs that have a specific field.
    pub fn getDocumentsWithField(self: *const MetadataStore, field: []const u8) ?*const std.AutoHashMapUnmanaged(u64, void) {
        return self.field_index.getPtr(field);
    }

    /// Count documents matching filter.
    pub fn countMatching(self: *const MetadataStore, filter: FilterExpression) usize {
        var count: usize = 0;
        var iter = self.documents.iterator();
        while (iter.next()) |entry| {
            if (filter.evaluate(entry.value_ptr)) {
                count += 1;
            }
        }
        return count;
    }

    /// Get all matching document IDs.
    pub fn getMatchingIds(self: *MetadataStore, filter: FilterExpression) ![]u64 {
        var ids = std.ArrayListUnmanaged(u64){};
        errdefer ids.deinit(self.allocator);

        var iter = self.documents.iterator();
        while (iter.next()) |entry| {
            if (filter.evaluate(entry.value_ptr)) {
                try ids.append(self.allocator, entry.key_ptr.*);
            }
        }

        return ids.toOwnedSlice(self.allocator);
    }
};

/// Filtered search configuration.
pub const FilteredSearchConfig = struct {
    /// Apply filter before vector search (reduces search space).
    pre_filter: bool = true,
    /// Apply filter after vector search (filters results).
    post_filter: bool = false,
    /// Maximum candidates to consider in pre-filter.
    max_candidates: usize = 10000,
    /// Expand search if too few results.
    expand_on_insufficient: bool = true,
    /// Expansion factor for insufficient results.
    expansion_factor: f32 = 2.0,
};

/// Filtered vector search engine.
pub const FilteredSearch = struct {
    allocator: std.mem.Allocator,
    metadata_store: MetadataStore,
    config: FilteredSearchConfig,

    pub fn init(allocator: std.mem.Allocator, config: FilteredSearchConfig) FilteredSearch {
        return .{
            .allocator = allocator,
            .metadata_store = MetadataStore.init(allocator),
            .config = config,
        };
    }

    pub fn deinit(self: *FilteredSearch) void {
        self.metadata_store.deinit();
        self.* = undefined;
    }

    /// Index document metadata.
    pub fn indexMetadata(self: *FilteredSearch, doc_id: u64, metadata: std.StringHashMapUnmanaged(MetadataValue)) !void {
        try self.metadata_store.put(doc_id, metadata);
    }

    /// Remove document metadata.
    pub fn removeMetadata(self: *FilteredSearch, doc_id: u64) void {
        self.metadata_store.remove(doc_id);
    }

    /// Get candidate document IDs matching filter (for pre-filter strategy).
    pub fn getFilteredCandidates(self: *FilteredSearch, filter: FilterExpression) ![]u64 {
        return self.metadata_store.getMatchingIds(filter);
    }

    /// Apply post-filter to search results.
    pub fn applyPostFilter(
        self: *const FilteredSearch,
        results: []const FilteredResult,
        filter: FilterExpression,
    ) ![]FilteredResult {
        var filtered = std.ArrayListUnmanaged(FilteredResult){};
        errdefer filtered.deinit(self.allocator);

        for (results) |result| {
            if (filter.evaluate(&result.metadata)) {
                try filtered.append(self.allocator, result);
            }
        }

        return filtered.toOwnedSlice(self.allocator);
    }

    /// Search with metadata filter.
    pub fn search(
        self: *FilteredSearch,
        scores: std.AutoHashMapUnmanaged(u64, f32),
        filter: FilterExpression,
        top_k: usize,
    ) ![]FilteredResult {
        var results = std.ArrayListUnmanaged(FilteredResult){};
        errdefer results.deinit(self.allocator);

        // Filter and collect results
        var iter = scores.iterator();
        while (iter.next()) |entry| {
            const doc_id = entry.key_ptr.*;
            const score = entry.value_ptr.*;

            if (self.metadata_store.get(doc_id)) |metadata| {
                if (filter.evaluate(metadata)) {
                    try results.append(self.allocator, .{
                        .id = doc_id,
                        .score = score,
                        .metadata = metadata.*,
                    });
                }
            }
        }

        // Sort by score
        std.mem.sort(FilteredResult, results.items, {}, FilteredResult.lessThan);

        // Return top_k
        const result_count = @min(top_k, results.items.len);
        const final_results = try self.allocator.alloc(FilteredResult, result_count);
        @memcpy(final_results, results.items[0..result_count]);
        results.deinit(self.allocator);

        return final_results;
    }

    /// Get filter statistics.
    pub fn getFilterStats(self: *const FilteredSearch, filter: FilterExpression) FilterStats {
        const matching = self.metadata_store.countMatching(filter);
        const total = self.metadata_store.documents.count();

        return .{
            .total_documents = total,
            .matching_documents = matching,
            .selectivity = if (total > 0)
                @as(f32, @floatFromInt(matching)) / @as(f32, @floatFromInt(total))
            else
                0,
        };
    }
};

/// Filter statistics.
pub const FilterStats = struct {
    total_documents: usize,
    matching_documents: usize,
    selectivity: f32,
};

test "filter operator eq" {
    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    defer metadata.deinit(std.testing.allocator);

    try metadata.put(std.testing.allocator, "name", .{ .string = "test" });
    try metadata.put(std.testing.allocator, "count", .{ .integer = 42 });

    const eq_cond = FilterCondition{
        .field = "name",
        .operator = .eq,
        .value = .{ .string = "test" },
    };
    try std.testing.expect(eq_cond.evaluate(&metadata));

    const ne_cond = FilterCondition{
        .field = "name",
        .operator = .eq,
        .value = .{ .string = "other" },
    };
    try std.testing.expect(!ne_cond.evaluate(&metadata));
}

test "filter operator comparison" {
    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    defer metadata.deinit(std.testing.allocator);

    try metadata.put(std.testing.allocator, "age", .{ .integer = 25 });

    const gt_cond = FilterCondition{
        .field = "age",
        .operator = .gt,
        .value = .{ .integer = 20 },
    };
    try std.testing.expect(gt_cond.evaluate(&metadata));

    const lt_cond = FilterCondition{
        .field = "age",
        .operator = .lt,
        .value = .{ .integer = 20 },
    };
    try std.testing.expect(!lt_cond.evaluate(&metadata));
}

test "filter logical and" {
    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    defer metadata.deinit(std.testing.allocator);

    try metadata.put(std.testing.allocator, "type", .{ .string = "document" });
    try metadata.put(std.testing.allocator, "year", .{ .integer = 2024 });

    const conditions = [_]FilterExpression{
        .{ .condition = .{
            .field = "type",
            .operator = .eq,
            .value = .{ .string = "document" },
        } },
        .{ .condition = .{
            .field = "year",
            .operator = .gte,
            .value = .{ .integer = 2020 },
        } },
    };

    const and_expr = FilterExpression{ .logical = .{
        .operator = .and_op,
        .operands = &conditions,
    } };

    try std.testing.expect(and_expr.evaluate(&metadata));
}

test "filter builder" {
    const allocator = std.testing.allocator;
    var builder = FilterBuilder.init(allocator);
    defer builder.deinit();

    _ = try builder.whereEq("status", .{ .string = "active" });
    _ = try builder.whereGte("score", .{ .float = 0.5 });

    const expr = try builder.buildAnd();

    var metadata = std.StringHashMapUnmanaged(MetadataValue){};
    defer metadata.deinit(allocator);

    try metadata.put(allocator, "status", .{ .string = "active" });
    try metadata.put(allocator, "score", .{ .float = 0.8 });

    try std.testing.expect(expr.evaluate(&metadata));
}

test "metadata store" {
    const allocator = std.testing.allocator;
    var store = MetadataStore.init(allocator);
    defer store.deinit();

    var meta1 = std.StringHashMapUnmanaged(MetadataValue){};
    try meta1.put(allocator, "category", .{ .string = "tech" });
    try store.put(1, meta1);

    var meta2 = std.StringHashMapUnmanaged(MetadataValue){};
    try meta2.put(allocator, "category", .{ .string = "science" });
    try store.put(2, meta2);

    const filter = FilterExpression{ .condition = .{
        .field = "category",
        .operator = .eq,
        .value = .{ .string = "tech" },
    } };

    const matching = try store.getMatchingIds(filter);
    defer allocator.free(matching);

    try std.testing.expectEqual(@as(usize, 1), matching.len);
    try std.testing.expectEqual(@as(u64, 1), matching[0]);
}
