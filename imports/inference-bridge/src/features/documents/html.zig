//! Native HTML and DOM Parser
//!
//! Exposes a zero-dependency DOM tree constructor that isolates semantic
//! meaning (paragraphs, headers, links) from scripts, CSS, and layout garbage
//! for deep internet research.

const std = @import("std");

pub const DomNode = struct {
    tag: []const u8,
    text_content: ?[]const u8,
    children: std.ArrayListUnmanaged(DomNode),

    pub fn deinit(self: *DomNode, allocator: std.mem.Allocator) void {
        for (self.children.items) |*child| {
            child.deinit(allocator);
        }
        self.children.deinit(allocator);
    }
};

pub const HtmlParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) HtmlParser {
        return .{ .allocator = allocator };
    }

    pub fn parse(self: *HtmlParser, raw_html: []const u8) !DomNode {
        std.log.info("[HTML Parser] Natively tokenizing DOM ({d} bytes)...", .{raw_html.len});

        var root = DomNode{
            .tag = try self.allocator.dupe(u8, "root"),
            .text_content = null,
            .children = .empty,
        };

        var node_stack = std.ArrayListUnmanaged(*DomNode).empty;
        defer node_stack.deinit(self.allocator);
        try node_stack.append(self.allocator, &root);

        var i: usize = 0;
        var in_tag = false;
        var tag_start: usize = 0;
        var text_start: usize = 0;

        var ignore_until_close: ?[]const u8 = null;

        while (i < raw_html.len) {
            const c = raw_html[i];

            if (!in_tag and c == '<') {
                // Capture preceding text
                if (i > text_start and ignore_until_close == null) {
                    const text = std.mem.trim(u8, raw_html[text_start..i], " \t\r\n");
                    if (text.len > 0) {
                        var current_parent = node_stack.items[node_stack.items.len - 1];
                        const text_node = DomNode{
                            .tag = try self.allocator.dupe(u8, "#text"),
                            .text_content = try self.allocator.dupe(u8, text),
                            .children = .empty,
                        };
                        try current_parent.children.append(self.allocator, text_node);
                    }
                }
                in_tag = true;
                tag_start = i + 1;
            } else if (in_tag and c == '>') {
                const tag_content = std.mem.trim(u8, raw_html[tag_start..i], " \t\r\n/");
                var iter = std.mem.splitScalar(u8, tag_content, ' ');
                const tag_name_raw = iter.first();

                var tag_name = try self.allocator.dupe(u8, tag_name_raw);
                for (tag_name) |*char| char.* = std.ascii.toLower(char.*);
                defer self.allocator.free(tag_name);

                const is_closing = tag_name.len > 0 and tag_name[0] == '/';
                const base_tag = if (is_closing) tag_name[1..] else tag_name;

                if (ignore_until_close) |ignored| {
                    if (is_closing and std.mem.eql(u8, base_tag, ignored)) {
                        ignore_until_close = null;
                    }
                } else {
                    if (std.mem.eql(u8, base_tag, "script") or std.mem.eql(u8, base_tag, "style")) {
                        if (!is_closing) {
                            ignore_until_close = try self.allocator.dupe(u8, base_tag);
                        }
                    } else if (is_closing) {
                        // Pop from stack if it matches (very basic tree building)
                        if (node_stack.items.len > 1) {
                            const top_tag = node_stack.items[node_stack.items.len - 1].tag;
                            if (std.mem.eql(u8, top_tag, base_tag)) {
                                _ = node_stack.pop();
                            }
                        }
                    } else {
                        // Push new node
                        var current_parent = node_stack.items[node_stack.items.len - 1];
                        const new_node = DomNode{
                            .tag = try self.allocator.dupe(u8, base_tag),
                            .text_content = null,
                            .children = .empty,
                        };

                        try current_parent.children.append(self.allocator, new_node);

                        // Self-closing tags heuristics
                        if (!std.mem.eql(u8, base_tag, "br") and !std.mem.eql(u8, base_tag, "img") and !std.mem.eql(u8, base_tag, "hr") and !std.mem.eql(u8, base_tag, "meta") and !std.mem.eql(u8, base_tag, "link")) {
                            const ref = &current_parent.children.items[current_parent.children.items.len - 1];
                            try node_stack.append(self.allocator, ref);
                        }
                    }
                }

                in_tag = false;
                text_start = i + 1;
            }
            i += 1;
        }

        if (ignore_until_close) |ign| self.allocator.free(ign);

        return root;
    }
};

test {
    std.testing.refAllDecls(@This());
}
