//! Local dry-run browser orchestration plan (Approach-1 leaf).
//! Records a structured task spec only — no embedded headless browser.
//! Real automation is delegated to an external MCP Playwright peer.

const std = @import("std");

pub const BrowserOrchestrationPlan = struct {
    output: []u8,
    requires_review: bool,
    execute_requested: bool,

    pub fn deinit(self: BrowserOrchestrationPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.output);
    }
};

pub fn planBrowserOrchestration(
    allocator: std.mem.Allocator,
    task: []const u8,
    url: ?[]const u8,
    execute_confirmed: bool,
) !BrowserOrchestrationPlan {
    if (task.len == 0) return error.InvalidAgentConfig;
    const mode: []const u8 = if (execute_confirmed) "execute-requested" else "dry-run";
    const url_line = if (url) |u| try std.fmt.allocPrint(allocator, "target_url={s}\n", .{u}) else try allocator.dupe(u8, "");
    defer allocator.free(url_line);

    const output = try std.fmt.allocPrint(
        allocator,
        "orchestration=browser-local\nmode={s}\nreview_required=true\nembedded_browser=false\ndelegation_hint=external-mcp-playwright\npolicy=loopback-credentials-user-consent\ntool_hints_enforced=false\n{s}task={s}\nsteps:\n  1. record structured browser task spec locally\n  2. record tool_hints in agent output (constitution does not consume hints today)\n  3. recommended next step: delegate to external MCP Playwright peer (not performed by ABI)\n  4. {s}\n",
        .{
            mode,
            url_line,
            task,
            if (execute_confirmed)
                "execute path requires explicit --confirm and an external browser MCP; ABI does not launch a headless browser in-process"
            else
                "dry-run only — no navigation or credential access",
        },
    );
    return .{
        .output = output,
        .requires_review = true,
        .execute_requested = execute_confirmed,
    };
}

test "browser orchestration stays dry-run honest" {
    var plan = try planBrowserOrchestration(std.testing.allocator, "open docs", "https://example.com", false);
    defer plan.deinit(std.testing.allocator);
    try std.testing.expect(plan.requires_review);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "embedded_browser=false") != null);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "dry-run only") != null);
}

test "browser orchestration execute-confirmed stays honest" {
    var plan = try planBrowserOrchestration(std.testing.allocator, "navigate", null, true);
    defer plan.deinit(std.testing.allocator);
    try std.testing.expect(plan.execute_requested);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "execute-requested") != null);
    try std.testing.expect(std.mem.indexOf(u8, plan.output, "does not launch a headless browser") != null);
}

test {
    std.testing.refAllDecls(@This());
}
