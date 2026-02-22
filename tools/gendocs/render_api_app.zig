const std = @import("std");
const model = @import("model.zig");
const site_map = @import("site_map.zig");

pub fn render(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    features: []const model.FeatureDoc,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    try renderJsonData(allocator, modules, commands, features, roadmap_entries, plan_entries, outputs);
    try model.pushOutput(allocator, outputs, "docs/api-app/index.html", html_template);
    try model.pushOutput(allocator, outputs, "docs/api-app/styles.css", css_template);
    try model.pushOutput(allocator, outputs, "docs/api-app/app.js", js_template);
}

fn renderJsonData(
    allocator: std.mem.Allocator,
    modules: []const model.ModuleDoc,
    commands: []const model.CliCommand,
    features: []const model.FeatureDoc,
    roadmap_entries: []const model.RoadmapDocEntry,
    plan_entries: []const model.PlanDocEntry,
    outputs: *std.ArrayListUnmanaged(model.OutputFile),
) !void {
    const JsonSymbol = struct {
        anchor: []const u8,
        signature: []const u8,
        doc: []const u8,
        kind: []const u8,
        line: usize,
    };
    const JsonModule = struct {
        name: []const u8,
        path: []const u8,
        description: []const u8,
        category: []const u8,
        build_flag: []const u8,
        symbols: []JsonSymbol,
    };

    var module_json = try allocator.alloc(JsonModule, modules.len);
    defer {
        for (module_json) |item| allocator.free(item.symbols);
        allocator.free(module_json);
    }

    for (modules, 0..) |mod, idx| {
        const symbols = try allocator.alloc(JsonSymbol, mod.symbols.len);
        for (mod.symbols, 0..) |symbol, sidx| {
            symbols[sidx] = .{
                .anchor = symbol.anchor,
                .signature = symbol.signature,
                .doc = symbol.doc,
                .kind = symbol.kind.badge(),
                .line = symbol.line,
            };
        }

        module_json[idx] = .{
            .name = mod.name,
            .path = mod.path,
            .description = mod.description,
            .category = mod.category.name(),
            .build_flag = mod.build_flag,
            .symbols = symbols,
        };
    }

    const JsonCommand = struct {
        name: []const u8,
        description: []const u8,
        aliases: []const []const u8,
        subcommands: []const []const u8,
    };

    var command_json = try allocator.alloc(JsonCommand, commands.len);
    defer allocator.free(command_json);
    for (commands, 0..) |command, idx| {
        command_json[idx] = .{
            .name = command.name,
            .description = command.description,
            .aliases = command.aliases,
            .subcommands = command.subcommands,
        };
    }

    const JsonGuide = struct {
        slug: []const u8,
        title: []const u8,
        section: []const u8,
        permalink: []const u8,
        description: []const u8,
    };

    var guides_json = try allocator.alloc(JsonGuide, site_map.guides.len);
    defer allocator.free(guides_json);
    for (site_map.guides, 0..) |guide, idx| {
        guides_json[idx] = .{
            .slug = guide.slug,
            .title = guide.title,
            .section = guide.section,
            .permalink = guide.permalink,
            .description = guide.description,
        };
    }

    const modules_json_text = try stringifyAlloc(allocator, module_json);
    defer allocator.free(modules_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/modules.json", modules_json_text);

    const commands_json_text = try stringifyAlloc(allocator, command_json);
    defer allocator.free(commands_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/commands.json", commands_json_text);

    const JsonFeature = struct {
        name: []const u8,
        description: []const u8,
        compile_flag: []const u8,
        parent: []const u8,
        real_module_path: []const u8,
        stub_module_path: []const u8,
    };

    var features_json = try allocator.alloc(JsonFeature, features.len);
    defer allocator.free(features_json);
    for (features, 0..) |feat, idx| {
        features_json[idx] = .{
            .name = feat.name,
            .description = feat.description,
            .compile_flag = feat.compile_flag,
            .parent = feat.parent,
            .real_module_path = feat.real_module_path,
            .stub_module_path = feat.stub_module_path,
        };
    }

    const features_json_text = try stringifyAlloc(allocator, features_json);
    defer allocator.free(features_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/features.json", features_json_text);

    const guides_json_text = try stringifyAlloc(allocator, guides_json);
    defer allocator.free(guides_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/guides.json", guides_json_text);

    const JsonPlan = struct {
        slug: []const u8,
        title: []const u8,
        status: []const u8,
        owner: []const u8,
        scope: []const u8,
        gate_commands: []const []const u8,
    };

    var plans_json = try allocator.alloc(JsonPlan, plan_entries.len);
    defer allocator.free(plans_json);
    for (plan_entries, 0..) |plan, idx| {
        plans_json[idx] = .{
            .slug = plan.slug,
            .title = plan.title,
            .status = plan.status,
            .owner = plan.owner,
            .scope = plan.scope,
            .gate_commands = plan.gate_commands,
        };
    }

    const plans_json_text = try stringifyAlloc(allocator, plans_json);
    defer allocator.free(plans_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/plans.json", plans_json_text);

    const JsonRoadmap = struct {
        id: []const u8,
        title: []const u8,
        summary: []const u8,
        track: []const u8,
        horizon: []const u8,
        status: []const u8,
        owner: []const u8,
        validation_gate: []const u8,
        plan_slug: []const u8,
        plan_title: []const u8,
    };

    var roadmap_json = try allocator.alloc(JsonRoadmap, roadmap_entries.len);
    defer allocator.free(roadmap_json);
    for (roadmap_entries, 0..) |entry, idx| {
        roadmap_json[idx] = .{
            .id = entry.id,
            .title = entry.title,
            .summary = entry.summary,
            .track = entry.track,
            .horizon = entry.horizon,
            .status = entry.status,
            .owner = entry.owner,
            .validation_gate = entry.validation_gate,
            .plan_slug = entry.plan_slug,
            .plan_title = entry.plan_title,
        };
    }

    const roadmap_json_text = try stringifyAlloc(allocator, roadmap_json);
    defer allocator.free(roadmap_json_text);
    try model.pushOutput(allocator, outputs, "docs/api-app/data/roadmap.json", roadmap_json_text);
}

fn stringifyAlloc(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();

    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, &json_writer.writer);
    try json_writer.writer.writeByte('\n');
    return json_writer.toOwnedSlice();
}

const html_template =
    \\<!doctype html>
    \\<html lang="en">
    \\<head>
    \\  <meta charset="utf-8" />
    \\  <meta name="viewport" content="width=device-width, initial-scale=1" />
    \\  <title>ABI API App</title>
    \\  <link rel="preconnect" href="https://fonts.googleapis.com" />
    \\  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    \\  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet" />
    \\  <link rel="stylesheet" href="./styles.css" />
    \\</head>
    \\<body>
    \\  <div class="bg-orb bg-orb-a"></div>
    \\  <div class="bg-orb bg-orb-b"></div>
    \\  <header class="topbar">
    \\    <div class="brand">ABI Docs App</div>
    \\    <nav>
    \\      <a href="../api/">Markdown API</a>
    \\      <a href="../">Docs Home</a>
    \\      <a href="../api-overview/">API Overview</a>
    \\    </nav>
    \\  </header>
    \\
    \\  <main class="layout">
    \\    <aside class="sidebar card">
    \\      <h2>Explore</h2>
    \\      <label>Search</label>
    \\      <input id="searchInput" type="text" placeholder="module, symbol, command, plan" />
    \\      <label>Category</label>
    \\      <select id="categoryFilter">
    \\        <option value="">All categories</option>
    \\      </select>
    \\      <label>Type</label>
    \\      <select id="typeFilter">
    \\        <option value="all">Everything</option>
    \\        <option value="modules">Modules</option>
    \\        <option value="symbols">Symbols</option>
    \\        <option value="commands">Commands</option>
    \\        <option value="guides">Guides</option>
    \\        <option value="plans">Plans</option>
    \\        <option value="roadmap">Roadmap</option>
    \\      </select>
    \\      <p class="hint">WASM-assisted ranking loads automatically when available.</p>
    \\    </aside>
    \\
    \\    <section class="content">
    \\      <div class="card">
    \\        <h1>Generated API Explorer</h1>
    \\        <p>Fast navigation for ABI modules, symbols, CLI commands, plans, and roadmap tracks.</p>
    \\        <div id="stats" class="stats"></div>
    \\      </div>
    \\      <div class="card split">
    \\        <div>
    \\          <h3>Results</h3>
    \\          <ul id="results" class="results"></ul>
    \\        </div>
    \\        <div>
    \\          <h3>Details</h3>
    \\          <div id="details" class="details">Select an item to inspect metadata.</div>
    \\        </div>
    \\      </div>
    \\    </section>
    \\  </main>
    \\
    \\  <script type="module" src="./app.js"></script>
    \\</body>
    \\</html>
;

const css_template =
    \\:root {
    \\  --bg: #0b1220;
    \\  --bg-soft: #111b2f;
    \\  --panel: rgba(16, 24, 40, 0.8);
    \\  --line: #2e3f5a;
    \\  --text: #f2f7ff;
    \\  --muted: #a9bbd3;
    \\  --accent: #32d7a3;
    \\  --accent-2: #4dc3ff;
    \\}
    \\* { box-sizing: border-box; }
    \\body {
    \\  margin: 0;
    \\  font-family: "Space Grotesk", system-ui, sans-serif;
    \\  color: var(--text);
    \\  background: radial-gradient(circle at 20% 15%, #1e2b48 0%, var(--bg) 45%), linear-gradient(150deg, #070b12 0%, #0f1a2f 100%);
    \\  min-height: 100vh;
    \\}
    \\a { color: var(--accent-2); text-decoration: none; }
    \\.topbar {
    \\  position: sticky;
    \\  top: 0;
    \\  z-index: 10;
    \\  display: flex;
    \\  justify-content: space-between;
    \\  align-items: center;
    \\  padding: 14px 22px;
    \\  background: rgba(8, 14, 24, 0.82);
    \\  backdrop-filter: blur(10px);
    \\  border-bottom: 1px solid var(--line);
    \\}
    \\.brand { font-weight: 700; letter-spacing: 0.03em; }
    \\.topbar nav { display: flex; gap: 16px; font-size: 0.95rem; }
    \\.layout {
    \\  max-width: 1240px;
    \\  margin: 26px auto;
    \\  padding: 0 18px;
    \\  display: grid;
    \\  grid-template-columns: 290px 1fr;
    \\  gap: 16px;
    \\}
    \\.card {
    \\  border: 1px solid var(--line);
    \\  background: var(--panel);
    \\  border-radius: 16px;
    \\  padding: 18px;
    \\  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    \\}
    \\.sidebar label {
    \\  display: block;
    \\  margin-top: 12px;
    \\  margin-bottom: 4px;
    \\  color: var(--muted);
    \\  font-size: 0.85rem;
    \\}
    \\input, select {
    \\  width: 100%;
    \\  border: 1px solid var(--line);
    \\  border-radius: 10px;
    \\  padding: 10px 12px;
    \\  background: #0f1728;
    \\  color: var(--text);
    \\  font-family: "IBM Plex Mono", monospace;
    \\}
    \\.hint { color: var(--muted); font-size: 0.82rem; margin-top: 12px; }
    \\.split { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    \\.results { list-style: none; margin: 0; padding: 0; max-height: 62vh; overflow: auto; }
    \\.results li {
    \\  border: 1px solid var(--line);
    \\  border-radius: 10px;
    \\  padding: 10px;
    \\  margin-bottom: 8px;
    \\  cursor: pointer;
    \\  transition: transform 140ms ease, border-color 140ms ease;
    \\}
    \\.results li:hover, .results li.active { border-color: var(--accent); transform: translateY(-1px); }
    \\.result-meta { color: var(--muted); font-size: 0.82rem; }
    \\.details {
    \\  border: 1px solid var(--line);
    \\  border-radius: 12px;
    \\  min-height: 280px;
    \\  padding: 14px;
    \\  font-family: "IBM Plex Mono", monospace;
    \\  white-space: pre-wrap;
    \\  line-height: 1.42;
    \\}
    \\.stats {
    \\  display: flex;
    \\  flex-wrap: wrap;
    \\  gap: 8px;
    \\  margin-top: 10px;
    \\}
    \\.chip {
    \\  display: inline-flex;
    \\  align-items: center;
    \\  border: 1px solid var(--line);
    \\  border-radius: 999px;
    \\  padding: 4px 10px;
    \\  font-size: 0.78rem;
    \\  color: var(--muted);
    \\}
    \\.bg-orb {
    \\  position: fixed;
    \\  border-radius: 50%;
    \\  filter: blur(75px);
    \\  pointer-events: none;
    \\  opacity: 0.32;
    \\}
    \\.bg-orb-a { width: 280px; height: 280px; background: #13b8ff; top: -50px; left: -70px; }
    \\.bg-orb-b { width: 320px; height: 320px; background: #00d18f; right: -80px; bottom: -90px; }
    \\@media (max-width: 980px) {
    \\  .layout { grid-template-columns: 1fr; }
    \\  .split { grid-template-columns: 1fr; }
    \\}
;

const js_template =
    \\const loadJson = async (path) => {
    \\  const res = await fetch(path, { cache: "no-store" });
    \\  if (!res.ok) throw new Error(`Failed to load ${path}`);
    \\  return await res.json();
    \\};
    \\
    \\const utf8 = new TextEncoder();
    \\let wasmApi = null;
    \\let rankerLabel = "JS ranker fallback";
    \\
    \\async function tryLoadRanker(path, label) {
    \\  try {
    \\    const bytes = await fetch(path).then((r) => {
    \\      if (!r.ok) throw new Error("missing");
    \\      return r.arrayBuffer();
    \\    });
    \\    const mod = await WebAssembly.instantiate(bytes, {});
    \\    const exp = mod.instance.exports;
    \\    if (exp.memory && exp.alloc && exp.reset_alloc && exp.score_query) {
    \\      wasmApi = exp;
    \\      rankerLabel = label;
    \\      return true;
    \\    }
    \\  } catch (_) {}
    \\  return false;
    \\}
    \\
    \\async function loadWasmRanker() {
    \\  // Prefer component output when present and instantiable, then fallback to plain wasm.
    \\  if (await tryLoadRanker("./data/docs_engine.component.wasm", "WASM ranker (component)")) return;
    \\  await tryLoadRanker("./data/docs_engine.wasm", "WASM ranker (plain)");
    \\}
    \\
    \\function jsScore(query, text) {
    \\  if (!query) return 1;
    \\  const q = query.toLowerCase();
    \\  const t = text.toLowerCase();
    \\  if (t === q) return 1200;
    \\  if (t.startsWith(q)) return 900 - (t.length - q.length);
    \\  if (t.includes(q)) return 600 - t.indexOf(q);
    \\  const words = q.split(/\\s+/).filter(Boolean);
    \\  let s = 0;
    \\  for (const w of words) if (t.includes(w)) s += 120;
    \\  return s;
    \\}
    \\
    \\function wasmScore(query, text) {
    \\  if (!wasmApi) return jsScore(query, text);
    \\  wasmApi.reset_alloc();
    \\  const qb = utf8.encode(query.toLowerCase());
    \\  const tb = utf8.encode(text.toLowerCase());
    \\  const qPtr = wasmApi.alloc(qb.length);
    \\  const tPtr = wasmApi.alloc(tb.length);
    \\  const mem = new Uint8Array(wasmApi.memory.buffer);
    \\  mem.set(qb, qPtr);
    \\  mem.set(tb, tPtr);
    \\  return wasmApi.score_query(qPtr, qb.length, tPtr, tb.length);
    \\}
    \\
    \\function commandSubcommands(cmd) {
    \\  if (cmd.subcommands.some((s) => s.startsWith("-"))) return [];
    \\  return cmd.subcommands;
    \\}
    \\
    \\function toResults(modules, commands, guides, plans, roadmap, query, category, type) {
    \\  const results = [];
    \\  const q = query.trim();
    \\
    \\  if (type === "all" || type === "modules") {
    \\    for (const mod of modules) {
    \\      if (category && mod.category !== category) continue;
    \\      const score = wasmScore(q, `${mod.name} ${mod.description}`);
    \\      if (!q || score > 0) {
    \\        results.push({
    \\          type: "module",
    \\          score,
    \\          title: mod.name,
    \\          subtitle: `${mod.category} • ${mod.build_flag}`,
    \\          detail: `Path: ${mod.path}\\n\\n${mod.description}`,
    \\          href: `../api/${mod.name}.html`,
    \\        });
    \\      }
    \\      if (type === "all" || type === "symbols") {
    \\        for (const symbol of mod.symbols) {
    \\          const symScore = wasmScore(q, `${mod.name} ${symbol.signature} ${symbol.doc}`);
    \\          if (!q || symScore > 0) {
    \\            results.push({
    \\              type: "symbol",
    \\              score: symScore,
    \\              title: symbol.signature,
    \\              subtitle: `${mod.name} • line ${symbol.line}`,
    \\              detail: `${symbol.doc}\\n\\nSource: ${mod.path}#L${symbol.line}`,
    \\              href: `../api/${mod.name}.html#${symbol.anchor}`,
    \\            });
    \\          }
    \\        }
    \\      }
    \\    }
    \\  }
    \\
    \\  if (type === "all" || type === "commands") {
    \\    for (const cmd of commands) {
    \\      const subs = commandSubcommands(cmd);
    \\      const score = wasmScore(q, `${cmd.name} ${cmd.description} ${subs.join(" ")}`);
    \\      if (!q || score > 0) {
    \\        results.push({
    \\          type: "command",
    \\          score,
    \\          title: cmd.name,
    \\          subtitle: `aliases: ${cmd.aliases.join(", ") || "none"}`,
    \\          detail: `${cmd.description}\\n\\nSubcommands: ${subs.join(", ") || "none"}`,
    \\          href: `../cli/`,
    \\        });
    \\      }
    \\    }
    \\  }
    \\
    \\  if (type === "all" || type === "guides") {
    \\    for (const guide of guides) {
    \\      const score = wasmScore(q, `${guide.title} ${guide.section} ${guide.description}`);
    \\      if (!q || score > 0) {
    \\        results.push({
    \\          type: "guide",
    \\          score,
    \\          title: guide.title,
    \\          subtitle: `${guide.section} • ${guide.slug}`,
    \\          detail: `${guide.description}\\n\\nPermalink: ${guide.permalink}`,
    \\          href: `../${guide.slug}/`,
    \\        });
    \\      }
    \\    }
    \\  }
    \\
    \\  if (type === "all" || type === "plans") {
    \\    for (const plan of plans) {
    \\      const gateText = (plan.gate_commands || []).join(" ; ");
    \\      const score = wasmScore(q, `${plan.title} ${plan.status} ${plan.owner} ${plan.scope} ${gateText}`);
    \\      if (!q || score > 0) {
    \\        results.push({
    \\          type: "plan",
    \\          score,
    \\          title: plan.title,
    \\          subtitle: `${plan.status} • owner: ${plan.owner}`,
    \\          detail: `${plan.scope}\\n\\nValidation: ${gateText || "none"}`,
    \\          href: `../plans/${plan.slug}.md`,
    \\        });
    \\      }
    \\    }
    \\  }
    \\
    \\  if (type === "all" || type === "roadmap") {
    \\    for (const item of roadmap) {
    \\      const score = wasmScore(q, `${item.id} ${item.title} ${item.summary} ${item.track} ${item.horizon} ${item.status} ${item.owner} ${item.plan_title}`);
    \\      if (!q || score > 0) {
    \\        results.push({
    \\          type: "roadmap",
    \\          score,
    \\          title: `${item.id} ${item.title}`,
    \\          subtitle: `${item.horizon} • ${item.track} • ${item.status}`,
    \\          detail: `${item.summary}\\n\\nOwner: ${item.owner}\\nValidation Gate: ${item.validation_gate}\\nPlan: ${item.plan_title}`,
    \\          href: `../roadmap/`,
    \\        });
    \\      }
    \\    }
    \\  }
    \\
    \\  results.sort((a, b) => b.score - a.score || a.title.localeCompare(b.title));
    \\  return results.slice(0, 350);
    \\}
    \\
    \\function renderStats(modules, commands, guides, plans, roadmap) {
    \\  const el = document.getElementById("stats");
    \\  const symbolCount = modules.reduce((n, m) => n + (m.symbols?.length || 0), 0);
    \\  const chips = [
    \\    `${modules.length} modules`,
    \\    `${symbolCount} symbols`,
    \\    `${commands.length} commands`,
    \\    `${guides.length} guides`,
    \\    `${plans.length} plans`,
    \\    `${roadmap.length} roadmap`,
    \\    rankerLabel,
    \\  ];
    \\  el.innerHTML = chips.map((c) => `<span class=\"chip\">${c}</span>`).join("");
    \\}
    \\
    \\function bindUI(modules, commands, guides, plans, roadmap) {
    \\  const search = document.getElementById("searchInput");
    \\  const category = document.getElementById("categoryFilter");
    \\  const type = document.getElementById("typeFilter");
    \\  const resultsEl = document.getElementById("results");
    \\  const details = document.getElementById("details");
    \\
    \\  const categories = [...new Set(modules.map((m) => m.category))].sort();
    \\  for (const cat of categories) {
    \\    const opt = document.createElement("option");
    \\    opt.value = cat;
    \\    opt.textContent = cat;
    \\    category.appendChild(opt);
    \\  }
    \\
    \\  let rows = [];
    \\  let nodes = [];
    \\  let activeIndex = -1;
    \\
    \\  function setActive(index, silent = false) {
    \\    if (!nodes.length) return;
    \\    const clamped = Math.max(0, Math.min(index, nodes.length - 1));
    \\    activeIndex = clamped;
    \\    nodes.forEach((n) => n.classList.remove("active"));
    \\    const node = nodes[clamped];
    \\    const row = rows[clamped];
    \\    node.classList.add("active");
    \\    details.innerHTML = `${row.detail}\\n\\n<a href=\"${row.href}\">Open source page</a>`;
    \\    if (!silent) node.scrollIntoView({ block: "nearest" });
    \\  }
    \\
    \\  function refresh() {
    \\    rows = toResults(modules, commands, guides, plans, roadmap, search.value, category.value, type.value);
    \\    nodes = [];
    \\    resultsEl.innerHTML = "";
    \\    if (!rows.length) {
    \\      activeIndex = -1;
    \\      details.textContent = "No matches for current query/filter.";
    \\      return;
    \\    }
    \\
    \\    for (const [idx, row] of rows.entries()) {
    \\      const li = document.createElement("li");
    \\      li.tabIndex = 0;
    \\      li.innerHTML = `<div><strong>${row.title}</strong></div><div class=\"result-meta\">${row.type} • ${row.subtitle}</div>`;
    \\      li.addEventListener("click", () => setActive(idx));
    \\      li.addEventListener("keydown", (event) => {
    \\        if (event.key === "Enter" || event.key === " ") {
    \\          event.preventDefault();
    \\          setActive(idx);
    \\        } else if (event.key === "ArrowDown") {
    \\          event.preventDefault();
    \\          setActive(activeIndex + 1);
    \\        } else if (event.key === "ArrowUp") {
    \\          event.preventDefault();
    \\          setActive(activeIndex - 1);
    \\        }
    \\      });
    \\      resultsEl.appendChild(li);
    \\      nodes.push(li);
    \\    }
    \\
    \\    setActive(0, true);
    \\  }
    \\
    \\  function handleListNavigation(event) {
    \\    if (!nodes.length) return;
    \\    if (event.key === "ArrowDown") {
    \\      event.preventDefault();
    \\      setActive(activeIndex + 1);
    \\    } else if (event.key === "ArrowUp") {
    \\      event.preventDefault();
    \\      setActive(activeIndex - 1);
    \\    } else if (event.key === "Enter" && activeIndex >= 0) {
    \\      event.preventDefault();
    \\      window.location.href = rows[activeIndex].href;
    \\    }
    \\  }
    \\
    \\  search.addEventListener("input", refresh);
    \\  category.addEventListener("change", refresh);
    \\  type.addEventListener("change", refresh);
    \\  search.addEventListener("keydown", handleListNavigation);
    \\  resultsEl.addEventListener("keydown", handleListNavigation);
    \\  refresh();
    \\}
    \\
    \\async function boot() {
    \\  const [modules, commands, guides, plans, roadmap] = await Promise.all([
    \\    loadJson("./data/modules.json"),
    \\    loadJson("./data/commands.json"),
    \\    loadJson("./data/guides.json"),
    \\    loadJson("./data/plans.json"),
    \\    loadJson("./data/roadmap.json"),
    \\  ]);
    \\  await loadWasmRanker();
    \\  renderStats(modules, commands, guides, plans, roadmap);
    \\  bindUI(modules, commands, guides, plans, roadmap);
    \\}
    \\
    \\boot().catch((err) => {
    \\  const details = document.getElementById("details");
    \\  details.textContent = `Failed to load API app data: ${err.message}`;
    \\});
;
