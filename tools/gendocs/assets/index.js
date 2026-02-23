const loadJson = async (path) => {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return await res.json();
};

const utf8 = new TextEncoder();
let wasmApi = null;
let rankerLabel = "JS ranker fallback";

async function tryLoadRanker(path, label) {
  try {
    const bytes = await fetch(path).then((r) => {
      if (!r.ok) throw new Error("missing");
      return r.arrayBuffer();
    });
    const mod = await WebAssembly.instantiate(bytes, {});
    const exp = mod.instance.exports;
    if (exp.memory && exp.alloc && exp.reset_alloc && exp.score_query) {
      wasmApi = exp;
      rankerLabel = label;
      return true;
    }
  } catch (_) {}
  return false;
}

async function loadWasmRanker() {
  // Prefer component output when present and instantiable, then fallback to plain wasm.
  if (await tryLoadRanker("./data/docs_engine.component.wasm", "WASM ranker (component)")) return;
  await tryLoadRanker("./data/docs_engine.wasm", "WASM ranker (plain)");
}

function jsScore(query, text) {
  if (!query) return 1;
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (t === q) return 1200;
  if (t.startsWith(q)) return 900 - (t.length - q.length);
  if (t.includes(q)) return 600 - t.indexOf(q);
  const words = q.split(/\s+/).filter(Boolean);
  let s = 0;
  for (const w of words) if (t.includes(w)) s += 120;
  return s;
}

function wasmScore(query, text) {
  if (!wasmApi) return jsScore(query, text);
  wasmApi.reset_alloc();
  const qb = utf8.encode(query.toLowerCase());
  const tb = utf8.encode(text.toLowerCase());
  const qPtr = wasmApi.alloc(qb.length);
  const tPtr = wasmApi.alloc(tb.length);
  const mem = new Uint8Array(wasmApi.memory.buffer);
  mem.set(qb, qPtr);
  mem.set(tb, tPtr);
  return wasmApi.score_query(qPtr, qb.length, tPtr, tb.length);
}

function commandSubcommands(cmd) {
  if (cmd.subcommands.some((s) => s.startsWith("-"))) return [];
  return cmd.subcommands;
}

function toResults(modules, commands, guides, plans, roadmap, query, category, type) {
  const results = [];
  const q = query.trim();

  if (type === "all" || type === "modules") {
    for (const mod of modules) {
      if (category && mod.category !== category) continue;
      const score = wasmScore(q, `${mod.name} ${mod.description}`);
      if (!q || score > 0) {
        results.push({
          type: "module",
          score,
          title: mod.name,
          subtitle: `${mod.category} • ${mod.build_flag}`,
          detail: `Path: ${mod.path}\n\n${mod.description}`,
          href: `./api/${mod.name}.html`,
        });
      }
      if (type === "all" || type === "symbols") {
        for (const symbol of mod.symbols) {
          const symScore = wasmScore(q, `${mod.name} ${symbol.signature} ${symbol.doc}`);
          if (!q || symScore > 0) {
            results.push({
              type: "symbol",
              score: symScore,
              title: symbol.signature,
              subtitle: `${mod.name} • line ${symbol.line}`,
              detail: `${symbol.doc}\n\nSource: ${mod.path}#L${symbol.line}`,
              href: `./api/${mod.name}.html#${symbol.anchor}`,
            });
          }
        }
      }
    }
  }

  if (type === "all" || type === "commands") {
    for (const cmd of commands) {
      const subs = commandSubcommands(cmd);
      const score = wasmScore(q, `${cmd.name} ${cmd.description} ${subs.join(" ")}`);
      if (!q || score > 0) {
        results.push({
          type: "command",
          score,
          title: cmd.name,
          subtitle: `aliases: ${cmd.aliases.join(", ") || "none"}`,
          detail: `${cmd.description}\n\nSubcommands: ${subs.join(", ") || "none"}`,
          href: `./cli/`,
        });
      }
    }
  }

  if (type === "all" || type === "guides") {
    for (const guide of guides) {
      const score = wasmScore(q, `${guide.title} ${guide.section} ${guide.description}`);
      if (!q || score > 0) {
        results.push({
          type: "guide",
          score,
          title: guide.title,
          subtitle: `${guide.section} • ${guide.slug}`,
          detail: `${guide.description}\n\nPermalink: ${guide.permalink}`,
          href: `./${guide.slug}/`,
        });
      }
    }
  }

  if (type === "all" || type === "plans") {
    for (const plan of plans) {
      const gateText = (plan.gate_commands || []).join(" ; ");
      const score = wasmScore(q, `${plan.title} ${plan.status} ${plan.owner} ${plan.scope} ${gateText}`);
      if (!q || score > 0) {
        results.push({
          type: "plan",
          score,
          title: plan.title,
          subtitle: `${plan.status} • owner: ${plan.owner}`,
          detail: `${plan.scope}\n\nValidation: ${gateText || "none"}`,
          href: `./plans/${plan.slug}.md`,
        });
      }
    }
  }

  if (type === "all" || type === "roadmap") {
    for (const item of roadmap) {
      const score = wasmScore(q, `${item.id} ${item.title} ${item.summary} ${item.track} ${item.horizon} ${item.status} ${item.owner} ${item.plan_title}`);
      if (!q || score > 0) {
        results.push({
          type: "roadmap",
          score,
          title: `${item.id} ${item.title}`,
          subtitle: `${item.horizon} • ${item.track} • ${item.status}`,
          detail: `${item.summary}\n\nOwner: ${item.owner}\nValidation Gate: ${item.validation_gate}\nPlan: ${item.plan_title}`,
          href: `./roadmap/`,
        });
      }
    }
  }

  results.sort((a, b) => b.score - a.score || a.title.localeCompare(b.title));
  return results.slice(0, 350);
}

function renderStats(modules, commands, guides, plans, roadmap) {
  const el = document.getElementById("stats");
  const symbolCount = modules.reduce((n, m) => n + (m.symbols?.length || 0), 0);
  const chips = [
    `${modules.length} modules`,
    `${symbolCount} symbols`,
    `${commands.length} commands`,
    `${guides.length} guides`,
    `${plans.length} plans`,
    `${roadmap.length} roadmap`,
    rankerLabel,
  ];
  el.innerHTML = chips.map((c) => `<span class="chip">${c}</span>`).join("");
}

function bindUI(modules, commands, guides, plans, roadmap) {
  const search = document.getElementById("searchInput");
  const category = document.getElementById("categoryFilter");
  const type = document.getElementById("typeFilter");
  const resultsEl = document.getElementById("results");
  const details = document.getElementById("details");

  const categories = [...new Set(modules.map((m) => m.category))].sort();
  for (const cat of categories) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    category.appendChild(opt);
  }

  let rows = [];
  let nodes = [];
  let activeIndex = -1;

  function setActive(index, silent = false) {
    if (!nodes.length) return;
    const clamped = Math.max(0, Math.min(index, nodes.length - 1));
    activeIndex = clamped;
    nodes.forEach((n) => n.classList.remove("active"));
    const node = nodes[clamped];
    const row = rows[clamped];
    node.classList.add("active");
    details.innerHTML = `${row.detail}\n\n<a href="${row.href}">Open source page</a>`;
    if (!silent) node.scrollIntoView({ block: "nearest" });
  }

  function refresh() {
    rows = toResults(modules, commands, guides, plans, roadmap, search.value, category.value, type.value);
    nodes = [];
    resultsEl.innerHTML = "";
    if (!rows.length) {
      activeIndex = -1;
      details.textContent = "No matches for current query/filter.";
      return;
    }

    for (const [idx, row] of rows.entries()) {
      const li = document.createElement("li");
      li.tabIndex = 0;
      li.innerHTML = `<div><strong>${row.title}</strong></div><div class="result-meta">${row.type} • ${row.subtitle}</div>`;
      li.addEventListener("click", () => setActive(idx));
      li.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          setActive(idx);
        } else if (event.key === "ArrowDown") {
          event.preventDefault();
          setActive(activeIndex + 1);
        } else if (event.key === "ArrowUp") {
          event.preventDefault();
          setActive(activeIndex - 1);
        }
      });
      resultsEl.appendChild(li);
      nodes.push(li);
    }

    setActive(0, true);
  }

  function handleListNavigation(event) {
    if (!nodes.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive(activeIndex + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive(activeIndex - 1);
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      window.location.href = rows[activeIndex].href;
    }
  }

  search.addEventListener("input", refresh);
  category.addEventListener("change", refresh);
  type.addEventListener("change", refresh);
  search.addEventListener("keydown", handleListNavigation);
  resultsEl.addEventListener("keydown", handleListNavigation);
  refresh();
}

async function boot() {
  const [modules, commands, guides, plans, roadmap] = await Promise.all([
    loadJson("./data/modules.json"),
    loadJson("./data/commands.json"),
    loadJson("./data/guides.json"),
    loadJson("./data/plans.json"),
    loadJson("./data/roadmap.json"),
  ]);
  await loadWasmRanker();
  renderStats(modules, commands, guides, plans, roadmap);
  bindUI(modules, commands, guides, plans, roadmap);
}

boot().catch((err) => {
  const details = document.getElementById("details");
  details.textContent = `Failed to load API app data: ${err.message}`;
});
