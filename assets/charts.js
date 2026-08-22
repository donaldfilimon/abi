/* Minimal dependency-free SVG charts for the ABI benchmark dashboard.
   Deliberately hand-rolled: GitHub Pages serves these files as-is, so the
   dashboard must not depend on a CDN, a bundler, or a build step. */

const NS = "http://www.w3.org/2000/svg";

const SERIES_COLORS = ["var(--accent)", "#7c9cf5", "#e08d6d", "#b48ce0"];

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function niceCeil(value) {
  if (!(value > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const scaled = value / magnitude;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10;
  return step * magnitude;
}

function shortLabel(label) {
  // "2026-01-07" -> "01-07"; anything else is passed through untouched.
  return /^\d{4}-\d{2}-\d{2}$/.test(label) ? label.slice(5) : label;
}

/* The SVG scales to the container width, so a fixed viewBox would shrink axis
   text on a phone. Narrow screens get a smaller viewBox and a larger type size,
   which keeps the rendered labels readable. */
function geometry(mount, { wide, compact }) {
  const width = mount.clientWidth || window.innerWidth || 760;
  return width < 560 ? compact : wide;
}

function axes(svg, geom, { yMax, yTicks, labels, formatY }) {
  const { left, right, top, bottom, width, height } = geom;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  const fontSize = geom.fontSize;

  for (let i = 0; i <= yTicks; i += 1) {
    const value = (yMax / yTicks) * i;
    const y = top + plotHeight - (plotHeight / yTicks) * i;
    svg.appendChild(el("line", { x1: left, x2: left + plotWidth, y1: y, y2: y, class: i === 0 ? "axis-line" : "grid-line" }));
    svg.appendChild(el("text", {
      x: left - 6,
      y: y + fontSize / 3,
      class: "axis-text",
      "font-size": fontSize,
      "text-anchor": "end",
    }, formatY(value)));
  }

  const stride = Math.max(1, Math.ceil(labels.length / geom.maxXLabels));
  labels.forEach((label, index) => {
    if (index % stride !== 0 && index !== labels.length - 1) return;
    const x = labels.length === 1 ? left + plotWidth / 2 : left + (plotWidth / (labels.length - 1)) * index;
    svg.appendChild(el("text", {
      x,
      y: top + plotHeight + fontSize + 6,
      class: "axis-text",
      "font-size": fontSize,
      "text-anchor": "middle",
    }, shortLabel(label)));
  });

  return { plotWidth, plotHeight };
}

function baseSvg(geom, ariaLabel) {
  const svg = el("svg", {
    viewBox: `0 0 ${geom.width} ${geom.height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
    "aria-label": ariaLabel,
  });
  return svg;
}

/**
 * Multi-series line chart.
 * @param {Element} mount target element (cleared before rendering)
 * @param {{labels: string[], series: {name: string, values: number[]}[],
 *          unit?: string, ariaLabel?: string, formatY?: (n: number) => string}} options
 */
export function renderLineChart(mount, options) {
  const { labels, series, unit = "", ariaLabel = "line chart" } = options;
  const formatY = options.formatY || ((value) => (value >= 100 ? value.toFixed(0) : value.toFixed(1)));
  const geom = geometry(mount, {
    wide: { width: 760, height: 300, left: 52, right: 14, top: 14, bottom: 34, fontSize: 11, maxXLabels: 8, dot: 3.5 },
    compact: { width: 360, height: 230, left: 34, right: 8, top: 10, bottom: 28, fontSize: 11, maxXLabels: 4, dot: 2.6 },
  });

  const svg = baseSvg(geom, ariaLabel);
  const peak = Math.max(...series.flatMap((entry) => entry.values.filter(Number.isFinite)), 0);
  const yMax = niceCeil(peak * 1.15);
  const { plotWidth, plotHeight } = axes(svg, geom, { yMax, yTicks: 4, labels, formatY });

  const xAt = (index) =>
    labels.length === 1 ? geom.left + plotWidth / 2 : geom.left + (plotWidth / (labels.length - 1)) * index;
  const yAt = (value) => geom.top + plotHeight - (value / yMax) * plotHeight;

  series.forEach((entry, seriesIndex) => {
    const color = entry.color || SERIES_COLORS[seriesIndex % SERIES_COLORS.length];
    const points = entry.values.map((value, index) => [xAt(index), yAt(value)]);
    const path = points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

    if (seriesIndex === 0 && points.length > 1) {
      const areaPath =
        `${path} L${points[points.length - 1][0].toFixed(1)},${(geom.top + plotHeight).toFixed(1)}` +
        ` L${points[0][0].toFixed(1)},${(geom.top + plotHeight).toFixed(1)} Z`;
      svg.appendChild(el("path", { d: areaPath, fill: color, class: "series-area" }));
    }

    svg.appendChild(el("path", { d: path, stroke: color, class: "series-line" }));

    points.forEach(([x, y], index) => {
      const dot = el("circle", { cx: x, cy: y, r: geom.dot, fill: color, class: "series-dot" });
      dot.appendChild(el("title", {}, `${entry.name} — ${labels[index]}: ${entry.values[index]}${unit}`));
      svg.appendChild(dot);
    });
  });

  mount.replaceChildren(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  series.forEach((entry, seriesIndex) => {
    const color = entry.color || SERIES_COLORS[seriesIndex % SERIES_COLORS.length];
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = color;
    item.append(swatch, document.createTextNode(entry.name));
    legend.appendChild(item);
  });
  mount.appendChild(legend);
}

/**
 * Single-series bar chart.
 * @param {Element} mount target element (cleared before rendering)
 * @param {{labels: string[], values: number[], name: string, unit?: string,
 *          ariaLabel?: string, formatY?: (n: number) => string}} options
 */
export function renderBarChart(mount, options) {
  const { labels, values, name, unit = "", ariaLabel = "bar chart" } = options;
  const formatY = options.formatY || ((value) => value.toFixed(0));
  const geom = geometry(mount, {
    wide: { width: 760, height: 260, left: 52, right: 14, top: 14, bottom: 34, fontSize: 11, maxXLabels: 8 },
    compact: { width: 360, height: 210, left: 34, right: 8, top: 10, bottom: 28, fontSize: 11, maxXLabels: 4 },
  });

  const svg = baseSvg(geom, ariaLabel);
  const yMax = niceCeil(Math.max(...values.filter(Number.isFinite), 0) * 1.15);
  const { plotWidth, plotHeight } = axes(svg, geom, { yMax, yTicks: 4, labels, formatY });

  const slot = plotWidth / Math.max(values.length, 1);
  const barWidth = Math.max(3, Math.min(34, slot * 0.55));

  values.forEach((value, index) => {
    const height = Math.max(0, (value / yMax) * plotHeight);
    const x = geom.left + slot * index + (slot - barWidth) / 2;
    const y = geom.top + plotHeight - height;
    const bar = el("rect", { x, y, width: barWidth, height, fill: "var(--accent)", class: "bar", opacity: 0.85 });
    bar.appendChild(el("title", {}, `${name} — ${labels[index]}: ${value}${unit}`));
    svg.appendChild(bar);
  });

  mount.replaceChildren(svg);
}

/**
 * Accessible data table mirroring the charted values.
 * @param {Element} mount target element (cleared before rendering)
 * @param {{columns: string[], rows: (string|number)[][], caption?: string}} options
 */
export function renderTable(mount, { columns, rows, caption }) {
  const table = document.createElement("table");
  table.className = "data";
  if (caption) {
    const captionNode = document.createElement("caption");
    captionNode.textContent = caption;
    table.appendChild(captionNode);
  }

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = column;
    headRow.appendChild(cell);
  });
  head.appendChild(headRow);

  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = String(value);
      tr.appendChild(cell);
    });
    body.appendChild(tr);
  });

  table.append(head, body);
  mount.replaceChildren(table);
}
