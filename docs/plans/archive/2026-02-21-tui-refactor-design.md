# TUI Scaling Fix & Full Refactor Design

**Date:** 2026-02-21
**Status:** Approved

## Problem

The TUI module has several scaling issues and architectural debt:
- Frame width hardcoded to 40–140 columns (doesn't fill wide terminals)
- Unicode emoji/CJK counted by byte-length, causing misaligned box borders
- 10-row overhead is fixed regardless of active panels
- Completion dropdown capped at 50 columns
- No SIGWINCH-driven redraw (500ms poll delay on resize)
- 2,110-line monolithic command launcher (`commands/tui/mod.zig`)
- 5 dashboards duplicate box-drawing, padding, and color logic

## Design Decisions

1. **Width handling:** Uncapped with max-content — remove the 140-col cap, let content fill available space, clamp individual elements to readable widths
2. **Widget unification:** Full component model — composable sub-panels (Header, MetricsRow, ChartArea, StatusBar) that nest and resize automatically
3. **Unicode width:** UTF-8 decode + East Asian Width property lookup for accurate terminal column counting
4. **Launcher split:** Domain-based — `types.zig`, `state.zig`, `render.zig`, `input.zig`, `completion.zig`, `menu.zig`

## Architecture

### New Core Files (`tools/cli/tui/`)

#### `unicode.zig`
- `decodeUtf8(bytes) -> ?{codepoint, len}` — full UTF-8 codepoint decode
- `charWidth(codepoint) -> u2` — East Asian Width property (0, 1, or 2 cols)
- `displayWidth(text) -> usize` — total display columns for a string
- `truncateToWidth(text, max_cols) -> []const u8` — truncate at column boundary
- `padToWidth(text, target_cols) -> usize` — returns padding columns needed
- East Asian Width table: sorted ranges, binary search (~2-3KB)

#### `layout.zig`
- `Rect` — position + size (`x`, `y`, `width`, `height`) with `shrink()`, `splitHorizontal()`, `splitVertical()`
- `Constraint` — `fixed(N)`, `min(N)`, `max(N)`, `percentage(P)`, `fill`
- `distribute(available, constraints) -> []u16` — resolve constraints against available space

#### `component.zig`
- `SubPanel` — label + constraint + render function pointer + context
- `Header(title_fn)` — reusable header sub-panel
- `StatusBar(items_fn)` — reusable status bar
- `MetricsRow(cols)` — N-column metrics display
- `ChartArea(data_fn)` — sparkline/gauge area
- `renderStack(terminal, rect, theme, panels)` — render panels top-to-bottom

#### `render_utils.zig`
- `drawBox(term, rect, style, theme)` — box drawing (single/double/rounded/heavy)
- `drawRow(term, rect, y, content, theme)` — single row within a box
- `fillRow(term, rect, y, char)` — fill row with character
- `writeClipped(term, text, max_cols) -> usize` — write truncated, return cols used
- `writePadded(term, text, target_cols)` — write with padding to fill width
- All functions use `unicode.displayWidth()` internally

### Command Launcher Split (`tools/cli/commands/tui/`)

| File | Lines | Content |
|------|-------|---------|
| `mod.zig` | ~80 | Orchestrator: `run()`, `runInteractive()`, main loop, `printHelp()` |
| `types.zig` | ~120 | Category, Action, Command, MenuItem, HistoryEntry, MatchType, CompletionSuggestion, CompletionState, box constants, color constants |
| `menu.zig` | ~100 | `menuItemsExtended()`, `findByShortcut()`, `commandDefaultArgs()` |
| `state.zig` | ~200 | TuiState struct + all mutation methods |
| `completion.zig` | ~180 | Search, fuzzy match, tab-complete, scoring |
| `render.zig` | ~400 | All rendering functions, using render_utils + unicode |
| `input.zig` | ~100 | Key/mouse event dispatch + action execution |
| `layout.zig` | ~70 | Launcher-specific: `menuStartRow()`, `computeVisibleRows()`, `clickedIndexFromRow()` |

### Dashboard Refactoring

Each dashboard (training, GPU, agent, model, streaming) refactored to:
1. Use `component.zig` sub-panels for composition
2. Use `render_utils.zig` for box-drawing and padding
3. Accept `Rect` for rendering area instead of computing raw positions
4. Use `unicode.zig` for all width calculations

## Implementation Order

1. `unicode.zig` — zero dependencies
2. `layout.zig` — depends on unicode
3. `render_utils.zig` — depends on unicode, layout, terminal
4. `component.zig` — depends on layout, render_utils
5. Command launcher split — extract from monolith, wire up new modules
6. Dashboard refactoring — one at a time

## Migration Strategy

Non-breaking: no public API changes, no feature flag changes, no CLI interface changes. Purely internal restructuring. Existing tests must continue to pass.
