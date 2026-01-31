# Documentation
> **Last reviewed:** 2026-01-31

This directory is the source for the ABI documentation site.

## Build

Generate the site with the custom Zig pipeline:

```
zig build docs-site
```

Output is written to:

```
zig-out/docs/
```

MkDocs is no longer used for this site.

## Layout

- `site.json` defines navigation and page metadata.
- `content/` holds HTML fragments for each page.
- `assets/` holds CSS and JavaScript.
