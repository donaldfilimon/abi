# ABI Documentation

This directory is the source for the ABI documentation site and is designed to work with **GitHub Pages** (Jekyll) and optionally with the Zig-based site generator. Built with Zig 0.16.0-dev.2611+f996d2866 (see repo root `.zigversion`).

## GitHub Pages (recommended)

The `docs/` folder is set up for [GitHub Pages](https://docs.github.com/en/pages) with Jekyll and the [Just the Docs](https://just-the-docs.github.io/just-the-docs/) theme.

1. In your repo: **Settings → Pages → Source**: Deploy from a **branch**, branch **main**, folder **/docs**.
2. Push to `main`; GitHub will build and serve the site (e.g. `https://username.github.io/abi/`).

### Local preview

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Open http://localhost:4000 (or the URL Jekyll prints). Use `--baseurl /abi` if your site will be at `https://username.github.io/abi/`.

## Structure

| Item | Description |
|------|-------------|
| `_config.yml` | Jekyll config, theme (Just the Docs), nav, collections |
| `_docs/` | Content pages (Markdown); copied from `content/` for Jekyll |
| `content/` | Canonical Markdown sources (used by Zig `docs-site` if you use it) |
| `index.md` | Home page |
| `api/` | Generated API reference (`abi gendocs` or `zig build gendocs`) |
| `site.json` | Nav manifest for the Zig docs-site generator (optional) |
| `Gemfile` | Ruby deps for Jekyll (optional for local preview) |

## API reference

Generate the API reference (writes into `docs/api/`) from the repo root:

```bash
abi gendocs
# or
zig build gendocs
```

Commit `docs/api/` if you want the generated API docs to appear on GitHub Pages.

## Zig docs-site (optional)

If you use the Zig-based site generator:

```bash
zig build docs-site
```

Output is written to `zig-out/docs/`. The generator reads `site.json` and `content/*.md`. You can keep using it for local builds; GitHub Pages will use the Jekyll setup above.
