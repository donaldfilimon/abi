# `site/` — GitHub Pages source

The published site at <https://donaldfilimon.github.io/abi/> is built from this
directory. It is plain HTML, CSS, and ES modules: no bundler, no framework, no
CDN, and no build step.

```
site/
├── index.html          landing page
├── benchmarks.html     benchmark dashboard
├── 404.html            not-found page (absolute /abi/ asset paths)
├── .nojekyll           publish the tree verbatim, no Jekyll pass
├── assets/
│   ├── styles.css      the whole design system
│   ├── app.js          theme toggle, copy buttons, footer year
│   └── charts.js       dependency-free SVG line/bar charts + data table
└── data/
    ├── sample_benchmarks.json   synthetic placeholder records
    └── README.md                schema + how to publish real numbers
```

## Deployment

`.github/workflows/benchmarks-gh-pages.yml` uploads this directory as a GitHub
Pages artifact when `site/` changes on `main` (or when manually dispatched),
then deploys it through the `github-pages` environment. Repository Pages
settings use **GitHub Actions** as the source; the workflow does not create a
deployment branch.

## Local preview

The pages load `assets/*.js` as ES modules, so `file://` will not work. Serve the
directory over HTTP instead:

```bash
python3 -m http.server 4173 --directory site
# then open http://localhost:4173/
```

## Conventions worth keeping

- **Relative asset paths** in `index.html` and `benchmarks.html`. The site is
  served from the `/abi/` sub-path, so a leading `/` resolves to the wrong place.
  `404.html` is the deliberate exception — GitHub serves it for URLs at any
  depth, so its asset links are absolute `/abi/…` paths.
- **No external requests.** Everything the page needs ships in this directory,
  which is why the charts are hand-rolled SVG rather than a charting library.
- **Claim honesty applies to the site too.** Copy here follows the same rules as
  the rest of the repository: no performance, deployment, or capability claim
  without a test, benchmark artifact, or source file behind it. See
  `docs/contracts/external-claims-audit.mdx`. The dashboard's data is synthetic
  and says so on the page.
- **Theme.** Colours come from the Mintlify docs theme in `docs/docs.json`
  (`#009688` / `#4db6ac` / `#00695c`) so the site and the docs match. Dark is the
  default; the toggle stores an explicit choice in `localStorage`.
