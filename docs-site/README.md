# ABI Framework Documentation Site

Professional static documentation site for the ABI Framework.

## Structure

```
docs-site/
├── index.html              # Landing page
├── pages/                  # Individual documentation pages
│   ├── quickstart.html
│   ├── intro.html
│   ├── agents.html
│   ├── api.html
│   └── ...
├── assets/
│   ├── css/
│   │   └── style.css    # Professional styling
│   └── js/
│       └── search.js     # Search functionality
└── generate.py             # Script to generate HTML from Markdown
```

## Features

- 📚 **Comprehensive Documentation**: Full coverage of all framework features
- 🔍 **Instant Search**: Real-time search across all documentation
- 📱 **Responsive Design**: Works seamlessly on mobile, tablet, and desktop
- 🎨 **Professional UI**: Clean, modern interface with smooth animations
- ⌨️ **Keyboard Shortcuts**: Ctrl/Cmd+K for search, Escape to close
- 🌙 **Dark/Light Ready**: CSS variables for easy theming
- ♿ **Accessible**: Semantic HTML, ARIA labels, keyboard navigation
- ⚡ **Fast**: Pure CSS/JS, no dependencies, instant loading

## Local Development

### View with Python

```bash
cd docs-site
python -m http.server 8000
# Then open: http://localhost:8000
```

### View with Node.js

```bash
cd docs-site
npx serve
# Then open: http://localhost:3000
```

### Generate HTML from Markdown

```bash
cd docs-site
python generate.py
```

This will generate HTML pages from the markdown files in the parent directory.

## Deployment

### GitHub Pages

The site is automatically deployed to GitHub Pages via GitHub Actions.

- **Source**: `docs-site/` directory
- **URL**: https://donaldfilimon.github.io/abi/
- **Branch**: `gh-pages`

To deploy manually:

```bash
git checkout -b gh-pages
git subtree push --prefix docs-site origin gh-pages
```

### Custom Domain

To use a custom domain (e.g., `docs.abiframework.dev`):

1. Go to GitHub repository Settings → Pages
2. Add your custom domain
3. Update DNS records (CNAME or A record)
4. Create `docs-site/CNAME` file with your domain

## Customization

### Styling

Edit `assets/css/style.css` to customize the appearance:

```css
:root {
    --primary: #6366f1;      /* Main brand color */
    --text: #1a202c;          /* Text color */
    --bg: #ffffff;             /* Background */
}
```

### Navigation

Edit the sidebar in the HTML templates or modify `generate.py` to add new pages.

### Search Index

Edit `assets/js/search.js` to add or remove pages from the search index:

```javascript
const docs = [
    {
        title: 'Page Title',
        url: 'pages/page.html',
        description: 'Page description',
        category: 'Category'
    },
    // ... more pages
];
```

## Browser Support

- Chrome/Edge (latest 2 versions)
- Firefox (latest 2 versions)
- Safari (latest 2 versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Lighthouse Score: 95+
- First Contentful Paint: < 1s
- Time to Interactive: < 2s
- Total Bundle Size: < 50KB

## Contributing

To add new documentation:

1. Write or update markdown in the parent directory
2. Run `python generate.py` to regenerate HTML
3. Test locally with `python -m http.server 8000`
4. Commit and push changes

## License

MIT License - See parent directory LICENSE file.
