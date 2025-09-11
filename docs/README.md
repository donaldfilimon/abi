# ABI Documentation

This directory contains the documentation for the ABI (Advanced AI Framework) project.

## Documentation Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── Gemfile             # Ruby dependencies
├── index.md            # Home page
├── getting-started.md  # Getting started guide
├── api.md              # API reference
├── performance.md      # Performance guide
├── _layouts/           # Jekyll layouts
├── _includes/          # Reusable content
├── assets/             # CSS, JS, images
└── generated/          # Auto-generated documentation
```

## Building Documentation

### Prerequisites

- Ruby 2.7+
- Bundler
- Jekyll

### Local Development

```bash
# Install dependencies
cd docs
bundle install

# Serve documentation locally
bundle exec jekyll serve

# Build for production
bundle exec jekyll build
```

The documentation will be available at `http://localhost:4000`.

## Documentation Guidelines

### Writing Style

- Use clear, concise language
- Include code examples for all major features
- Provide both basic and advanced usage examples
- Document all public APIs
- Include performance considerations

### Code Examples

- Use syntax highlighting for code blocks
- Include imports and complete examples
- Show error handling patterns
- Provide both simple and complex examples

### API Documentation

- Document all public functions, types, and constants
- Include parameter descriptions and return values
- Show usage examples for each API
- Document error conditions

## Auto-Generated Content

Some documentation is auto-generated from the codebase:

- API reference from Zig doc comments
- Performance benchmarks
- Code examples from test files

To regenerate auto-generated content:

```bash
# Generate API documentation
zig build docs

# Generate performance reports
zig build perf-ci

# Update examples
zig build docs-examples
```

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

### Manual Deployment

```bash
# Build the site
cd docs
bundle exec jekyll build --destination ../_site

# The _site directory contains the built documentation
```

## Contributing to Documentation

1. Make changes to the appropriate `.md` files
2. Test locally with `bundle exec jekyll serve`
3. Ensure all links are working
4. Follow the established style guidelines
5. Submit a pull request

### Documentation Checklist

- [ ] All public APIs documented
- [ ] Code examples provided
- [ ] Links are working
- [ ] Consistent formatting
- [ ] Performance considerations included
- [ ] Error handling documented

## Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)
- [ABI Codebase](../src/) - Source code for reference