# Documentation Generation Guide

## Quick Start

### Generate Documentation

```bash
# Using Zig build system
zig build docs

# Or use the verification script
./verify_docs.sh
```

### Preview Documentation

```bash
# Serve locally
python3 -m http.server --directory docs 8000

# Open in browser
http://localhost:8000
```

## What Gets Generated

The documentation generator creates:

- **Module Reference** - Complete API documentation for all modules
- **API Reference** - Detailed API guide with examples
- **Examples** - Usage examples and tutorials
- **Performance Guide** - Optimization tips and best practices
- **Code Index** - Scanned index of all public declarations
- **Search Functionality** - JSON-based search index
- **GitHub Pages Assets** - CSS, JavaScript, and layouts
- **Native Zig Docs** - Generated via `zig doc` command

## Directory Structure

```
docs/
├── generated/          # Auto-generated documentation
│   ├── MODULE_REFERENCE.md
│   ├── API_REFERENCE.md
│   ├── EXAMPLES.md
│   ├── PERFORMANCE_GUIDE.md
│   ├── DEFINITIONS_REFERENCE.md
│   ├── CODE_API_INDEX.md
│   └── search_index.json
├── assets/            # Styles and scripts
│   ├── css/
│   │   └── documentation.css
│   └── js/
│       ├── documentation.js
│       └── search.js
├── _layouts/          # Jekyll layouts
│   └── documentation.html
├── _data/            # Navigation data
│   └── navigation.yml
├── zig-docs/         # Native Zig documentation
│   └── index.html
├── .nojekyll         # Disable Jekyll processing
├── _config.yml       # Jekyll configuration
├── index.html        # Main entry point
└── README.md         # Documentation landing page
```

## Build System Integration

The documentation generator is integrated into `build.zig`:

```zig
const docs_step = b.step("docs", "Generate API documentation");
```

Available commands:

```bash
# Generate documentation
zig build docs

# Run application
zig build run

# Run tests
zig build test

# Build only (no run)
zig build
```

## Verification

Use the verification script to ensure all files are generated correctly:

```bash
./verify_docs.sh
```

This will:
1. Check for Zig compiler
2. Clean old documentation
3. Generate new documentation
4. Verify all expected files exist
5. Show the documentation structure

## Deployment

### GitHub Pages (Automatic)

Push to main branch and GitHub Actions will automatically deploy:

```bash
git add docs/
git commit -m "Update documentation"
git push origin main
```

The workflow (`.github/workflows/deploy_docs.yml`) handles deployment automatically.

### Manual Deployment

Enable GitHub Pages in repository settings:
1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: /docs
5. Save

## Troubleshooting

### Documentation Not Generating

**Problem**: `zig build docs` fails

**Solutions**:
1. Check Zig version: `zig version` (should match `.zigversion`)
2. Clean build cache: `rm -rf zig-cache zig-out`
3. Rebuild: `zig build docs`

### Missing Files

**Problem**: Some documentation files are missing

**Solutions**:
1. Run verification: `./verify_docs.sh`
2. Check build output for errors
3. Ensure `src/` directory is accessible

### Native Docs Not Generated

**Problem**: `docs/zig-docs/` is empty or has placeholder

**Cause**: `zig doc` command failed (this is normal in some environments)

**Result**: A fallback placeholder is created automatically

**Fix**: The docs generator handles this gracefully

## File Descriptions

### Generated Content

- **MODULE_REFERENCE.md** - Documentation for all framework modules
- **API_REFERENCE.md** - Comprehensive API guide
- **EXAMPLES.md** - Code examples and tutorials
- **PERFORMANCE_GUIDE.md** - Performance optimization guide
- **DEFINITIONS_REFERENCE.md** - Glossary and quick reference
- **CODE_API_INDEX.md** - Scanned source code index
- **search_index.json** - Search index data

### Configuration

- **.nojekyll** - Tells GitHub Pages not to process with Jekyll
- **_config.yml** - Jekyll configuration for GitHub Pages
- **_layouts/documentation.html** - HTML layout template
- **_data/navigation.yml** - Navigation menu structure

### Assets

- **documentation.css** - Styles for documentation pages
- **documentation.js** - Table of contents generation and utilities
- **search.js** - Search functionality

### Entry Points

- **index.html** - Main documentation page
- **README.md** - Documentation landing page

## Advanced Usage

### Customize Documentation

Edit the generators in `src/tools/docs_generator/generators/`:

- `module_docs.zig` - Module documentation
- `api_reference.zig` - API reference
- `examples.zig` - Examples
- `performance_guide.zig` - Performance guide
- `code_index.zig` - Code scanner
- `search_index.zig` - Search index builder

### Add New Documentation Steps

1. Create a new generator in `src/tools/docs_generator/generators/`
2. Add it to `planner.zig` default steps
3. Rebuild and run: `zig build docs`

### Modify Styles

Edit `src/tools/docs_generator/config.zig` to customize:
- CSS styles
- JavaScript functionality
- Jekyll configuration
- Navigation structure

## Testing

Test documentation generation without Zig:

```bash
# Run simulation
./test_docs_generation.sh
```

This creates sample files to verify the structure.

## Support

For issues or questions:
1. Check `DOCS_VERIFICATION_COMPLETE.md` for verification status
2. Check `DOCS_REGENERATION_STATUS.md` for detailed information
3. Run `./verify_docs.sh` to diagnose problems
4. Check build output: `zig build docs --verbose`

## Summary

```bash
# Clean and regenerate everything
./verify_docs.sh

# Just generate
zig build docs

# Preview locally
python3 -m http.server --directory docs 8000
```

Documentation is automatically versioned and deployed via GitHub Pages when pushed to the main branch.
