const std = @import("std");
const abi = @import("abi");

/// Documentation generator for WDBX-AI project
/// Generates comprehensive API documentation from source code with enhanced GitHub Pages support
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating WDBX-AI API Documentation with GitHub Pages optimization", .{});

    // Create comprehensive docs directory structure
    try std.fs.cwd().makePath("docs/generated");
    try std.fs.cwd().makePath("docs/assets");
    // Static site (no Jekyll)
    try generateNoJekyll(allocator);

    // Generate module documentation
    try generateModuleDocs(allocator);
    try generateApiReference(allocator);
    try generateExamples(allocator);
    try generatePerformanceGuide(allocator);
    try generateDefinitionsReference(allocator);

    // Enhanced documentation features
    try generateCodeApiIndex(allocator);
    try generateSearchIndex(allocator);

    // GitHub Pages optimizations
    try generateJekyllConfig(allocator);
    try generateGitHubPagesLayout(allocator);
    try generateNavigationData(allocator);
    try generateSEOMetadata(allocator);
    try generateDocsIndexHtml(allocator);
    try generateReadmeRedirect(allocator);

    // Generate Zig native documentation
    try generateZigNativeDocs(allocator);

    std.log.info("✅ GitHub Pages documentation generation completed!", .{});
}

/// Generate Jekyll configuration for GitHub Pages
fn generateJekyllConfig(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_config.yml", .{});
    defer file.close();

    const content =
        \\# WDBX-AI Documentation - Jekyll Configuration
        \\title: "WDBX-AI Documentation"
        \\description: "High-performance vector database with AI capabilities"
        \\url: "https://your-username.github.io"
        \\baseurl: "/wdbx-ai"
        \\
        \\# GitHub Pages settings
        \\remote_theme: pages-themes/minimal@v0.2.0
        \\plugins:
        \\  - jekyll-remote-theme
        \\  - jekyll-sitemap
        \\  - jekyll-feed
        \\  - jekyll-seo-tag
        \\
        \\# Navigation structure
        \\navigation:
        \\  - title: "Home"
        \\    url: "/"
        \\  - title: "API Reference"
        \\    url: "/generated/API_REFERENCE"
        \\  - title: "Module Reference"
        \\    url: "/generated/MODULE_REFERENCE"
        \\  - title: "Examples"
        \\    url: "/generated/EXAMPLES"
        \\  - title: "Performance Guide"
        \\    url: "/generated/PERFORMANCE_GUIDE"
        \\  - title: "Definitions"
        \\    url: "/generated/DEFINITIONS_REFERENCE"
        \\  - title: "Code Index"
        \\    url: "/generated/CODE_API_INDEX"
        \\
        \\# SEO and metadata
        \\lang: en
        \\author:
        \\  name: "WDBX-AI Team"
        \\  email: "team@wdbx-ai.dev"
        \\
        \\# GitHub repository
        \\github:
        \\  repository_url: "https://github.com/your-username/wdbx-ai"
        \\  repository_name: "wdbx-ai"
        \\  owner_name: "your-username"
        \\
        \\# Social media
        \\social:
        \\  type: "Organization"
        \\  links:
        \\    - "https://github.com/your-username/wdbx-ai"
        \\
        \\# Build settings
        \\markdown: kramdown
        \\highlighter: rouge
        \\theme: minima
        \\
        \\# Exclude from build
        \\exclude:
        \\  - "*.zig"
        \\  - "zig-*"
        \\  - "build.zig"
        \\  - "README.md"
        \\
        \\# Include in build
        \\include:
        \\  - "_redirects"
        \\
        \\# Kramdown settings
        \\kramdown:
        \\  input: GFM
        \\  syntax_highlighter: rouge
        \\  syntax_highlighter_opts:
        \\    css_class: 'highlight'
        \\    span:
        \\      line_numbers: false
        \\    block:
        \\      line_numbers: true
        \\
        \\# Collections
        \\collections:
        \\  generated:
        \\    output: true
        \\    permalink: /:collection/:name/
        \\
        \\# Defaults
        \\defaults:
        \\  - scope:
        \\      path: "generated"
        \\      type: "generated"
        \\    values:
        \\      layout: "documentation"
        \\      sitemap: true
        \\
    ;

    try file.writeAll(content);
}

/// Generate GitHub Pages layout template
fn generateGitHubPagesLayout(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_layouts/documentation.html", .{});
    defer file.close();

    const content =
        \\---
        \\layout: default
        \\---
        \\
        \\<!DOCTYPE html>
        \\<html lang="{{ page.lang | default: site.lang | default: 'en' }}">
        \\<head>
        \\  <meta charset="utf-8">
        \\  <meta http-equiv="X-UA-Compatible" content="IE=edge">
        \\  <meta name="viewport" content="width=device-width, initial-scale=1">
        \\  <link rel="stylesheet" href="{{ '/assets/css/style.css?v=' | append: site.github.build_revision | relative_url }}">
        \\  <link rel="stylesheet" href="{{ '/assets/css/documentation.css' | relative_url }}">
        \\  
        \\  <!-- SEO tags -->
        \\  {% seo %}
        \\  
        \\  <!-- Syntax highlighting -->
        \\  <link rel="stylesheet" href="{{ '/assets/css/syntax.css' | relative_url }}">
        \\  
        \\  <!-- Search functionality -->
        \\  <script src="{{ '/assets/js/search.js' | relative_url }}" defer></script>
        \\</head>
        \\<body>
        \\  <div class="wrapper">
        \\    <header>
        \\      <h1><a href="{{ '/' | relative_url }}">{{ site.title | default: site.github.repository_name }}</a></h1>
        \\      <p>{{ site.description | default: site.github.project_tagline }}</p>
        \\      
        \\      <!-- Search box -->
        \\      <div class="search-container">
        \\        <input type="search" id="search-input" placeholder="Search documentation..." autocomplete="off">
        \\        <div id="search-results" class="search-results hidden"></div>
        \\      </div>
        \\      
        \\      <!-- Navigation -->
        \\      <nav class="doc-navigation">
        \\        <h3>Documentation</h3>
        \\        <ul>
        \\          {% for nav_item in site.navigation %}
        \\          <li><a href="{{ nav_item.url | relative_url }}" {% if page.url == nav_item.url %}class="current"{% endif %}>{{ nav_item.title }}</a></li>
        \\          {% endfor %}
        \\        </ul>
        \\      </nav>
        \\      
        \\      <!-- GitHub links -->
        \\      {% if site.github.is_project_page %}
        \\      <p class="view">
        \\        <a href="{{ site.github.repository_url }}">View the Project on GitHub <small>{{ site.github.repository_nwo }}</small></a>
        \\      </p>
        \\      {% endif %}
        \\      
        \\      <!-- Download links -->
        \\      {% if site.github.is_project_page %}
        \\      <ul class="downloads">
        \\        {% if site.github.zip_url %}
        \\        <li><a href="{{ site.github.zip_url }}">Download <strong>ZIP File</strong></a></li>
        \\        {% endif %}
        \\        {% if site.github.tar_url %}
        \\        <li><a href="{{ site.github.tar_url }}">Download <strong>TAR Ball</strong></a></li>
        \\        {% endif %}
        \\        <li><a href="{{ site.github.repository_url }}">View On <strong>GitHub</strong></a></li>
        \\      </ul>
        \\      {% endif %}
        \\    </header>
        \\    
        \\    <section class="documentation-content">
        \\      <!-- Table of contents -->
        \\      <div id="toc" class="table-of-contents">
        \\        <h4>Table of Contents</h4>
        \\        <ul id="toc-list"></ul>
        \\      </div>
        \\      
        \\      <!-- Main content -->
        \\      <div class="content">
        \\        {{ content }}
        \\        
        \\        <!-- Feedback section -->
        \\        <div class="feedback-section">
        \\          <h3>Feedback</h3>
        \\          <p>Found an issue with this documentation? 
        \\             <a href="{{ site.github.repository_url }}/issues/new?title=Documentation%20Issue&body=Page:%20{{ page.url }}">Report it on GitHub</a>
        \\          </p>
        \\        </div>
        \\      </div>
        \\    </section>
        \\    
        \\    <footer>
        \\      {% if site.github.is_project_page %}
        \\      <p>This project is maintained by <a href="{{ site.github.owner_url }}">{{ site.github.owner_name }}</a></p>
        \\      {% endif %}
        \\      <p><small>Hosted on GitHub Pages &mdash; Theme by <a href="https://github.com/orderedlist">orderedlist</a></small></p>
        \\      <p><small>Generated with Zig documentation tools</small></p>
        \\    </footer>
        \\  </div>
        \\  
        \\  <!-- JavaScript for enhanced functionality -->
        \\  <script src="{{ '/assets/js/documentation.js' | relative_url }}"></script>
        \\  <script src="{{ '/assets/js/scale.fix.js' | relative_url }}"></script>
        \\  
        \\  <!-- Analytics (if configured) -->
        \\  {% if site.google_analytics %}
        \\  <script>
        \\    (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
        \\    (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
        \\    m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
        \\    })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
        \\    ga('create', '{{ site.google_analytics }}', 'auto');
        \\    ga('send', 'pageview');
        \\  </script>
        \\  {% endif %}
        \\</body>
        \\</html>
        \\
    ;

    try file.writeAll(content);
}

/// Generate navigation data for enhanced site structure
fn generateNavigationData(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_data/navigation.yml", .{});
    defer file.close();

    const content =
        \\# Main navigation structure
        \\main:
        \\  - title: "Getting Started"
        \\    children:
        \\      - title: "Quick Start"
        \\        url: "/examples/#quick-start"
        \\      - title: "Installation"
        \\        url: "/examples/#installation"
        \\      - title: "Basic Usage"
        \\        url: "/examples/#basic-vector-database"
        \\
        \\  - title: "API Documentation"
        \\    children:
        \\      - title: "Database API"
        \\        url: "/generated/API_REFERENCE/#database-api"
        \\      - title: "AI API"
        \\        url: "/generated/API_REFERENCE/#ai-api"
        \\      - title: "SIMD API"
        \\        url: "/generated/API_REFERENCE/#simd-api"
        \\      - title: "Plugin API"
        \\        url: "/generated/API_REFERENCE/#plugin-api"
        \\
        \\  - title: "Reference"
        \\    children:
        \\      - title: "Module Reference"
        \\        url: "/generated/MODULE_REFERENCE/"
        \\      - title: "Code Index"
        \\        url: "/generated/CODE_API_INDEX/"
        \\      - title: "Definitions"
        \\        url: "/generated/DEFINITIONS_REFERENCE/"
        \\
        \\  - title: "Guides"
        \\    children:
        \\      - title: "Performance Guide"
        \\        url: "/generated/PERFORMANCE_GUIDE/"
        \\      - title: "Examples"
        \\        url: "/generated/EXAMPLES/"
        \\      - title: "Best Practices"
        \\        url: "/generated/EXAMPLES/#performance-optimization"
        \\
        \\# Sidebar navigation for documentation pages
        \\docs:
        \\  - title: "Core Concepts"
        \\    children:
        \\      - title: "Vector Database"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#vector-database"
        \\      - title: "Embeddings"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#embeddings"
        \\      - title: "HNSW Index"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#hnsw-hierarchical-navigable-small-world"
        \\
        \\  - title: "AI & ML"
        \\    children:
        \\      - title: "Neural Networks"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#neural-network"
        \\      - title: "Training"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#backpropagation"
        \\      - title: "Agents"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#agent-based-systems"
        \\
        \\  - title: "Performance"
        \\    children:
        \\      - title: "SIMD Operations"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#simd-single-instruction-multiple-data"
        \\      - title: "Memory Management"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#memory-management"
        \\      - title: "Caching"
        \\        url: "/generated/DEFINITIONS_REFERENCE/#caching-strategies"
        \\
    ;

    try file.writeAll(content);
}

/// Generate SEO metadata and frontmatter for pages
fn generateSEOMetadata(_: std.mem.Allocator) !void {
    // Generate sitemap.xml
    const sitemap_file = try std.fs.cwd().createFile("docs/sitemap.xml", .{});
    defer sitemap_file.close();

    const sitemap_content =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>1.0</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/API_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.9</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/MODULE_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.8</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/EXAMPLES/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.8</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/PERFORMANCE_GUIDE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>monthly</changefreq>
        \\    <priority>0.7</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/DEFINITIONS_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>monthly</changefreq>
        \\    <priority>0.7</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://your-username.github.io/wdbx-ai/generated/CODE_API_INDEX/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>daily</changefreq>
        \\    <priority>0.6</priority>
        \\  </url>
        \\</urlset>
        \\
    ;

    try sitemap_file.writeAll(sitemap_content);

    // Generate robots.txt
    const robots_file = try std.fs.cwd().createFile("docs/robots.txt", .{});
    defer robots_file.close();

    const robots_content =
        \\User-agent: *
        \\Allow: /
        \\Sitemap: https://your-username.github.io/wdbx-ai/sitemap.xml
        \\
        \\# Disallow build artifacts
        \\Disallow: /zig-out/
        \\Disallow: /zig-cache/
        \\Disallow: /*.zig$
        \\
    ;

    try robots_file.writeAll(robots_content);
}

/// Generate native Zig documentation using built-in tools
fn generateZigNativeDocs(allocator: std.mem.Allocator) !void {
    // Create directory for native docs
    try std.fs.cwd().makePath("docs/zig-docs");

    // Generate documentation using Zig's built-in doc generation
    // This would typically be done via: zig build-lib -femit-docs src/main.zig
    // For now, we'll create a placeholder script
    const script_file = try std.fs.cwd().createFile("docs/generate_zig_docs.sh", .{});
    defer script_file.close();

    const script_content =
        \\#!/bin/bash
        \\# Generate native Zig documentation
        \\echo "Generating native Zig documentation..."
        \\
        \\# Ensure we're in the project root
        \\cd "$(dirname "$0")/.."
        \\
        \\# Generate documentation using Zig's built-in tools
        \\if command -v zig &> /dev/null; then
        \\    echo "Using Zig compiler to generate documentation..."
        \\    zig build-lib -femit-docs=docs/zig-docs src/main.zig
        \\    
        \\    # Copy generated docs to GitHub Pages structure
        \\    if [ -d "docs/zig-docs" ]; then
        \\        echo "Native Zig documentation generated successfully!"
        \\        echo "Documentation available at: docs/zig-docs/"
        \\    else
        \\        echo "Warning: Native documentation generation may have failed"
        \\    fi
        \\else
        \\    echo "Zig compiler not found. Please install Zig to generate native documentation."
        \\    echo "Visit: https://ziglang.org/download/"
        \\fi
        \\
        \\# Generate integration with GitHub Pages
        \\echo "Integrating with GitHub Pages structure..."
        \\
        \\# Create redirect page for native docs
        \\cat > docs/native-docs.md << 'EOF'
        \\---
        \\layout: documentation
        \\title: "Native Zig Documentation"
        \\description: "Auto-generated documentation from Zig source code"
        \\permalink: /native-docs/
        \\---
        \\
        \\# Native Zig Documentation
        \\
        \\This section contains automatically generated documentation from the Zig source code using Zig's built-in documentation tools.
        \\
        \\## Features
        \\
        \\- **Auto-generated**: Documentation is automatically extracted from source code comments
        \\- **Type information**: Complete type signatures and relationships
        \\- **Cross-references**: Links between related functions and types
        \\- **Examples**: Code examples from doc comments
        \\
        \\## Accessing the Documentation
        \\
        \\The native documentation is available in the following formats:
        \\
        \\- [Browse Online](./zig-docs/) - Interactive web interface
        \\- [Source Integration](../generated/CODE_API_INDEX/) - Integrated with manual documentation
        \\
        \\## Generation Process
        \\
        \\The documentation is generated using:
        \\
        \\```bash
        \\zig build-lib -femit-docs=docs/zig-docs src/main.zig
        \\```
        \\
        \\This leverages Zig's built-in documentation generation capabilities that parse:
        \\
        \\- Doc comments (`///`)
        \\- Public declarations
        \\- Type information
        \\- Function signatures
        \\- Module structure
        \\
        \\## Integration
        \\
        \\The native documentation is integrated with our manual documentation through:
        \\
        \\- Cross-references in the [API Reference](../generated/API_REFERENCE/)
        \\- Links from the [Code Index](../generated/CODE_API_INDEX/)
        \\- Search integration in the main documentation site
        \\
        \\EOF
        \\
        \\echo "Documentation generation complete!"
        \\
    ;

    try script_file.writeAll(script_content);

    // Make script executable (on Unix systems)
    if (std.builtin.os.tag != .windows) {
        _ = std.fs.cwd().chmod("docs/generate_zig_docs.sh", 0o755) catch {};
    }

    // Create a README for the zig-docs directory
    const readme_file = try std.fs.cwd().createFile("docs/zig-docs/README.md", .{});
    defer readme_file.close();

    const readme_content =
        \\# Native Zig Documentation
        \\
        \\This directory contains automatically generated documentation from the Zig source code.
        \\
        \\## Generation
        \\
        \\To generate the documentation:
        \\
        \\```bash
        \\cd docs
        \\./generate_zig_docs.sh
        \\```
        \\
        \\Or manually:
        \\
        \\```bash
        \\zig build-lib -femit-docs=docs/zig-docs src/main.zig
        \\```
        \\
        \\## Integration with GitHub Pages
        \\
        \\The generated documentation integrates with the main GitHub Pages site through:
        \\
        \\1. **Jekyll integration**: Documentation pages have proper frontmatter
        \\2. **Navigation**: Links are included in the main site navigation
        \\3. **Search**: Content is indexed by the site search functionality
        \\4. **Cross-references**: Links between manual and generated documentation
        \\
        \\## Features
        \\
        \\- Complete API coverage from source code
        \\- Interactive browsing
        \\- Type information and signatures
        \\- Cross-references between modules
        \\- Example code from doc comments
        \\
    ;

    try readme_file.writeAll(readme_content);

    std.log.info("Native Zig documentation setup created. Run docs/generate_zig_docs.sh to generate.", .{});
}

/// Generate README redirect for GitHub
fn generateReadmeRedirect(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/README.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "WDBX-AI Documentation"
        \\description: "High-performance vector database with AI capabilities - Complete documentation"
        \\permalink: /
        \\---
        \\
        \\# WDBX-AI Documentation
        \\
        \\Welcome to the comprehensive documentation for WDBX-AI, a high-performance vector database with integrated AI capabilities.
        \\
        \\## 🚀 Quick Navigation
        \\
        \\<div class="quick-nav">
        \\  <div class="nav-card">
        \\    <h3><a href="./generated/API_REFERENCE/">📘 API Reference</a></h3>
        \\    <p>Complete API documentation with examples and detailed function signatures.</p>
        \\  </div>
        \\  
        \\  <div class="nav-card">
        \\    <h3><a href="./generated/EXAMPLES/">💡 Examples</a></h3>
        \\    <p>Practical examples and tutorials to get you started quickly.</p>
        \\  </div>
        \\  
        \\  <div class="nav-card">
        \\    <h3><a href="./generated/MODULE_REFERENCE/">📦 Module Reference</a></h3>
        \\    <p>Detailed module documentation and architecture overview.</p>
        \\  </div>
        \\  
        \\  <div class="nav-card">
        \\    <h3><a href="./generated/PERFORMANCE_GUIDE/">⚡ Performance Guide</a></h3>
        \\    <p>Optimization tips, benchmarks, and performance best practices.</p>
        \\  </div>
        \\</div>
        \\
        \\## 📖 What's Inside
        \\
        \\### Core Documentation
        \\- **[API Reference](./generated/API_REFERENCE/)** - Complete function and type documentation
        \\- **[Module Reference](./generated/MODULE_REFERENCE/)** - Module structure and relationships
        \\- **[Examples](./generated/EXAMPLES/)** - Practical usage examples and tutorials
        \\- **[Performance Guide](./generated/PERFORMANCE_GUIDE/)** - Optimization and benchmarking
        \\- **[Definitions](./generated/DEFINITIONS_REFERENCE/)** - Comprehensive glossary and concepts
        \\
        \\### Developer Resources
        \\- **[Code Index](./generated/CODE_API_INDEX/)** - Auto-generated API index from source
        \\- **[Native Docs](./native-docs/)** - Zig compiler-generated documentation
        \\- **[Search](./index.html)** - Interactive documentation browser
        \\
        \\## 🔍 Features
        \\
        \\- **🚄 High Performance**: Optimized vector operations with SIMD support
        \\- **🧠 AI Integration**: Built-in neural networks and machine learning
        \\- **🗄️ Vector Database**: Efficient storage and similarity search
        \\- **🔌 Plugin System**: Extensible architecture for custom functionality
        \\- **📊 Analytics**: Performance monitoring and optimization tools
        \\
        \\## 🛠️ Getting Started
        \\
        \\1. **Installation**: Check the [Examples](./generated/EXAMPLES/) for setup instructions
        \\2. **Quick Start**: Follow the [basic usage examples](./generated/EXAMPLES/#quick-start)
        \\3. **API Learning**: Explore the [API Reference](./generated/API_REFERENCE/) for detailed function documentation
        \\4. **Optimization**: Read the [Performance Guide](./generated/PERFORMANCE_GUIDE/) for best practices
        \\
        \\## 📚 Documentation Types
        \\
        \\This documentation is generated using multiple approaches:
        \\
        \\### Manual Documentation
        \\- Curated guides and examples
        \\- Performance analysis and optimization tips
        \\- Comprehensive concept explanations
        \\- Best practices and design patterns
        \\
        \\### Auto-Generated Documentation
        \\- Source code scanning for public APIs
        \\- Zig compiler documentation extraction
        \\- Type information and signatures
        \\- Cross-references and relationships
        \\
        \\## 🔗 External Resources
        \\
        \\- **[GitHub Repository](https://github.com/your-username/wdbx-ai)** - Source code and issues
        \\- **[Zig Language](https://ziglang.org/)** - Learn about the Zig programming language
        \\- **[Vector Databases](./generated/DEFINITIONS_REFERENCE/#vector-database)** - Learn about vector database concepts
        \\
        \\## 📧 Support
        \\
        \\- **Issues**: [Report bugs or request features](https://github.com/your-username/wdbx-ai/issues)
        \\- **Discussions**: [Join community discussions](https://github.com/your-username/wdbx-ai/discussions)
        \\- **Documentation**: [Improve documentation](https://github.com/your-username/wdbx-ai/issues/new?title=Documentation%20Improvement)
        \\
        \\---
        \\
        \\<style>
        \\.quick-nav {
        \\  display: grid;
        \\  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        \\  gap: 1rem;
        \\  margin: 2rem 0;
        \\}
        \\
        \\.nav-card {
        \\  border: 1px solid #e1e4e8;
        \\  border-radius: 8px;
        \\  padding: 1.5rem;
        \\  background: #f6f8fa;
        \\}
        \\
        \\.nav-card h3 {
        \\  margin-top: 0;
        \\  margin-bottom: 0.5rem;
        \\}
        \\
        \\.nav-card h3 a {
        \\  text-decoration: none;
        \\  color: #0366d6;
        \\}
        \\
        \\.nav-card p {
        \\  margin-bottom: 0;
        \\  color: #586069;
        \\  font-size: 0.9rem;
        \\}
        \\
        \\@media (prefers-color-scheme: dark) {
        \\  .nav-card {
        \\    border-color: #30363d;
        \\    background: #21262d;
        \\  }
        \\  
        \\  .nav-card h3 a {
        \\    color: #58a6ff;
        \\  }
        \\  
        \\  .nav-card p {
        \\    color: #8b949e;
        \\  }
        \\}
        \\</style>
        \\
    ;

    try file.writeAll(content);
}

/// Generate comprehensive module documentation
fn generateModuleDocs(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/MODULE_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Module Reference"
        \\description: "Comprehensive reference for all WDBX-AI modules and components"
        \\---
        \\
        \\# WDBX-AI Module Reference
        \\
        \\## 📦 Core Modules
        \\
        \\### `abi` - Main Module
        \\The primary module containing all core functionality.
        \\
        \\#### Key Components:
        \\- **Database Engine**: High-performance vector database with HNSW indexing
        \\- **AI System**: Neural networks and machine learning capabilities
        \\- **SIMD Operations**: Optimized vector operations
        \\- **Plugin System**: Extensible architecture for custom functionality
        \\
        \\### `abi.database` - Database Module
        \\Vector database operations and management.
        \\
        \\#### Functions:
        \\```zig
        \\// Initialize database
        \\pub fn init(allocator: Allocator, config: DatabaseConfig) !Database
        \\
        \\// Insert vector
        \\pub fn insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64
        \\
        \\// Search vectors
        \\pub fn search(self: *Database, query: []const f32, k: usize) ![]SearchResult
        \\
        \\// Update vector
        \\pub fn update(self: *Database, id: u64, vector: []const f32) !void
        \\
        \\// Delete vector
        \\pub fn delete(self: *Database, id: u64) !void
        \\```
        \\
        \\### `abi.ai` - AI Module
        \\Artificial intelligence and machine learning capabilities.
        \\
        \\#### Functions:
        \\```zig
        \\// Create neural network
        \\pub fn createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork
        \\
        \\// Train network
        \\pub fn train(self: *NeuralNetwork, data: []const TrainingData) !f32
        \\
        \\// Predict/Infer
        \\pub fn predict(self: *NeuralNetwork, input: []const f32) ![]f32
        \\
        \\// Enhanced agent operations
        \\pub fn createAgent(allocator: Allocator, config: AgentConfig) !EnhancedAgent
        \\```
        \\
        \\### `abi.simd` - SIMD Module
        \\SIMD-optimized vector operations.
        \\
        \\#### Functions:
        \\```zig
        \\// Vector addition
        \\pub fn add(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector subtraction
        \\pub fn subtract(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector multiplication
        \\pub fn multiply(result: []f32, a: []const f32, b: []const f32) void
        \\
        \\// Vector normalization
        \\pub fn normalize(result: []f32, input: []const f32) void
        \\```
        \\
        \\### `abi.plugins` - Plugin System
        \\Extensible plugin architecture.
        \\
        \\#### Functions:
        \\```zig
        \\// Load plugin
        \\pub fn loadPlugin(path: []const u8) !Plugin
        \\
        \\// Register plugin
        \\pub fn registerPlugin(plugin: Plugin) !void
        \\
        \\// Execute plugin function
        \\pub fn executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8
        \\```
        \\
        \\## 🔧 Configuration Types
        \\
        \\### DatabaseConfig
        \\```zig
        \\pub const DatabaseConfig = struct {
        \\    max_vectors: usize = 1000000,
        \\    vector_dimension: usize = 128,
        \\    index_type: IndexType = .hnsw,
        \\    storage_path: ?[]const u8 = null,
        \\    enable_caching: bool = true,
        \\    cache_size: usize = 1024 * 1024, // 1MB
        \\};
        \\```
        \\
        \\### NetworkConfig
        \\```zig
        \\pub const NetworkConfig = struct {
        \\    input_size: usize,
        \\    hidden_sizes: []const usize,
        \\    output_size: usize,
        \\    activation: ActivationType = .relu,
        \\    learning_rate: f32 = 0.01,
        \\    batch_size: usize = 32,
        \\};
        \\```
        \\
        \\## 📊 Performance Characteristics
        \\
        \\| Operation | Performance | Memory Usage |
        \\|-----------|-------------|--------------|
        \\| Vector Insert | ~2.5ms (1000 vectors) | ~512 bytes/vector |
        \\| Vector Search | ~13ms (10k vectors, k=10) | ~160 bytes/result |
        \\| Neural Training | ~30μs/iteration | ~1MB/network |
        \\| SIMD Operations | ~3μs (2048 elements) | ~16KB/batch |
        \\
        \\## 🚀 Usage Examples
        \\
        \\### Basic Database Usage
        \\```zig
        \\const std = @import("std");
        \\const abi = @import("abi");
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Initialize database
        \\    const config = abi.DatabaseConfig{
        \\        .max_vectors = 10000,
        \\        .vector_dimension = 128,
        \\    };
        \\    var db = try abi.database.init(allocator, config);
        \\    defer db.deinit();
        \\
        \\    // Insert vectors
        \\    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions
        \\    const id = try db.insert(&vector, "sample_data");
        \\
        \\    // Search for similar vectors
        \\    const results = try db.search(&vector, 10);
        \\    defer allocator.free(results);
        \\
        \\    std.log.info("Found {} similar vectors", .{results.len});
        \\}
        \\```
        \\
        \\### Neural Network Training
        \\```zig
        \\const config = abi.NetworkConfig{
        \\    .input_size = 128,
        \\    .hidden_sizes = &[_]usize{64, 32},
        \\    .output_size = 10,
        \\    .learning_rate = 0.01,
        \\};
        \\
        \\var network = try abi.ai.createNetwork(allocator, config);
        \\defer network.deinit();
        \\
        \\// Training data
        \\const training_data = [_]abi.TrainingData{
        \\    .{ .input = &input1, .output = &output1 },
        \\    .{ .input = &input2, .output = &output2 },
        \\};
        \\
        \\// Train network
        \\const loss = try network.train(&training_data);
        \\std.log.info("Training loss: {}", .{loss});
        \\```
        \\
        \\## 🔍 Error Handling
        \\
        \\All functions return appropriate error types:
        \\- `DatabaseError` - Database-specific errors
        \\- `AIError` - AI/ML operation errors
        \\- `SIMDError` - SIMD operation errors
        \\- `PluginError` - Plugin system errors
        \\
        \\## 📈 Performance Tips
        \\
        \\1. **Use appropriate vector dimensions** - 128-512 dimensions typically optimal
        \\2. **Batch operations** - Group multiple operations for better performance
        \\3. **Enable caching** - Significant performance improvement for repeated queries
        \\4. **SIMD optimization** - Automatically enabled for supported operations
        \\5. **Memory management** - Use arena allocators for bulk operations
        \\
    ;

    try file.writeAll(content);
}

/// Generate API reference documentation
fn generateApiReference(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/API_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "API Reference"
        \\description: "Complete API reference for WDBX-AI with detailed function documentation"
        \\---
        \\
        \\# WDBX-AI API Reference
        \\
        \\## 🗄️ Database API
        \\
        \\### Database
        \\Main database interface for vector operations.
        \\
        \\#### Methods
        \\
        \\##### `init(allocator: Allocator, config: DatabaseConfig) !Database`
        \\Initialize a new database instance.
        \\
        \\**Parameters:**
        \\- `allocator`: Memory allocator to use
        \\- `config`: Database configuration
        \\
        \\**Returns:** Initialized database instance
        \\
        \\**Errors:** `DatabaseError.OutOfMemory`, `DatabaseError.InvalidConfig`
        \\
        \\##### `insert(self: *Database, vector: []const f32, metadata: ?[]const u8) !u64`
        \\Insert a vector into the database.
        \\
        \\**Parameters:**
        \\- `vector`: Vector data (must match configured dimension)
        \\- `metadata`: Optional metadata string
        \\
        \\**Returns:** Unique ID for the inserted vector
        \\
        \\**Performance:** ~2.5ms for 1000 vectors
        \\
        \\##### `search(self: *Database, query: []const f32, k: usize) ![]SearchResult`
        \\Search for k nearest neighbors.
        \\
        \\**Parameters:**
        \\- `query`: Query vector
        \\- `k`: Number of results to return
        \\
        \\**Returns:** Array of search results (caller must free)
        \\
        \\**Performance:** ~13ms for 10k vectors, k=10
        \\
        \\## 🧠 AI API
        \\
        \\### NeuralNetwork
        \\Neural network for machine learning operations.
        \\
        \\#### Methods
        \\
        \\##### `createNetwork(allocator: Allocator, config: NetworkConfig) !NeuralNetwork`
        \\Create a new neural network.
        \\
        \\**Parameters:**
        \\- `allocator`: Memory allocator
        \\- `config`: Network configuration
        \\
        \\**Returns:** Initialized neural network
        \\
        \\##### `train(self: *NeuralNetwork, data: []const TrainingData) !f32`
        \\Train the neural network.
        \\
        \\**Parameters:**
        \\- `data`: Training data array
        \\
        \\**Returns:** Final training loss
        \\
        \\##### `predict(self: *NeuralNetwork, input: []const f32) ![]f32`
        \\Make predictions using the trained network.
        \\
        \\**Parameters:**
        \\- `input`: Input vector
        \\
        \\**Returns:** Prediction results (caller must free)
        \\
        \\## ⚡ SIMD API
        \\
        \\### Vector Operations
        \\SIMD-optimized vector operations.
        \\
        \\#### Functions
        \\
        \\##### `add(result: []f32, a: []const f32, b: []const f32) void`
        \\Add two vectors element-wise.
        \\
        \\**Parameters:**
        \\- `result`: Output vector (must be same size as inputs)
        \\- `a`: First input vector
        \\- `b`: Second input vector
        \\
        \\**Performance:** ~3μs for 2048 elements
        \\
        \\##### `normalize(result: []f32, input: []const f32) void`
        \\Normalize a vector to unit length.
        \\
        \\**Parameters:**
        \\- `result`: Output normalized vector
        \\- `input`: Input vector to normalize
        \\
        \\## 🔌 Plugin API
        \\
        \\### Plugin System
        \\Extensible plugin architecture.
        \\
        \\#### Functions
        \\
        \\##### `loadPlugin(path: []const u8) !Plugin`
        \\Load a plugin from file.
        \\
        \\**Parameters:**
        \\- `path`: Path to plugin file
        \\
        \\**Returns:** Loaded plugin instance
        \\
        \\##### `executePlugin(plugin: Plugin, function: []const u8, args: []const u8) ![]u8`
        \\Execute a plugin function.
        \\
        \\**Parameters:**
        \\- `plugin`: Plugin instance
        \\- `function`: Function name to execute
        \\- `args`: JSON-encoded arguments
        \\
        \\**Returns:** JSON-encoded result (caller must free)
        \\
        \\## 📊 Data Types
        \\
        \\### SearchResult
        \\```zig
        \\pub const SearchResult = struct {
        \\    id: u64,
        \\    distance: f32,
        \\    metadata: ?[]const u8,
        \\};
        \\```
        \\
        \\### TrainingData
        \\```zig
        \\pub const TrainingData = struct {
        \\    input: []const f32,
        \\    output: []const f32,
        \\};
        \\```
        \\
        \\## ⚠️ Error Types
        \\
        \\### DatabaseError
        \\```zig
        \\pub const DatabaseError = error{
        \\    OutOfMemory,
        \\    InvalidConfig,
        \\    VectorDimensionMismatch,
        \\    IndexNotFound,
        \\    StorageError,
        \\};
        \\```
        \\
        \\### AIError
        \\```zig
        \\pub const AIError = error{
        \\    InvalidNetworkConfig,
        \\    TrainingDataEmpty,
        \\    ConvergenceFailed,
        \\    InvalidInputSize,
        \\};
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate usage examples
fn generateExamples(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/EXAMPLES.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Examples & Tutorials"
        \\description: "Practical examples and tutorials for using WDBX-AI effectively"
        \\---
        \\
        \\# WDBX-AI Usage Examples
        \\
        \\## 🚀 Quick Start
        \\
        \\### Basic Vector Database
        \\```zig
        \\const std = @import("std");
        \\const abi = @import("abi");
        \\
        \\pub fn main() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Initialize database
        \\    const config = abi.DatabaseConfig{
        \\        .max_vectors = 10000,
        \\        .vector_dimension = 128,
        \\        .enable_caching = true,
        \\    };
        \\    var db = try abi.database.init(allocator, config);
        \\    defer db.deinit();
        \\
        \\    // Insert sample vectors
        \\    for (0..100) |i| {
        \\        var vector: [128]f32 = undefined;
        \\        for (&vector, 0..) |*v, j| {
        \\            v.* = @as(f32, @floatFromInt(i + j)) * 0.1;
        \\        }
        \\        const id = try db.insert(&vector, "vector_{}");
        \\        std.log.info("Inserted vector with ID: {}", .{id});
        \\    }
        \\
        \\    // Search for similar vectors
        \\    const query = [_]f32{1.0} ** 128;
        \\    const results = try db.search(&query, 5);
        \\    defer allocator.free(results);
        \\
        \\    std.log.info("Found {} similar vectors:", .{results.len});
        \\    for (results, 0..) |result, i| {
        \\        std.log.info("  {}: ID={}, Distance={}", .{ i, result.id, result.distance });
        \\    }
        \\}
        \\```
        \\
        \\## 🧠 Machine Learning Pipeline
        \\
        \\### Neural Network Training
        \\```zig
        \\pub fn trainModel() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Create network
        \\    const config = abi.NetworkConfig{
        \\        .input_size = 128,
        \\        .hidden_sizes = &[_]usize{64, 32},
        \\        .output_size = 10,
        \\        .learning_rate = 0.01,
        \\        .batch_size = 32,
        \\    };
        \\    var network = try abi.ai.createNetwork(allocator, config);
        \\    defer network.deinit();
        \\
        \\    // Generate training data
        \\    var training_data = std.ArrayList(abi.TrainingData).init(allocator);
        \\    defer training_data.deinit();
        \\
        \\    for (0..1000) |i| {
        \\        var input: [128]f32 = undefined;
        \\        var output: [10]f32 = undefined;
        \\
        \\        // Generate random input
        \\        for (&input) |*v| {
        \\            v.* = std.rand.DefaultPrng.init(@as(u64, i)).random().float(f32);
        \\        }
        \\
        \\        // Generate target output (one-hot encoding)
        \\        @memset(&output, 0);
        \\        output[i % 10] = 1.0;
        \\
        \\        try training_data.append(abi.TrainingData{
        \\            .input = &input,
        \\            .output = &output,
        \\        });
        \\    }
        \\
        \\    // Train network
        \\    const loss = try network.train(training_data.items);
        \\    std.log.info("Training completed with loss: {}", .{loss});
        \\
        \\    // Test prediction
        \\    const test_input = [_]f32{0.5} ** 128;
        \\    const prediction = try network.predict(&test_input);
        \\    defer allocator.free(prediction);
        \\
        \\    std.log.info("Prediction: {any}", .{prediction});
        \\}
        \\```
        \\
        \\## ⚡ SIMD Operations
        \\
        \\### Vector Processing
        \\```zig
        \\pub fn vectorProcessing() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Allocate vectors
        \\    const size = 2048;
        \\    const a = try allocator.alloc(f32, size);
        \\    defer allocator.free(a);
        \\    const b = try allocator.alloc(f32, size);
        \\    defer allocator.free(b);
        \\    const result = try allocator.alloc(f32, size);
        \\    defer allocator.free(result);
        \\
        \\    // Initialize vectors
        \\    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i));
        \\    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2));
        \\
        \\    // SIMD operations
        \\    const start_time = std.time.nanoTimestamp();
        \\
        \\    abi.simd.add(result, a, b);
        \\    abi.simd.subtract(result, result, a);
        \\    abi.simd.multiply(result, result, b);
        \\    abi.simd.normalize(result, result);
        \\
        \\    const end_time = std.time.nanoTimestamp();
        \\    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
        \\
        \\    std.log.info("SIMD operations completed in {}ms", .{duration});
        \\    std.log.info("Result sample: [{}, {}, {}]", .{ result[0], result[1], result[2] });
        \\}
        \\```
        \\
        \\## 🔌 Plugin System
        \\
        \\### Custom Plugin
        \\```zig
        \\// plugin_example.zig
        \\const std = @import("std");
        \\
        \\export fn process_data(input: [*c]const u8, input_len: usize, output: [*c]u8, output_len: *usize) c_int {
        \\    // Process input data
        \\    const input_slice = input[0..input_len];
        \\
        \\    // Example: convert to uppercase
        \\    var result = std.ArrayList(u8).init(std.heap.page_allocator);
        \\    defer result.deinit();
        \\
        \\    for (input_slice) |byte| {
        \\        if (byte >= 'a' and byte <= 'z') {
        \\            try result.append(byte - 32);
        \\        } else {
        \\            try result.append(byte);
        \\        }
        \\    }
        \\
        \\    // Copy result to output
        \\    if (result.items.len > output_len.*) {
        \\        return -1; // Buffer too small
        \\    }
        \\
        \\    @memcpy(output[0..result.items.len], result.items);
        \\    output_len.* = result.items.len;
        \\    return 0; // Success
        \\}
        \\
        \\// Using the plugin
        \\pub fn usePlugin() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    // Load plugin
        \\    const plugin = try abi.plugins.loadPlugin("plugin_example.zig");
        \\    defer plugin.deinit();
        \\
        \\    // Execute plugin function
        \\    const input = "hello world";
        \\    const result = try abi.plugins.executePlugin(plugin, "process_data", input);
        \\    defer allocator.free(result);
        \\
        \\    std.log.info("Plugin result: {s}", .{result});
        \\}
        \\```
        \\
        \\## 🎯 Performance Optimization
        \\
        \\### Batch Operations
        \\```zig
        \\pub fn batchOperations() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    var db = try abi.database.init(allocator, abi.DatabaseConfig{});
        \\    defer db.deinit();
        \\
        \\    // Batch insert
        \\    const batch_size = 1000;
        \\    var vectors = try allocator.alloc([]f32, batch_size);
        \\    defer {
        \\        for (vectors) |vec| allocator.free(vec);
        \\        allocator.free(vectors);
        \\    }
        \\
        \\    // Generate batch data
        \\    for (vectors, 0..) |*vec, i| {
        \\        vec.* = try allocator.alloc(f32, 128);
        \\        for (vec.*, 0..) |*v, j| {
        \\            v.* = @as(f32, @floatFromInt(i + j)) * 0.01;
        \\        }
        \\    }
        \\
        \\    // Insert batch
        \\    const start_time = std.time.nanoTimestamp();
        \\    for (vectors) |vec| {
        \\        _ = try db.insert(vec, null);
        \\    }
        \\    const end_time = std.time.nanoTimestamp();
        \\
        \\    const duration = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
        \\    const throughput = @as(f64, @floatFromInt(batch_size)) / (duration / 1000.0);
        \\
        \\    std.log.info("Batch insert: {} vectors in {}ms", .{ batch_size, duration });
        \\    std.log.info("Throughput: {} vectors/sec", .{throughput});
        \\}
        \\```
        \\
        \\## 🔧 Error Handling
        \\
        \\### Comprehensive Error Handling
        \\```zig
        \\pub fn robustOperations() !void {
        \\    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        \\    defer _ = gpa.deinit();
        \\    const allocator = gpa.allocator();
        \\
        \\    var db = abi.database.init(allocator, abi.DatabaseConfig{}) catch |err| switch (err) {
        \\        error.OutOfMemory => {
        \\            std.log.err("Failed to allocate memory for database");
        \\            return;
        \\        },
        \\        error.InvalidConfig => {
        \\            std.log.err("Invalid database configuration");
        \\            return;
        \\        },
        \\        else => return err,
        \\    };
        \\    defer db.deinit();
        \\
        \\    // Safe vector operations
        \\    const vector = [_]f32{1.0, 2.0, 3.0} ** 43; // 128 dimensions
        \\
        \\    const id = db.insert(&vector, "test") catch |err| switch (err) {
        \\        error.VectorDimensionMismatch => {
        \\            std.log.err("Vector dimension mismatch");
        \\            return;
        \\        },
        \\        error.StorageError => {
        \\            std.log.err("Storage operation failed");
        \\            return;
        \\        },
        \\        else => return err,
        \\    };
        \\
        \\    std.log.info("Successfully inserted vector with ID: {}", .{id});
        \\}
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate performance guide
fn generatePerformanceGuide(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/PERFORMANCE_GUIDE.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Performance Guide"
        \\description: "Comprehensive performance optimization guide with benchmarks and best practices"
        \\---
        \\
        \\# WDBX-AI Performance Guide
        \\
        \\## 🚀 Performance Characteristics
        \\
        \\### Database Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Single Insert | ~2.5ms | ~512 bytes | 128-dim vectors |
        \\| Batch Insert (100) | ~40ms | ~51KB | 100 vectors |
        \\| Batch Insert (1000) | ~400ms | ~512KB | 1000 vectors |
        \\| Search (k=10) | ~13ms | ~1.6KB | 10k vectors |
        \\| Search (k=100) | ~14ms | ~16KB | 10k vectors |
        \\| Update | ~1ms | ~512 bytes | Single vector |
        \\| Delete | ~0.5ms | ~0 bytes | Single vector |
        \\
        \\### AI/ML Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Network Creation | ~1ms | ~1MB | 128→64→32→10 |
        \\| Training Iteration | ~30μs | ~1MB | Batch size 32 |
        \\| Prediction | ~10μs | ~1KB | Single input |
        \\| Batch Prediction | ~100μs | ~10KB | 100 inputs |
        \\
        \\### SIMD Operations
        \\| Operation | Performance | Memory | Notes |
        \\|-----------|-------------|--------|-------|
        \\| Vector Add (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Multiply (2048) | ~3μs | ~16KB | SIMD optimized |
        \\| Vector Normalize (2048) | ~5μs | ~16KB | Includes sqrt |
        \\| Matrix Multiply (64x64) | ~50μs | ~32KB | SIMD optimized |
        \\
        \\## ⚡ Optimization Strategies
        \\
        \\### 1. Memory Management
        \\```zig
        \\// Use arena allocators for bulk operations
        \\var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        \\defer arena.deinit();
        \\const allocator = arena.allocator();
        \\
        \\// Pre-allocate buffers for repeated operations
        \\const buffer_size = 1024 * 1024; // 1MB
        \\const buffer = try allocator.alloc(u8, buffer_size);
        \\defer allocator.free(buffer);
        \\```
        \\
        \\### 2. Batch Processing
        \\```zig
        \\// Process vectors in batches for better performance
        \\const BATCH_SIZE = 100;
        \\for (0..total_vectors / BATCH_SIZE) |batch| {
        \\    const start = batch * BATCH_SIZE;
        \\    const end = @min(start + BATCH_SIZE, total_vectors);
        \\    
        \\    // Process batch
        \\    for (vectors[start..end]) |vector| {
        \\        _ = try db.insert(vector, null);
        \\    }
        \\}
        \\```
        \\
        \\### 3. SIMD Optimization
        \\```zig
        \\// Use SIMD operations for vector processing
        \\const VECTOR_SIZE = 128;
        \\const SIMD_SIZE = 4; // Process 4 elements at once
        \\
        \\var i: usize = 0;
        \\while (i + SIMD_SIZE <= VECTOR_SIZE) : (i += SIMD_SIZE) {
        \\    const va = @as(@Vector(4, f32), a[i..][0..4].*);
        \\    const vb = @as(@Vector(4, f32), b[i..][0..4].*);
        \\    const result = va + vb;
        \\    @memcpy(output[i..][0..4], @as([4]f32, result)[0..]);
        \\}
        \\```
        \\
        \\### 4. Caching Strategy
        \\```zig
        \\// Enable database caching for repeated queries
        \\const config = abi.DatabaseConfig{
        \\    .enable_caching = true,
        \\    .cache_size = 1024 * 1024, // 1MB cache
        \\};
        \\
        \\// Use LRU cache for frequently accessed data
        \\var cache = std.HashMap(u64, []f32, std.hash_map.default_hash_fn(u64), std.hash_map.default_eql_fn(u64)).init(allocator);
        \\defer {
        \\    var iterator = cache.iterator();
        \\    while (iterator.next()) |entry| {
        \\        allocator.free(entry.value_ptr.*);
        \\    }
        \\    cache.deinit();
        \\}
        \\```
        \\
        \\## 📊 Benchmarking
        \\
        \\### Running Benchmarks
        \\```bash
        \\# Run all benchmarks
        \\zig build benchmark
        \\
        \\# Run specific benchmark types
        \\zig build benchmark-db      # Database performance
        \\zig build benchmark-neural  # AI/ML performance
        \\zig build benchmark-simple  # General performance
        \\
        \\# Run with profiling
        \\zig build profile
        \\```
        \\
        \\### Custom Benchmarking
        \\```zig
        \\pub fn benchmarkOperation() !void {
        \\    const iterations = 1000;
        \\    var times = try allocator.alloc(u64, iterations);
        \\    defer allocator.free(times);
        \\
        \\    // Warm up
        \\    for (0..10) |_| {
        \\        // Perform operation
        \\    }
        \\
        \\    // Benchmark
        \\    for (times, 0..) |*time, i| {
        \\        const start = std.time.nanoTimestamp();
        \\        
        \\        // Perform operation
        \\        
        \\        const end = std.time.nanoTimestamp();
        \\        time.* = end - start;
        \\    }
        \\
        \\    // Calculate statistics
        \\    std.sort.heap(u64, times, {}, comptime std.sort.asc(u64));
        \\    const p50 = times[iterations / 2];
        \\    const p95 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.95))];
        \\    const p99 = times[@as(usize, @intFromFloat(@as(f64, @floatFromInt(iterations)) * 0.99))];
        \\
        \\    std.log.info("P50: {}ns, P95: {}ns, P99: {}ns", .{ p50, p95, p99 });
        \\}
        \\```
        \\
        \\## 🔍 Profiling Tools
        \\
        \\### Memory Profiling
        \\```zig
        \\// Enable memory tracking
        \\const memory_tracker = abi.memory_tracker.init(allocator);
        \\defer memory_tracker.deinit();
        \\
        \\// Track allocations
        \\memory_tracker.startTracking();
        \\
        \\// Perform operations
        \\
        \\// Get memory statistics
        \\const stats = memory_tracker.getStats();
        \\std.log.info("Peak memory: {} bytes", .{stats.peak_memory});
        \\std.log.info("Total allocations: {}", .{stats.total_allocations});
        \\```
        \\
        \\### Performance Profiling
        \\```zig
        \\// Use performance profiler
        \\const profiler = abi.performance_profiler.init(allocator);
        \\defer profiler.deinit();
        \\
        \\// Start profiling
        \\profiler.startProfiling("operation_name");
        \\
        \\// Perform operation
        \\
        \\// Stop profiling
        \\profiler.stopProfiling("operation_name");
        \\
        \\// Get results
        \\const results = profiler.getResults();
        \\for (results) |result| {
        \\    std.log.info("{}: {}ms", .{ result.name, result.duration_ms });
        \\}
        \\```
        \\
        \\## 🎯 Performance Tips
        \\
        \\### 1. Vector Dimensions
        \\- **Optimal range**: 128-512 dimensions
        \\- **Too small**: Poor representation quality
        \\- **Too large**: Increased memory and computation
        \\
        \\### 2. Batch Sizes
        \\- **Database inserts**: 100-1000 vectors per batch
        \\- **Neural training**: 32-128 samples per batch
        \\- **SIMD operations**: 1024-4096 elements per batch
        \\
        \\### 3. Memory Allocation
        \\- **Use arena allocators** for bulk operations
        \\- **Pre-allocate buffers** for repeated operations
        \\- **Enable caching** for frequently accessed data
        \\
        \\### 4. SIMD Usage
        \\- **Automatic optimization** for supported operations
        \\- **Vector size alignment** for best performance
        \\- **Batch processing** for maximum throughput
        \\
        \\## 📈 Performance Monitoring
        \\
        \\### Real-time Metrics
        \\```zig
        \\// Monitor performance in real-time
        \\const monitor = abi.performance_monitor.init(allocator);
        \\defer monitor.deinit();
        \\
        \\// Start monitoring
        \\monitor.startMonitoring();
        \\
        \\// Perform operations
        \\
        \\// Get metrics
        \\const metrics = monitor.getMetrics();
        \\std.log.info("Operations/sec: {}", .{metrics.operations_per_second});
        \\std.log.info("Average latency: {}ms", .{metrics.average_latency_ms});
        \\std.log.info("Memory usage: {}MB", .{metrics.memory_usage_mb});
        \\```
        \\
        \\### Performance Regression Detection
        \\```zig
        \\// Compare with baseline performance
        \\const baseline = try loadBaselinePerformance("baseline.json");
        \\const current = try measureCurrentPerformance();
        \\
        \\const regression_threshold = 0.05; // 5% regression
        \\if (current.avg_latency > baseline.avg_latency * (1.0 + regression_threshold)) {
        \\    std.log.warn("Performance regression detected!");
        \\    std.log.warn("Baseline: {}ms, Current: {}ms", .{ baseline.avg_latency, current.avg_latency });
        \\}
        \\```
        \\
    ;

    try file.writeAll(content);
}

/// Generate comprehensive definitions reference documentation
fn generateDefinitionsReference(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/DEFINITIONS_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Definitions Reference"
        \\description: "Comprehensive glossary and concepts for WDBX-AI technology"
        \\---
        \\
        \\# WDBX-AI Definitions Reference
        \\
        \\## 📚 Core Concepts
        \\
        \\### Vector Database
        \\A specialized database designed to store, index, and search high-dimensional vectors efficiently. Unlike traditional databases that store scalar values, vector databases are optimized for similarity search operations using metrics like cosine similarity, Euclidean distance, or dot product.
        \\
        \\**Key characteristics:**
        \\- **High-dimensional storage**: Handles vectors with hundreds or thousands of dimensions
        \\- **Similarity search**: Finds vectors most similar to a query vector
        \\- **Indexing algorithms**: Uses specialized indices like HNSW, IVF, or LSH for fast retrieval
        \\- **Scalability**: Designed to handle millions or billions of vectors
        \\
        \\### Embeddings
        \\Dense vector representations of data (text, images, audio, etc.) that capture semantic meaning in a continuous vector space. Embeddings are typically generated by machine learning models and enable similarity comparisons between different data points.
        \\
        \\**Examples:**
        \\- **Text embeddings**: "cat" and "kitten" have similar vector representations
        \\- **Image embeddings**: Photos of similar objects cluster together in vector space
        \\- **Audio embeddings**: Similar sounds or music pieces have nearby representations
        \\
        \\### HNSW (Hierarchical Navigable Small World)
        \\A graph-based indexing algorithm that builds a multi-layered network of connections between vectors. It provides excellent performance for approximate nearest neighbor search with logarithmic time complexity.
        \\
        \\**Structure:**
        \\- **Bottom layer**: Contains all vectors with short-range connections
        \\- **Upper layers**: Contain subsets of vectors with long-range connections
        \\- **Navigation**: Search starts from top layer and progressively moves down
        \\
        \\**Parameters:**
        \\- `M`: Maximum number of connections per vector (typically 16-64)
        \\- `efConstruction`: Size of candidate set during construction (typically 200-800)
        \\- `efSearch`: Size of candidate set during search (affects recall vs speed trade-off)
        \\
        \\## 🧠 AI & Machine Learning
        \\
        \\### Neural Network
        \\A computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines the strength of the signal passed between neurons.
        \\
        \\**Architecture components:**
        \\- **Input layer**: Receives raw data features
        \\- **Hidden layers**: Process and transform the input through weighted connections
        \\- **Output layer**: Produces final predictions or classifications
        \\- **Activation functions**: Non-linear functions that introduce complexity (ReLU, Sigmoid, Tanh)
        \\
        \\### Backpropagation
        \\The fundamental algorithm for training neural networks. It calculates gradients of the loss function with respect to each weight by propagating errors backwards through the network, then updates weights to minimize the loss.
        \\
        \\**Process:**
        \\1. **Forward pass**: Input flows through network to produce output
        \\2. **Loss calculation**: Compare output to target, calculate error
        \\3. **Backward pass**: Propagate error gradients back through layers
        \\4. **Weight update**: Adjust weights using gradients and learning rate
        \\
        \\### Gradient Descent
        \\An optimization algorithm that iteratively adjusts model parameters to minimize a loss function. It moves in the direction of steepest descent of the loss landscape.
        \\
        \\**Variants:**
        \\- **Batch gradient descent**: Uses entire dataset for each update
        \\- **Stochastic gradient descent (SGD)**: Uses one sample at a time
        \\- **Mini-batch gradient descent**: Uses small batches (typically 32-256 samples)
        \\
        \\**Hyperparameters:**
        \\- **Learning rate**: Step size for parameter updates (typically 0.001-0.1)
        \\- **Momentum**: Helps overcome local minima and speeds convergence
        \\- **Weight decay**: Regularization term to prevent overfitting
        \\
        \\### Agent-Based Systems
        \\Autonomous software entities that perceive their environment, make decisions, and take actions to achieve specific goals. In AI systems, agents can be simple rule-based systems or complex neural networks.
        \\
        \\**Components:**
        \\- **Perception**: Sensors to observe environment state
        \\- **Decision making**: Logic or learned behavior to choose actions
        \\- **Action**: Effectors to modify the environment
        \\- **Memory**: Storage of past experiences and learned knowledge
        \\
        \\**Types:**
        \\- **Reactive agents**: Respond directly to current perceptions
        \\- **Deliberative agents**: Plan sequences of actions to achieve goals
        \\- **Learning agents**: Improve performance through experience
        \\- **Multi-agent systems**: Multiple agents cooperating or competing
        \\
        \\## ⚡ Performance & Optimization
        \\
        \\### SIMD (Single Instruction, Multiple Data)
        \\A parallel computing technique where a single instruction operates on multiple data points simultaneously. Modern CPUs have SIMD units that can process multiple floating-point numbers in one cycle.
        \\
        \\**Benefits:**
        \\- **Vectorized operations**: Process entire arrays in fewer CPU cycles
        \\- **Memory bandwidth**: More efficient use of memory bandwidth
        \\- **Energy efficiency**: Better performance per watt consumption
        \\
        \\**Common instruction sets:**
        \\- **SSE (128-bit)**: 4 float32 or 2 float64 operations per instruction
        \\- **AVX (256-bit)**: 8 float32 or 4 float64 operations per instruction
        \\- **AVX-512 (512-bit)**: 16 float32 or 8 float64 operations per instruction
        \\
        \\### Memory Alignment
        \\The practice of organizing data in memory so that it starts at addresses that are multiples of the data type size or cache line size. Proper alignment improves CPU access speed and enables SIMD optimizations.
        \\
        \\**Alignment requirements:**
        \\- **Cache line alignment**: Data aligned to 64-byte boundaries (typical cache line size)
        \\- **SIMD alignment**: Vectors aligned to 16, 32, or 64-byte boundaries
        \\- **Page alignment**: Large allocations aligned to 4KB page boundaries
        \\
        \\### Batch Processing
        \\The practice of grouping multiple operations together to improve throughput and reduce overhead. Batching amortizes the cost of setup operations across multiple data items.
        \\
        \\**Advantages:**
        \\- **Reduced function call overhead**: Fewer individual operation calls
        \\- **Better memory locality**: Sequential access patterns
        \\- **SIMD utilization**: Process multiple items with vector instructions
        \\- **Cache efficiency**: Better temporal and spatial locality
        \\
        \\## 📊 Distance Metrics
        \\
        \\### Euclidean Distance
        \\The straight-line distance between two points in multidimensional space. Most intuitive distance metric, corresponding to physical distance in 2D/3D space.
        \\
        \\**Formula:** `√(Σ(a_i - b_i)²)`
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞)
        \\- **Symmetric**: d(a,b) = d(b,a)
        \\- **Triangle inequality**: d(a,c) ≤ d(a,b) + d(b,c)
        \\- **Best for**: Continuous features, image pixels, physical measurements
        \\
        \\### Cosine Similarity
        \\Measures the cosine of the angle between two vectors, effectively measuring their directional similarity regardless of magnitude. Widely used for text embeddings and recommendation systems.
        \\
        \\**Formula:** `(a·b) / (||a|| × ||b||)`
        \\
        \\**Properties:**
        \\- **Range**: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite
        \\- **Magnitude invariant**: Only considers direction, not length
        \\- **Best for**: Text embeddings, sparse features, normalized data
        \\
        \\### Manhattan Distance (L1)
        \\The sum of absolute differences between corresponding elements. Named after the grid-like street pattern of Manhattan where you can only travel along grid lines.
        \\
        \\**Formula:** `Σ|a_i - b_i|`
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞)
        \\- **Robust to outliers**: Less sensitive than Euclidean distance
        \\- **Sparsity promoting**: Tends to produce sparse solutions in optimization
        \\- **Best for**: Sparse data, robustness to outliers, certain optimization problems
        \\
        \\### Dot Product
        \\The sum of products of corresponding elements. While not technically a distance metric, it's commonly used in neural networks and similarity calculations.
        \\
        \\**Formula:** `Σ(a_i × b_i)`
        \\
        \\**Properties:**
        \\- **Range**: (-∞, ∞)
        \\- **Not symmetric for distance**: Higher values indicate greater similarity
        \\- **Computationally efficient**: Single pass through vectors
        \\- **Best for**: Neural network computations, attention mechanisms
        \\
        \\## 🔧 System Architecture
        \\
        \\### Plugin Architecture
        \\A software design pattern that allows extending core functionality through dynamically loaded modules. Plugins are separate, compiled units that implement predefined interfaces.
        \\
        \\**Benefits:**
        \\- **Modularity**: Core system remains lean, features added as needed
        \\- **Extensibility**: Third-party developers can add functionality
        \\- **Isolation**: Plugin failures don't crash the core system
        \\- **Hot-swapping**: Plugins can be loaded/unloaded at runtime
        \\
        \\**Implementation approaches:**
        \\- **Dynamic libraries**: Shared objects (.so, .dll, .dylib) loaded at runtime
        \\- **Process isolation**: Plugins run in separate processes with IPC
        \\- **Scripting languages**: Embed interpreters for plugin languages
        \\- **WebAssembly**: Sandboxed plugins with near-native performance
        \\
        \\### Memory Management
        \\The practice of efficiently allocating, using, and deallocating memory in programs. Critical for performance and preventing memory leaks or corruption.
        \\
        \\**Allocation strategies:**
        \\- **Stack allocation**: Fast, automatic cleanup, limited size
        \\- **Heap allocation**: Flexible size, manual management required
        \\- **Pool allocation**: Pre-allocate fixed-size blocks for efficiency
        \\- **Arena allocation**: Bulk allocation with batch cleanup
        \\
        \\**Best practices:**
        \\- **RAII**: Resource Acquisition Is Initialization - tie resource lifetime to object lifetime
        \\- **Reference counting**: Track object usage automatically
        \\- **Garbage collection**: Automatic memory management (with performance trade-offs)
        \\- **Custom allocators**: Optimize for specific usage patterns
        \\
        \\### Caching Strategies
        \\Techniques for storing frequently accessed data in faster storage layers to improve performance. Caches exploit temporal and spatial locality of access patterns.
        \\
        \\**Cache types:**
        \\- **CPU caches**: L1/L2/L3 caches built into processors
        \\- **Memory caches**: Software caches in RAM
        \\- **Disk caches**: SSD or fast disk storage for slower storage
        \\- **Network caches**: CDNs and edge caches for distributed systems
        \\
        \\**Eviction policies:**
        \\- **LRU (Least Recently Used)**: Remove oldest accessed items
        \\- **LFU (Least Frequently Used)**: Remove least popular items
        \\- **FIFO (First In, First Out)**: Simple queue-based removal
        \\- **Random**: Simple but effective for many workloads
        \\
        \\## 📈 Performance Metrics
        \\
        \\### Throughput
        \\The number of operations completed per unit time. For vector databases, this typically measures insertions per second or queries per second.
        \\
        \\**Measurement:**
        \\- **Operations per second (OPS)**: Raw operation count
        \\- **Requests per second (RPS)**: For client-server systems
        \\- **Bandwidth**: Data processed per unit time (MB/s, GB/s)
        \\
        \\**Optimization factors:**
        \\- **Parallelism**: Concurrent processing of multiple operations
        \\- **Batching**: Group operations to reduce overhead
        \\- **Pipeline depth**: Overlap different stages of processing
        \\
        \\### Latency
        \\The time required to complete a single operation from start to finish. Low latency is critical for real-time applications and user experience.
        \\
        \\**Types:**
        \\- **Mean latency**: Average response time across all operations
        \\- **Percentile latency**: P50, P95, P99 latencies for understanding distribution
        \\- **Tail latency**: Worst-case response times that affect user experience
        \\
        \\**Factors affecting latency:**
        \\- **Algorithm complexity**: O(1) vs O(log n) vs O(n) operations
        \\- **Memory hierarchy**: Cache hits vs misses vs disk access
        \\- **Network delays**: Physical distance and congestion
        \\- **Queueing delays**: Waiting time under high load
        \\
        \\### Recall and Precision
        \\Metrics for evaluating the quality of search results, particularly important for approximate nearest neighbor search where exact results may be traded for speed.
        \\
        \\**Recall:**
        \\- **Definition**: Fraction of relevant results that were retrieved
        \\- **Formula**: True Positives / (True Positives + False Negatives)
        \\- **Range**: [0, 1] where 1 = perfect recall
        \\
        \\**Precision:**
        \\- **Definition**: Fraction of retrieved results that are relevant
        \\- **Formula**: True Positives / (True Positives + False Positives)
        \\- **Range**: [0, 1] where 1 = perfect precision
        \\
        \\**Trade-offs:**
        \\- **Speed vs Accuracy**: Faster algorithms often sacrifice some recall
        \\- **Memory vs Quality**: Larger indices typically provide better recall
        \\- **Index parameters**: Tuning affects the recall-speed trade-off
        \\
        \\## 🔐 Data Types & Formats
        \\
        \\### Floating Point Precision
        \\Different levels of precision for storing real numbers, each with trade-offs between accuracy, memory usage, and computational speed.
        \\
        \\**Common formats:**
        \\- **float16 (half)**: 16-bit, ±65504 range, ~3 decimal digits precision
        \\- **float32 (single)**: 32-bit, ±3.4×10³⁸ range, ~7 decimal digits precision
        \\- **float64 (double)**: 64-bit, ±1.8×10³⁰⁸ range, ~15 decimal digits precision
        \\- **bfloat16**: 16-bit with float32 range but reduced precision, popular in ML
        \\
        \\**Usage considerations:**
        \\- **ML inference**: float16 often sufficient, 2x memory savings
        \\- **Scientific computing**: float64 needed for numerical stability
        \\- **Vector embeddings**: float32 typically optimal balance
        \\- **Storage optimization**: Consider quantization for large datasets
        \\
        \\### Vector Quantization
        \\Techniques for reducing the memory footprint of high-dimensional vectors while preserving essential similarity relationships.
        \\
        \\**Methods:**
        \\- **Scalar quantization**: Map float32 to int8/int16 with scale factors
        \\- **Product quantization**: Divide vectors into subvectors, quantize each separately
        \\- **Binary quantization**: Extreme compression to binary vectors (1 bit per dimension)
        \\- **Learned quantization**: Use neural networks to optimize quantization
        \\
        \\**Benefits:**
        \\- **Memory reduction**: 4-32x compression possible
        \\- **Faster search**: Integer operations faster than floating point
        \\- **Cache efficiency**: More vectors fit in CPU cache
        \\- **Storage cost**: Reduced disk and network transfer requirements
        \\
        \\### Sparse vs Dense Vectors
        \\Two fundamental representations for high-dimensional data with different performance and storage characteristics.
        \\
        \\**Dense vectors:**
        \\- **Storage**: Every dimension explicitly stored (even zeros)
        \\- **Memory**: Fixed memory usage regardless of sparsity
        \\- **Computation**: Regular, predictable access patterns
        \\- **Best for**: Neural network embeddings, image features, audio features
        \\
        \\**Sparse vectors:**
        \\- **Storage**: Only non-zero dimensions stored with indices
        \\- **Memory**: Proportional to number of non-zero elements
        \\- **Computation**: Irregular access patterns, potential cache misses
        \\- **Best for**: Text features (TF-IDF), categorical features, user-item matrices
        \\
        \\**Hybrid approaches:**
        \\- **Compressed sparse**: Further compress indices and values
        \\- **Block sparse**: Sparse at block level, dense within blocks
        \\- **Adaptive**: Switch between representations based on sparsity level
        \\
    ;

    try file.writeAll(content);
}

// ===== New Code: Source scanner for public declarations =====
const Declaration = struct {
    name: []u8,
    kind: []u8,
    signature: []u8,
    doc: []u8,
};

fn generateCodeApiIndex(allocator: std.mem.Allocator) !void {
    // Use an arena for all temporary allocations in scanning to avoid leaks and simplify ownership
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var files = std.ArrayListUnmanaged([]u8){};
    defer files.deinit(a);

    try collectZigFiles(a, "src", &files);

    var out = try std.fs.cwd().createFile("docs/generated/CODE_API_INDEX.md", .{ .truncate = true });
    defer out.close();

    const writef = struct {
        fn go(file: std.fs.File, alloc2: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(alloc2, fmt, args);
            defer alloc2.free(s);
            try file.writeAll(s);
        }
    }.go;

    try writef(out, a, "# Code API Index (Scanned)\n\n", .{});
    try writef(out, a, "Scanned {d} Zig files under `src/`. This index lists public declarations discovered along with leading doc comments.\n\n", .{files.items.len});

    var decls = std.ArrayListUnmanaged(Declaration){};
    defer decls.deinit(a);

    for (files.items) |rel| {
        decls.clearRetainingCapacity();
        try scanFile(a, rel, &decls);
        if (decls.items.len == 0) continue;

        try writef(out, a, "## {s}\n\n", .{rel});
        for (decls.items) |d| {
            try writef(out, a, "- {s} `{s}`\n\n", .{ d.kind, d.name });
            if (d.doc.len > 0) {
                try writef(out, a, "{s}\n\n", .{d.doc});
            }
            if (d.signature.len > 0) {
                try writef(out, a, "```zig\n{s}\n```\n\n", .{d.signature});
            }
        }
    }
}

fn collectZigFiles(allocator: std.mem.Allocator, dir_path: []const u8, out_files: *std.ArrayListUnmanaged([]u8)) !void {
    var stack = std.ArrayListUnmanaged([]u8){};
    defer {
        for (stack.items) |p| allocator.free(p);
        stack.deinit(allocator);
    }
    try stack.append(allocator, try allocator.dupe(u8, dir_path));

    while (stack.items.len > 0) {
        const idx = stack.items.len - 1;
        const path = stack.items[idx];
        _ = stack.pop();
        defer allocator.free(path);

        var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch continue;
        defer dir.close();

        var it = dir.iterate();
        while (it.next() catch null) |entry| {
            if (entry.kind == .file) {
                if (std.mem.endsWith(u8, entry.name, ".zig")) {
                    const rel = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                    try out_files.append(allocator, rel);
                }
            } else if (entry.kind == .directory) {
                if (std.mem.eql(u8, entry.name, ".") or std.mem.eql(u8, entry.name, "..")) continue;
                const sub = try std.fs.path.join(allocator, &[_][]const u8{ path, entry.name });
                try stack.append(allocator, sub);
            }
        }
    }
}

fn scanFile(allocator: std.mem.Allocator, rel_path: []const u8, decls: *std.ArrayListUnmanaged(Declaration)) !void {
    const file = try std.fs.cwd().openFile(rel_path, .{});
    defer file.close();

    const data = try file.readToEndAlloc(allocator, 1024 * 1024 * 4);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var doc_buf = std.ArrayListUnmanaged(u8){};
    defer doc_buf.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "///")) {
            // accumulate doc lines
            const doc_line = std.mem.trim(u8, trimmed[3..], " \t");
            try doc_buf.appendSlice(allocator, doc_line);
            try doc_buf.append(allocator, '\n');
            continue;
        }

        // Identify public declarations after doc comments
        if (isPubDecl(trimmed)) {
            const kind = detectKind(trimmed);
            const name = extractName(allocator, trimmed) catch {
                // reset doc buffer and continue
                doc_buf.clearRetainingCapacity();
                continue;
            };
            const sig = try allocator.dupe(u8, trimmed);
            const doc = try allocator.dupe(u8, doc_buf.items);
            doc_buf.clearRetainingCapacity();

            try decls.append(allocator, .{
                .name = name,
                .kind = try allocator.dupe(u8, kind),
                .signature = sig,
                .doc = doc,
            });
            continue;
        } else {
            // reset doc buffer if we encounter a non-doc, non-decl line
            if (trimmed.len > 0 and !std.mem.startsWith(u8, trimmed, "//")) {
                doc_buf.clearRetainingCapacity();
            }
        }
    }
}

fn isPubDecl(line: []const u8) bool {
    // consider pub fn/const/var/type usingnamespace
    if (!std.mem.startsWith(u8, line, "pub ")) return false;
    return std.mem.indexOfAny(u8, line[4..], "fctuv") != null // quick filter
    or std.mem.startsWith(u8, line, "pub usingnamespace") or std.mem.indexOf(u8, line, " struct") != null or std.mem.indexOf(u8, line, " enum") != null;
}

fn detectKind(line: []const u8) []const u8 {
    if (std.mem.startsWith(u8, line, "pub fn ")) return "fn";
    if (std.mem.startsWith(u8, line, "pub const ")) {
        if (std.mem.indexOf(u8, line, " struct") != null) return "type";
        if (std.mem.indexOf(u8, line, " enum") != null) return "type";
        return "const";
    }
    if (std.mem.startsWith(u8, line, "pub var ")) return "var";
    if (std.mem.startsWith(u8, line, "pub usingnamespace")) return "usingnamespace";
    return "pub";
}

fn extractName(allocator: std.mem.Allocator, line: []const u8) ![]u8 {
    // naive name extraction: after `pub fn|const|var` read identifier
    var rest: []const u8 = line;
    if (std.mem.startsWith(u8, rest, "pub fn ")) rest = rest[7..] else if (std.mem.startsWith(u8, rest, "pub const ")) rest = rest[10..] else if (std.mem.startsWith(u8, rest, "pub var ")) rest = rest[8..] else if (std.mem.startsWith(u8, rest, "pub usingnamespace ")) rest = rest[18..] else if (std.mem.startsWith(u8, rest, "pub ")) rest = rest[4..];

    // identifier: letters, digits, underscore
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const c = rest[i];
        const is_id = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or (c == '_') or (c == '.');
        if (!is_id) break;
    }
    const ident = std.mem.trim(u8, rest[0..i], " \t");
    if (ident.len == 0) return error.Invalid;
    return allocator.dupe(u8, ident);
}

fn generateSearchIndex(allocator: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Ensure output dir exists
    try std.fs.cwd().makePath("docs/generated");

    // Collect Markdown files in docs/generated
    var dir = std.fs.cwd().openDir("docs/generated", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return, // nothing to index yet
        else => return err,
    };
    defer dir.close();

    var files = std.ArrayListUnmanaged([]const u8){};
    defer files.deinit(a);

    var it = dir.iterate();
    while (it.next() catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".md")) {
            const rel = try std.fs.path.join(a, &[_][]const u8{ "generated", entry.name });
            try files.append(a, rel);
        }
    }

    var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
    defer out.close();

    try out.writeAll("[\n");
    var first = true;

    for (files.items) |rel| {
        const full = try std.fs.path.join(a, &[_][]const u8{ "docs", rel });
        var title_buf: []const u8 = "";
        var excerpt_buf: []const u8 = "";
        getTitleAndExcerpt(a, full, &title_buf, &excerpt_buf) catch {
            // Fallbacks
            title_buf = std.fs.path.basename(rel);
            excerpt_buf = "";
        };

        if (!first) {
            try out.writeAll(",\n");
        } else {
            first = false;
        }

        try out.writeAll("  {\"file\": ");
        try writeJsonString(out, rel);
        try out.writeAll(", \"title\": ");
        try writeJsonString(out, title_buf);
        try out.writeAll(", \"excerpt\": ");
        try writeJsonString(out, excerpt_buf);
        try out.writeAll("}");
    }

    try out.writeAll("\n]\n");
}

fn getTitleAndExcerpt(allocator: std.mem.Allocator, path: []const u8, title_out: *[]const u8, excerpt_out: *[]const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 1024 * 1024 * 4);
    defer allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');

    var first_heading: ?[]const u8 = null;
    var in_code = false;

    var excerpt = std.ArrayListUnmanaged(u8){};
    defer excerpt.deinit(allocator);

    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) continue;

        if (first_heading == null and std.mem.startsWith(u8, trimmed, "#")) {
            var j: usize = 0;
            while (j < trimmed.len and trimmed[j] == '#') j += 1;
            const after = std.mem.trim(u8, trimmed[j..], " \t");
            if (after.len > 0) first_heading = after;
            continue;
        }

        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue; // skip headings in excerpt
        if (trimmed[0] == '|') continue; // skip tables

        // Append to excerpt up to ~300 chars
        if (excerpt.items.len > 0) try excerpt.append(allocator, ' ');
        var k: usize = 0;
        while (k < trimmed.len and excerpt.items.len < 300) : (k += 1) {
            const c = trimmed[k];
            if (c == '`') continue;
            try excerpt.append(allocator, c);
        }
        if (excerpt.items.len >= 300) break;
    }

    if (first_heading) |h| {
        title_out.* = try allocator.dupe(u8, h);
    } else {
        const base = std.fs.path.basename(path);
        title_out.* = try allocator.dupe(u8, base);
    }
    excerpt_out.* = try allocator.dupe(u8, excerpt.items);
}

fn writeJsonString(out: std.fs.File, s: []const u8) !void {
    try out.writeAll("\"");
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        switch (c) {
            '\\' => try out.writeAll("\\\\"),
            '"' => try out.writeAll("\\\""),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            else => {
                var buf: [1]u8 = .{c};
                try out.writeAll(buf[0..1]);
            },
        }
    }
    try out.writeAll("\"");
}

fn generateDocsIndexHtml(_: std.mem.Allocator) !void {
    // Write a GitHub Pages friendly index.html that renders docs/generated/*.md client-side
    try std.fs.cwd().makePath("docs");
    var out = try std.fs.cwd().createFile("docs/index.html", .{ .truncate = true });
    defer out.close();

    const html =
        \\<!DOCTYPE html>
        \\<html lang="en">
        \\<head>
        \\  <meta charset="UTF-8" />
        \\  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        \\  <title>ABI Documentation</title>
        \\  <style>
        \\    body {
        \\      margin: 0;
        \\      font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        \\      color-scheme: light dark;
        \\      --bg: #0b0d10;
        \\      --fg: #e6edf3;
        \\      --muted: #9aa4b2;
        \\      --accent: #5eb1ff;
        \\      --panel: #111418;
        \\      background: var(--bg);
        \\      color: var(--fg);
        \\      display: grid;
        \\      grid-template-columns: 320px 1fr;
        \\      height: 100svh;
        \\      overflow: hidden;
        \\    }
        \\
        \\    aside {
        \\      background: var(--panel);
        \\      border-right: 1px solid #1f2328;
        \\      display: flex;
        \\      flex-direction: column;
        \\      padding: 16px;
        \\      overflow: hidden;
        \\    }
        \\
        \\    main {
        \\      overflow: auto;
        \\      padding: 24px 32px;
        \\    }
        \\
        \\    #brand { font-weight: 700; letter-spacing: 0.2px; margin-bottom: 8px; }
        \\    #desc { color: var(--muted); font-size: 13px; margin-bottom: 12px; }
        \\    #search { padding: 10px 12px; border-radius: 8px; border: 1px solid #2a2f36; background: #0f1216; color: var(--fg); outline: none; width: 100%; box-sizing: border-box; }
        \\    #nav { overflow: auto; margin-top: 12px; padding-right: 6px; }
        \\    .nav-item { padding: 10px 8px; border-radius: 8px; }
        \\    .nav-item:hover { background: rgba(94, 177, 255, 0.12); }
        \\    .nav-item a { color: var(--fg); text-decoration: none; font-weight: 600; }
        \\    .nav-excerpt { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.35; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; overflow: hidden; }
        \\
        \\    /* Content styling */
        \\    #content { max-width: 1100px; margin: 0 auto; line-height: 1.6; }
        \\    #content h1, #content h2, #content h3 { margin-top: 26px; scroll-margin-top: 80px; }
        \\    #content pre { background: #0f1216; padding: 14px; border-radius: 8px; overflow: auto; }
        \\    #content code { background: #0f1216; padding: 2px 4px; border-radius: 4px; }
        \\    #content a { color: var(--accent); }
        \\    #topbar { display: flex; align-items: center; justify-content: space-between; }
        \\    #topbar .right { display: flex; gap: 8px; align-items: center; }
        \\    button.small { background: #0f1216; border: 1px solid #2a2f36; color: var(--fg); border-radius: 8px; padding: 6px 10px; cursor: pointer; }
        \\    button.small:hover { border-color: var(--accent); }
        \\  </style>
        \\</head>
        \\<body>
        \\  <aside>
        \\    <div id="brand">ABI Docs</div>
        \\    <div id="desc">Search and browse documentation generated from the codebase.</div>
        \\    <input id="search" type="search" placeholder="Search docs..." />
        \\    <div id="nav"></div>
        \\  </aside>
        \\  <main>
        \\    <div id="topbar">
        \\      <div></div>
        \\      <div class="right">
        \\        <button class="small" id="open_md">Open in raw Markdown</button>
        \\      </div>
        \\    </div>
        \\    <div id="content"></div>
        \\  </main>
        \\  <script>
        \\  async function fetchJSON(p) { const r = await fetch(p); return await r.json(); }
        \\  async function fetchText(p) { const r = await fetch(p); return await r.text(); }
        \\  function escapeHTML(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
        \\  function mdToHtml(md) {
        \\    const lines = md.split('\\n');
        \\    let html = '';
        \\    let inCode = false; let codeLang = '';
        \\    for (let i=0;i<lines.length;i++){
        \\      let line = lines[i];
        \\      if (line.startsWith('```')) {
        \\        if (!inCode) { inCode = true; codeLang = line.slice(3).trim(); html += '<pre><code class="lang-'+escapeHTML(codeLang)+'">'; }
        \\        else { inCode = false; html += '</code></pre>'; }
        \\        continue;
        \\      }
        \\      if (inCode) { html += escapeHTML(line)+'\\n'; continue; }
        \\      if (line.startsWith('#')) {
        \\        const m = line.match(/^#+/); const level = m ? m[0].length : 1;
        \\        const text = line.slice(level).trim();
        \\        const id = text.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,'');
        \\        html += `<h${level} id="${id}">${inline(text)}</h${level}>`;
        \\        continue;
        \\      }
        \\      if (/^\\s*[-*] /.test(line)) {
        \\        let items = []; let j=i;
        \\        while (j<lines.length && /^\\s*[-*] /.test(lines[j])) { items.push(lines[j].replace(/^\\s*[-*] /,'')); j++; }
        \\        html += '<ul>' + items.map(it => `<li>${inline(it)}</li>`).join('') + '</ul>';
        \\        i = j-1; continue;
        \\      }
        \\      if (/^\\s*[0-9]+\\. /.test(line)) {
        \\        let items=[]; let j=i;
        \\        while (j<lines.length && /^\\s*[0-9]+\\. /.test(lines[j])) { items.push(lines[j].replace(/^\\s*[0-9]+\\. /,'')); j++; }
        \\        html += '<ol>' + items.map(it => `<li>${inline(it)}</li>`).join('') + '</ol>';
        \\        i = j-1; continue;
        \\      }
        \\      if (line.trim() === '') { html += ''; continue; }
        \\      html += `<p>${inline(line)}</p>`;
        \\    }
        \\    return html;
        \\    function inline(t) {
        \\      t = t.replace(/`([^`]+)`/g,'<code>$1</code>');
        \\      t = t.replace(/\\*\\*([^*]+)\\*\\*/g,'<strong>$1</strong>');
        \\      t = t.replace(/\\*([^*]+)\\*/g,'<em>$1</em>');
        \\      t = t.replace(/\\!\\[([^\\]]*)\\]\\(([^)]+)\\)/g,'<img alt="$1" src="$2" />');
        \\      t = t.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g,(m,txt,url) => {\n        \\        if (url.startsWith('http') || url.startsWith('mailto:') || url.startsWith('#')) return `<a href=\"${url}\" target=\"_blank\" rel=\"noopener\">${txt}</a>`;\n        \\        if (url.endsWith('.md')) return `<a href=\"#${url}\" onclick=\"loadDoc('${url}'); return false;\">${txt}</a>`;\n        \\        return `<a href=\"${url}\" target=\"_blank\" rel=\"noopener\">${txt}</a>`;\n        \\      });
        \\      return t;
        \\    }
        \\  }
        \\  let searchData = [];
        \\  let currentPath = '';
        \\  async function init(){
        \\    try { searchData = await fetchJSON('generated/search_index.json'); } catch(e){ console.error('search index', e); }
        \\    renderNav(searchData);
        \\    const initial = location.hash ? decodeURIComponent(location.hash.slice(1)) : (searchData[0]?.file || 'generated/API_REFERENCE.md');
        \\    loadDoc(initial);
        \\    document.getElementById('search').addEventListener('input', (e) => {
        \\      const q = e.target.value.toLowerCase();
        \\      const results = searchData.filter(it => it.title.toLowerCase().includes(q) || it.excerpt.toLowerCase().includes(q));
        \\      renderNav(results);
        \\    });
        \\    document.getElementById('open_md').addEventListener('click', () => { if (currentPath) window.open(currentPath, '_blank'); });
        \\  }
        \\  async function loadDoc(path){
        \\    currentPath = path;
        \\    location.hash = encodeURIComponent(path);
        \\    const md = await fetchText(path);
        \\    document.getElementById('content').innerHTML = mdToHtml(md);
        \\    window.scrollTo(0,0);
        \\  }
        \\  function renderNav(list){
        \\    const nav = document.getElementById('nav');
        \\    nav.innerHTML = list.map(it => `<div class="nav-item"><a href="#${encodeURIComponent(it.file)}" onclick="loadDoc('${it.file}'); return false;">${it.title}</a><div class="nav-excerpt">${escapeHTML(it.excerpt)}</div></div>`).join('');
        \\  }
        \\  window.addEventListener('hashchange', () => {
        \\    const p = decodeURIComponent(location.hash.slice(1));
        \\    if (p) loadDoc(p);
        \\  });
        \\  init();
        \\  </script>
        \\</body>
        \\</html>
    ;

    try out.writeAll(html);
}
