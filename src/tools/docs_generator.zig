const std = @import("std");
const abi = @import("abi");

/// Documentation generator for ABI project
/// Generates comprehensive API documentation from source code with enhanced GitHub Pages support
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating ABI API Documentation with GitHub Pages optimization", .{});

    // Create comprehensive docs directory structure
    try std.fs.cwd().makePath("docs/generated");
    try std.fs.cwd().makePath("docs/assets/css");
    try std.fs.cwd().makePath("docs/assets/js");
    try std.fs.cwd().makePath("docs/_layouts");
    try std.fs.cwd().makePath("docs/_data");
    try std.fs.cwd().makePath(".github/workflows");

    // GitHub Pages configuration (no Jekyll processing)
    try generateNoJekyll();

    // Generate Jekyll configuration for enhanced GitHub Pages support
    try generateJekyllConfig(allocator);
    try generateGitHubPagesLayout(allocator);
    try generateNavigationData(allocator);
    try generateSEOMetadata(allocator);

    // Generate core documentation
    try generateModuleDocs(allocator);
    try generateApiReference(allocator);
    try generateExamples(allocator);
    try generatePerformanceGuide(allocator);
    try generateDefinitionsReference(allocator);

    // Enhanced documentation features
    try generateCodeApiIndex(allocator);
    try generateSearchIndex(allocator);

    // GitHub Pages assets and styling
    try generateGitHubPagesAssets(allocator);

    // Static index and assets
    try generateDocsIndexHtml(allocator);
    try generateReadmeRedirect(allocator);

    // GitHub Actions workflow for automated deployment
    try generateGitHubActionsWorkflow(allocator);

    // Generate Zig native documentation
    try generateZigNativeDocs();

    std.log.info("✅ GitHub Pages documentation generation completed!", .{});
    std.log.info("📝 To deploy: Enable GitHub Pages in repository settings (source: docs folder)", .{});
    std.log.info("🚀 GitHub Actions workflow created for automated deployment", .{});
}

fn generateNoJekyll() !void {
    // Ensure GitHub Pages does not run Jekyll
    var file = try std.fs.cwd().createFile("docs/.nojekyll", .{});
    defer file.close();
}

/// Generate Jekyll configuration for GitHub Pages
fn generateJekyllConfig(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_config.yml", .{});
    defer file.close();

    const content =
        \\# ABI Documentation - Jekyll Configuration
        \\title: "ABI Documentation"
        \\description: "High-performance vector database with AI capabilities"
        \\url: "https://donaldfilimon.github.io/abi/"
        \\baseurl: "/abi"
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
        \\  name: "ABI Team"
        \\  email: "team@abi.dev"
        \\
        \\# GitHub repository
        \\github:
        \\  repository_url: "https://github.com/donaldfilimon/abi"
        \\  repository_name: "abi"
        \\  owner_name: "donaldfilimon"
        \\
        \\# Social media
        \\social:
        \\  type: "Organization"
        \\  links:
        \\    - "https://github.com/donaldfilimon/abi"
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
        \\      - title: "Testing Strategy"
        \\        url: "/TESTING_STRATEGY/"
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
        \\    <loc>https://donaldfilimon.github.io/abi/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>1.0</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/API_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.9</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/MODULE_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.8</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/EXAMPLES/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>weekly</changefreq>
        \\    <priority>0.8</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/PERFORMANCE_GUIDE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>monthly</changefreq>
        \\    <priority>0.7</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/DEFINITIONS_REFERENCE/</loc>
        \\    <lastmod>{{ site.time | date_to_xmlschema }}</lastmod>
        \\    <changefreq>monthly</changefreq>
        \\    <priority>0.7</priority>
        \\  </url>
        \\  <url>
        \\    <loc>https://donaldfilimon.github.io/abi/generated/CODE_API_INDEX/</loc>
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
        \\Sitemap: https://donaldfilimon.github.io/abi/sitemap.xml
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
fn generateZigNativeDocs() !void {
    // Create directory for native docs
    try std.fs.cwd().makePath("docs/zig-docs");
    // In a CI environment, you could run: zig build docs
    // Here we create a placeholder index to avoid empty directory
    var out = try std.fs.cwd().createFile("docs/zig-docs/index.html", .{ .truncate = true });
    defer out.close();
    try out.writeAll("<html><body><h1>Zig Native Docs</h1><p>Generated offline.</p></body></html>");
}

/// Generate README redirect for GitHub
fn generateReadmeRedirect(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/README.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "ABI Documentation"
        \\description: "High-performance vector database with AI capabilities - Complete documentation"
        \\permalink: /
        \\---
        \\
        \\# ABI Documentation
        \\
        \\Welcome to the comprehensive documentation for ABI, a high-performance vector database with integrated AI capabilities.
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
        \\- **[GitHub Repository](https://github.com/donaldfilimon/abi/)** - Source code and issues
        \\- **[Zig Language](https://ziglang.org/)** - Learn about the Zig programming language
        \\- **[Vector Databases](./generated/DEFINITIONS_REFERENCE/#vector-database)** - Learn about vector database concepts
        \\
        \\## 📧 Support
        \\
        \\- **Issues**: [Report bugs or request features](https://github.com/donaldfilimon/abi/issues)
        \\- **Discussions**: [Join community discussions](https://github.com/donaldfilimon/abi/discussions)
        \\- **Documentation**: [Improve documentation](https://github.com/donaldfilimon/abi/issues/new?title=Documentation%20Improvement)
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

// ===== New Code: Source scanner for public declarations =====
const Declaration = struct {
    name: []u8,
    kind: []u8,
    signature: []u8,
    doc: []u8,
};

fn docPathLessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn generateCodeApiIndex(allocator: std.mem.Allocator) !void {
    // Use an arena for all temporary allocations in scanning to avoid leaks and simplify ownership
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var files = std.ArrayListUnmanaged([]const u8){};
    defer files.deinit(a);

    try collectZigFiles(a, "src", &files);

    std.sort.block([]const u8, files.items, {}, docPathLessThan);

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

fn collectZigFiles(allocator: std.mem.Allocator, dir_path: []const u8, out_files: *std.ArrayListUnmanaged([]const u8)) !void {
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

    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    var reader = file.reader();
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = reader.read(&buf) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        if (n == 0) break;
        try buffer.appendSlice(buf[0..n]);
    }
    const data = try buffer.toOwnedSlice();
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

    std.sort.block([]const u8, files.items, {}, docPathLessThan);

    var out = try std.fs.cwd().createFile("docs/generated/search_index.json", .{ .truncate = true });
    defer out.close();

    try out.writeAll("[\n");
    var first = true;

    for (files.items) |rel| {
        const full = try std.fs.path.join(a, &[_][]const u8{ "docs", rel });
        // Normalize relative path for web (forward slashes)
        const rel_web = try a.dupe(u8, rel);
        for (rel_web) |*ch| {
            if (ch.* == std.fs.path.sep) ch.* = '/';
        }
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
        try writeJsonString(out, rel_web);
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
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    var reader = file.reader();
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = try reader.read(&buf);
        if (n == 0) break;
        try buffer.appendSlice(buf[0..n]);
    }
    const data = try buffer.toOwnedSlice();
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

/// Generate CSS and JavaScript assets for GitHub Pages
fn generateGitHubPagesAssets(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // Generate enhanced CSS
    const css_file = try std.fs.cwd().createFile("docs/assets/css/documentation.css", .{});
    defer css_file.close();

    const css_content =
        \\/* Enhanced GitHub Pages Documentation Styles */
        \\:root {
        \\  --color-canvas-default: #ffffff;
        \\  --color-canvas-subtle: #f6f8fa;
        \\  --color-border-default: #d0d7de;
        \\  --color-border-muted: #d8dee4;
        \\  --color-fg-default: #1f2328;
        \\  --color-fg-muted: #656d76;
        \\  --color-accent-fg: #0969da;
        \\  --color-accent-emphasis: #0969da;
        \\  --color-success-fg: #1a7f37;
        \\  --color-attention-fg: #9a6700;
        \\  --color-severe-fg: #d1242f;
        \\}
        \\
        \\@media (prefers-color-scheme: dark) {
        \\  :root {
        \\    --color-canvas-default: #0d1117;
        \\    --color-canvas-subtle: #161b22;
        \\    --color-border-default: #30363d;
        \\    --color-border-muted: #21262d;
        \\    --color-fg-default: #e6edf3;
        \\    --color-fg-muted: #8b949e;
        \\    --color-accent-fg: #58a6ff;
        \\    --color-accent-emphasis: #1f6feb;
        \\    --color-success-fg: #3fb950;
        \\    --color-attention-fg: #d29922;
        \\    --color-severe-fg: #f85149;
        \\  }
        \\}
        \\
        \\/* Documentation-specific styles */
        \\.documentation-content {
        \\  max-width: 1012px;
        \\  margin: 0 auto;
        \\  padding: 32px;
        \\  line-height: 1.6;
        \\}
        \\
        \\.table-of-contents {
        \\  background: var(--color-canvas-subtle);
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 6px;
        \\  padding: 16px;
        \\  margin: 16px 0 24px 0;
        \\  font-size: 14px;
        \\}
        \\
        \\.table-of-contents h4 {
        \\  margin: 0 0 8px 0;
        \\  color: var(--color-fg-default);
        \\  font-weight: 600;
        \\}
        \\
        \\.table-of-contents ul {
        \\  margin: 0;
        \\  padding-left: 20px;
        \\}
        \\
        \\.table-of-contents li {
        \\  margin: 4px 0;
        \\}
        \\
        \\.table-of-contents a {
        \\  color: var(--color-accent-fg);
        \\  text-decoration: none;
        \\}
        \\
        \\.table-of-contents a:hover {
        \\  text-decoration: underline;
        \\}
        \\
        \\/* Search functionality */
        \\.search-container {
        \\  position: relative;
        \\  margin: 16px 0;
        \\}
        \\
        \\#search-input {
        \\  width: 100%;
        \\  padding: 8px 12px;
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 6px;
        \\  background: var(--color-canvas-default);
        \\  color: var(--color-fg-default);
        \\  font-size: 14px;
        \\}
        \\
        \\.search-results {
        \\  position: absolute;
        \\  top: 100%;
        \\  left: 0;
        \\  right: 0;
        \\  background: var(--color-canvas-default);
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 6px;
        \\  max-height: 300px;
        \\  overflow-y: auto;
        \\  z-index: 1000;
        \\  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        \\}
        \\
        \\.search-result-item {
        \\  padding: 12px;
        \\  border-bottom: 1px solid var(--color-border-muted);
        \\  cursor: pointer;
        \\}
        \\
        \\.search-result-item:hover {
        \\  background: var(--color-canvas-subtle);
        \\}
        \\
        \\.search-result-title {
        \\  font-weight: 600;
        \\  color: var(--color-accent-fg);
        \\  margin-bottom: 4px;
        \\}
        \\
        \\.search-result-excerpt {
        \\  color: var(--color-fg-muted);
        \\  font-size: 13px;
        \\  line-height: 1.4;
        \\}
        \\
        \\/* Navigation improvements */
        \\.doc-navigation {
        \\  margin: 24px 0;
        \\}
        \\
        \\.doc-navigation h3 {
        \\  margin: 0 0 12px 0;
        \\  font-size: 16px;
        \\  font-weight: 600;
        \\  color: var(--color-fg-default);
        \\}
        \\
        \\.doc-navigation ul {
        \\  list-style: none;
        \\  margin: 0;
        \\  padding: 0;
        \\}
        \\
        \\.doc-navigation li {
        \\  margin: 0;
        \\}
        \\
        \\.doc-navigation a {
        \\  display: block;
        \\  padding: 8px 12px;
        \\  color: var(--color-fg-default);
        \\  text-decoration: none;
        \\  border-radius: 6px;
        \\  transition: background-color 0.1s ease;
        \\}
        \\
        \\.doc-navigation a:hover {
        \\  background: var(--color-canvas-subtle);
        \\  text-decoration: none;
        \\}
        \\
        \\.doc-navigation a.current {
        \\  background: var(--color-accent-emphasis);
        \\  color: #ffffff;
        \\  font-weight: 600;
        \\}
        \\
        \\/* Code syntax highlighting */
        \\.highlight {
        \\  background: var(--color-canvas-subtle);
        \\  border-radius: 6px;
        \\  padding: 16px;
        \\  overflow-x: auto;
        \\  margin: 16px 0;
        \\}
        \\
        \\.highlight pre {
        \\  margin: 0;
        \\  background: transparent;
        \\}
        \\
        \\/* Responsive design */
        \\@media (max-width: 768px) {
        \\  .documentation-content {
        \\    padding: 16px;
        \\  }
        \\  
        \\  .table-of-contents {
        \\    margin: 16px -16px 24px -16px;
        \\    border-radius: 0;
        \\    border-left: none;
        \\    border-right: none;
        \\  }
        \\}
        \\
        \\/* Feedback section */
        \\.feedback-section {
        \\  margin-top: 48px;
        \\  padding: 24px;
        \\  background: var(--color-canvas-subtle);
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 6px;
        \\}
        \\
        \\.feedback-section h3 {
        \\  margin: 0 0 12px 0;
        \\  color: var(--color-fg-default);
        \\}
        \\
        \\.feedback-section p {
        \\  margin: 0;
        \\  color: var(--color-fg-muted);
        \\}
        \\
        \\.feedback-section a {
        \\  color: var(--color-accent-fg);
        \\  text-decoration: none;
        \\}
        \\
        \\.feedback-section a:hover {
        \\  text-decoration: underline;
        \\}
        \\
        \\/* Performance indicators */
        \\.performance-badge {
        \\  display: inline-block;
        \\  padding: 2px 6px;
        \\  background: var(--color-success-fg);
        \\  color: #ffffff;
        \\  font-size: 11px;
        \\  font-weight: 600;
        \\  border-radius: 12px;
        \\  text-transform: uppercase;
        \\  letter-spacing: 0.5px;
        \\}
        \\
        \\.performance-badge.warning {
        \\  background: var(--color-attention-fg);
        \\}
        \\
        \\.performance-badge.error {
        \\  background: var(--color-severe-fg);
        \\}
        \\
        \\/* Quick navigation cards */
        \\.quick-nav {
        \\  display: grid;
        \\  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        \\  gap: 16px;
        \\  margin: 24px 0;
        \\}
        \\
        \\.nav-card {
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 8px;
        \\  padding: 20px;
        \\  background: var(--color-canvas-default);
        \\  transition: border-color 0.2s ease, box-shadow 0.2s ease;
        \\}
        \\
        \\.nav-card:hover {
        \\  border-color: var(--color-accent-emphasis);
        \\  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        \\}
        \\
        \\.nav-card h3 {
        \\  margin: 0 0 8px 0;
        \\  font-size: 18px;
        \\}
        \\
        \\.nav-card h3 a {
        \\  color: var(--color-accent-fg);
        \\  text-decoration: none;
        \\}
        \\
        \\.nav-card p {
        \\  margin: 0;
        \\  color: var(--color-fg-muted);
        \\  font-size: 14px;
        \\  line-height: 1.5;
        \\}
        \\
        \\/* Print styles */
        \\@media print {
        \\  .doc-navigation,
        \\  .search-container,
        \\  .feedback-section {
        \\    display: none;
        \\  }
        \\  
        \\  .documentation-content {
        \\    max-width: none;
        \\    margin: 0;
        \\    padding: 0;
        \\  }
        \\}
        \\
    ;

    try css_file.writeAll(css_content);

    // Generate JavaScript for enhanced functionality
    const js_file = try std.fs.cwd().createFile("docs/assets/js/documentation.js", .{});
    defer js_file.close();

    const js_content =
        \\// Enhanced GitHub Pages Documentation JavaScript
        \\(function() {
        \\  'use strict';
        \\
        \\  // Generate table of contents
        \\  function generateTOC() {
        \\    const content = document.querySelector('.documentation-content .content');
        \\    const tocList = document.getElementById('toc-list');
        \\    
        \\    if (!content || !tocList) return;
        \\
        \\    const headings = content.querySelectorAll('h2, h3, h4');
        \\    if (headings.length === 0) {
        \\      document.getElementById('toc').style.display = 'none';
        \\      return;
        \\    }
        \\
        \\    headings.forEach((heading, index) => {
        \\      const id = heading.id || `heading-${index}`;
        \\      heading.id = id;
        \\      
        \\      const li = document.createElement('li');
        \\      const a = document.createElement('a');
        \\      a.href = `#${id}`;
        \\      a.textContent = heading.textContent;
        \\      a.className = `toc-${heading.tagName.toLowerCase()}`;
        \\      
        \\      li.appendChild(a);
        \\      tocList.appendChild(li);
        \\    });
        \\  }
        \\
        \\  // Search functionality
        \\  function initializeSearch() {
        \\    const searchInput = document.getElementById('search-input');
        \\    const searchResults = document.getElementById('search-results');
        \\    
        \\    if (!searchInput || !searchResults) return;
        \\
        \\    let searchData = [];
        \\    
        \\    // Load search index
        \\    fetch('/generated/search_index.json')
        \\      .then(response => response.json())
        \\      .then(data => {
        \\        searchData = data;
        \\      })
        \\      .catch(error => {
        \\        console.warn('Search index not available:', error);
        \\      });
        \\
        \\    let searchTimeout;
        \\    searchInput.addEventListener('input', function() {
        \\      clearTimeout(searchTimeout);
        \\      const query = this.value.trim().toLowerCase();
        \\      
        \\      if (query.length < 2) {
        \\        searchResults.classList.add('hidden');
        \\        return;
        \\      }
        \\
        \\      searchTimeout = setTimeout(() => {
        \\        const results = searchData.filter(item => 
        \\          item.title.toLowerCase().includes(query) || 
        \\          item.excerpt.toLowerCase().includes(query)
        \\        ).slice(0, 10);
        \\
        \\        displaySearchResults(results, query);
        \\      }, 200);
        \\    });
        \\
        \\    // Hide search results when clicking outside
        \\    document.addEventListener('click', function(e) {
        \\      if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        \\        searchResults.classList.add('hidden');
        \\      }
        \\    });
        \\  }
        \\
        \\  function displaySearchResults(results, query) {
        \\    const searchResults = document.getElementById('search-results');
        \\    if (!searchResults) return;
        \\
        \\    if (results.length === 0) {
        \\      searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
        \\    } else {
        \\      searchResults.innerHTML = results.map(result => 
        \\        `<div class="search-result-item" onclick="navigateToPage('${result.file}')">
        \\          <div class="search-result-title">${highlightText(result.title, query)}</div>
        \\          <div class="search-result-excerpt">${highlightText(result.excerpt, query)}</div>
        \\        </div>`
        \\      ).join('');
        \\    }
        \\    
        \\    searchResults.classList.remove('hidden');
        \\  }
        \\
        \\  function highlightText(text, query) {
        \\    if (!query) return text;
        \\    const regex = new RegExp(`(${query})`, 'gi');
        \\    return text.replace(regex, '<strong>$1</strong>');
        \\  }
        \\
        \\  function navigateToPage(file) {
        \\    window.location.href = `/${file}`;
        \\  }
        \\
        \\  // Smooth scrolling for anchor links
        \\  function initializeSmoothScrolling() {
        \\    document.addEventListener('click', function(e) {
        \\      if (e.target.tagName === 'A' && e.target.getAttribute('href').startsWith('#')) {
        \\        e.preventDefault();
        \\        const targetId = e.target.getAttribute('href').substring(1);
        \\        const targetElement = document.getElementById(targetId);
        \\        
        \\        if (targetElement) {
        \\          targetElement.scrollIntoView({
        \\            behavior: 'smooth',
        \\            block: 'start'
        \\          });
        \\          
        \\          // Update URL without triggering navigation
        \\          history.pushState(null, null, `#${targetId}`);
        \\        }
        \\      }
        \\    });
        \\  }
        \\
        \\  // Copy code functionality
        \\  function addCopyButtons() {
        \\    const codeBlocks = document.querySelectorAll('pre code');
        \\    
        \\    codeBlocks.forEach(function(codeBlock) {
        \\      const pre = codeBlock.parentElement;
        \\      const button = document.createElement('button');
        \\      button.textContent = 'Copy';
        \\      button.className = 'copy-button';
        \\      button.style.cssText = `
        \\        position: absolute;
        \\        top: 8px;
        \\        right: 8px;
        \\        background: var(--color-canvas-subtle);
        \\        border: 1px solid var(--color-border-default);
        \\        border-radius: 4px;
        \\        padding: 4px 8px;
        \\        font-size: 12px;
        \\        cursor: pointer;
        \\        color: var(--color-fg-default);
        \\      `;
        \\      
        \\      pre.style.position = 'relative';
        \\      pre.appendChild(button);
        \\      
        \\      button.addEventListener('click', function() {
        \\        navigator.clipboard.writeText(codeBlock.textContent).then(function() {
        \\          button.textContent = 'Copied!';
        \\          setTimeout(function() {
        \\            button.textContent = 'Copy';
        \\          }, 2000);
        \\        });
        \\      });
        \\    });
        \\  }
        \\
        \\  // Performance monitoring
        \\  function trackPerformance() {
        \\    if ('performance' in window) {
        \\      window.addEventListener('load', function() {
        \\        setTimeout(function() {
        \\          const perfData = performance.getEntriesByType('navigation')[0];
        \\          const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
        \\          
        \\          if (loadTime > 0) {
        \\            console.log(`Page load time: ${loadTime}ms`);
        \\          }
        \\        }, 0);
        \\      });
        \\    }
        \\  }
        \\
        \\  // Initialize all functionality when DOM is ready
        \\  function initialize() {
        \\    generateTOC();
        \\    initializeSearch();
        \\    initializeSmoothScrolling();
        \\    addCopyButtons();
        \\    trackPerformance();
        \\    
        \\    // Add performance badges to relevant sections
        \\    const performanceMarkers = document.querySelectorAll('code:contains("~"), code:contains("ms"), code:contains("μs")');
        \\    performanceMarkers.forEach(function(marker) {
        \\      if (marker.textContent.includes('~')) {
        \\        const badge = document.createElement('span');
        \\        badge.className = 'performance-badge';
        \\        badge.textContent = 'PERF';
        \\        marker.parentElement.insertBefore(badge, marker.nextSibling);
        \\      }
        \\    });
        \\  }
        \\
        \\  // DOM ready check
        \\  if (document.readyState === 'loading') {
        \\    document.addEventListener('DOMContentLoaded', initialize);
        \\  } else {
        \\    initialize();
        \\  }
        \\
        \\})();
        \\
    ;

    try js_file.writeAll(js_content);

    // Generate search JavaScript
    const search_js_file = try std.fs.cwd().createFile("docs/assets/js/search.js", .{});
    defer search_js_file.close();

    const search_js_content =
        \\// Advanced search functionality for GitHub Pages
        \\(function() {
        \\  'use strict';
        \\
        \\  let searchIndex = [];
        \\  let searchWorker;
        \\
        \\  // Initialize search with web worker for better performance
        \\  function initializeAdvancedSearch() {
        \\    // Load search index
        \\    fetch('/generated/search_index.json')
        \\      .then(response => response.json())
        \\      .then(data => {
        \\        searchIndex = data;
        \\        setupSearchInterface();
        \\      })
        \\      .catch(error => {
        \\        console.warn('Search functionality unavailable:', error);
        \\      });
        \\  }
        \\
        \\  function setupSearchInterface() {
        \\    const searchInput = document.getElementById('search-input');
        \\    if (!searchInput) return;
        \\
        \\    // Add keyboard shortcuts
        \\    document.addEventListener('keydown', function(e) {
        \\      // Ctrl/Cmd + K to focus search
        \\      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        \\        e.preventDefault();
        \\        searchInput.focus();
        \\        searchInput.select();
        \\      }
        \\      
        \\      // Escape to clear search
        \\      if (e.key === 'Escape' && document.activeElement === searchInput) {
        \\        searchInput.value = '';
        \\        hideSearchResults();
        \\      }
        \\    });
        \\
        \\    // Add search suggestions
        \\    searchInput.addEventListener('focus', function() {
        \\      if (this.value.trim() === '') {
        \\        showSearchSuggestions();
        \\      }
        \\    });
        \\  }
        \\
        \\  function showSearchSuggestions() {
        \\    const suggestions = [
        \\      'database API',
        \\      'neural networks',
        \\      'SIMD operations',
        \\      'performance guide',
        \\      'plugin system',
        \\      'vector search',
        \\      'machine learning'
        \\    ];
        \\
        \\    const searchResults = document.getElementById('search-results');
        \\    if (!searchResults) return;
        \\
        \\    searchResults.innerHTML = suggestions.map(suggestion =>
        \\      `<div class="search-result-item suggestion" onclick="searchFor('${suggestion}')">
        \\        <div class="search-result-title">💡 ${suggestion}</div>
        \\        <div class="search-result-excerpt">Search suggestion</div>
        \\      </div>`
        \\    ).join('');
        \\    
        \\    searchResults.classList.remove('hidden');
        \\  }
        \\
        \\  function searchFor(query) {
        \\    const searchInput = document.getElementById('search-input');
        \\    if (searchInput) {
        \\      searchInput.value = query;
        \\      searchInput.dispatchEvent(new Event('input'));
        \\    }
        \\  }
        \\
        \\  function hideSearchResults() {
        \\    const searchResults = document.getElementById('search-results');
        \\    if (searchResults) {
        \\      searchResults.classList.add('hidden');
        \\    }
        \\  }
        \\
        \\  // Fuzzy search implementation
        \\  function fuzzySearch(query, items) {
        \\    const fuse = new Fuse(items, {
        \\      keys: ['title', 'excerpt'],
        \\      threshold: 0.4,
        \\      distance: 100,
        \\      includeScore: true
        \\    });
        \\    
        \\    return fuse.search(query).map(result => result.item);
        \\  }
        \\
        \\  // Initialize when DOM is ready
        \\  if (document.readyState === 'loading') {
        \\    document.addEventListener('DOMContentLoaded', initializeAdvancedSearch);
        \\  } else {
        \\    initializeAdvancedSearch();
        \\  }
        \\
        \\})();
        \\
    ;

    try search_js_file.writeAll(search_js_content);
}

/// Generate comprehensive module documentation
fn generateModuleDocs(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/generated/MODULE_REFERENCE.md", .{});
    defer file.close();

    const content =
        \\---
        \\layout: documentation
        \\title: "Module Reference"
        \\description: "Comprehensive reference for all ABI modules and components"
        \\---
        \\
        \\# ABI Module Reference
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
        \\description: "Complete API reference for ABI with detailed function documentation"
        \\---
        \\
        \\# ABI API Reference
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
        \\description: "Practical examples and tutorials for using ABI effectively"
        \\---
        \\
        \\# ABI Usage Examples
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
        \\    var training_data = std.array_list.Managed(abi.TrainingData).init(allocator);
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
        \\    var result = std.array_list.Managed(u8).init(std.heap.page_allocator);
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
        \\# ABI Performance Guide
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
        \\description: "Comprehensive glossary and concepts for ABI technology"
        \\keywords: ["vector database", "AI", "machine learning", "SIMD", "neural networks", "embeddings"]
        \\---
        \\
        \\# ABI Definitions Reference
        \\
        \\<div class="definition-search">
        \\  <input type="search" id="definition-search" placeholder="Search definitions..." autocomplete="off">
        \\  <div class="definition-categories">
        \\    <button class="category-filter active" data-category="all">All</button>
        \\    <button class="category-filter" data-category="database">Database</button>
        \\    <button class="category-filter" data-category="ai">AI/ML</button>
        \\    <button class="category-filter" data-category="performance">Performance</button>
        \\    <button class="category-filter" data-category="algorithms">Algorithms</button>
        \\    <button class="category-filter" data-category="system">System</button>
        \\  </div>
        \\</div>
        \\
        \\## 📊 Quick Reference Index
        \\
        \\| Term | Category | Definition |
        \\|------|----------|------------|
        \\| [Vector Database](#vector-database) | Database | Specialized storage for high-dimensional vectors |
        \\| [Embeddings](#embeddings) | AI/ML | Dense vector representations of data |
        \\| [HNSW](#hnsw-hierarchical-navigable-small-world) | Algorithms | Graph-based indexing for similarity search |
        \\| [Neural Network](#neural-network) | AI/ML | Computational model inspired by biological networks |
        \\| [SIMD](#simd-single-instruction-multiple-data) | Performance | Parallel processing technique |
        \\| [Cosine Similarity](#cosine-similarity) | Algorithms | Directional similarity metric |
        \\| [Backpropagation](#backpropagation) | AI/ML | Neural network training algorithm |
        \\| [Plugin Architecture](#plugin-architecture) | System | Extensible software design pattern |
        \\
        \\---
        \\
        \\## 🗄️ Database & Storage {#database}
        \\
        \\### Vector Database
        \\<div class="definition-card" data-category="database">
        \\
        \\A specialized database system designed to store, index, and search high-dimensional vectors efficiently. Unlike traditional relational databases that work with scalar values and structured data, vector databases are optimized for similarity search operations using various distance metrics.
        \\
        \\**Key Characteristics:**
        \\- **High-dimensional storage**: Efficiently handles vectors with hundreds to thousands of dimensions
        \\- **Similarity search**: Primary operation is finding vectors most similar to a query vector
        \\- **Specialized indexing**: Uses algorithms like HNSW, IVF, or LSH for fast approximate nearest neighbor search
        \\- **Scalability**: Designed to handle millions to billions of vectors with sub-linear search complexity
        \\- **Metadata support**: Associates additional information with each vector for filtering and retrieval
        \\
        \\**Common Use Cases:**
        \\- Semantic search in documents and images
        \\- Recommendation systems
        \\- Content-based filtering
        \\- Duplicate detection and deduplication
        \\- Anomaly detection in high-dimensional data
        \\
        \\**Performance Characteristics:**
        \\- Insert: ~2.5ms per vector (128 dimensions)
        \\- Search: ~13ms for k=10 in 10k vectors
        \\- Memory: ~512 bytes per vector + index overhead
        \\
        \\</div>
        \\
        \\### Embeddings
        \\<div class="definition-card" data-category="ai database">
        \\
        \\Dense, fixed-size vector representations that capture semantic meaning and relationships in a continuous mathematical space. Embeddings are typically generated by machine learning models and enable mathematical operations on complex data types.
        \\
        \\**Types of Embeddings:**
        \\- **Text embeddings**: Word2Vec, GloVe, BERT, sentence transformers
        \\- **Image embeddings**: CNN features, CLIP, vision transformers
        \\- **Audio embeddings**: Mel spectrograms, audio neural networks
        \\- **Graph embeddings**: Node2Vec, GraphSAGE for network data
        \\- **Multimodal embeddings**: CLIP, ALIGN for cross-modal understanding
        \\
        \\**Properties:**
        \\- **Dimensionality**: Typically 128-1024 dimensions for most applications
        \\- **Semantic similarity**: Similar concepts have similar vector representations
        \\- **Arithmetic operations**: Support vector arithmetic (king - man + woman ≈ queen)
        \\- **Transfer learning**: Pre-trained embeddings can be fine-tuned for specific tasks
        \\
        \\**Quality Metrics:**
        \\- **Cosine similarity**: Measures directional similarity
        \\- **Clustering coefficient**: How well similar items cluster together
        \\- **Downstream task performance**: Effectiveness in specific applications
        \\
        \\</div>
        \\
        \\### Indexing Algorithms
        \\<div class="definition-card" data-category="algorithms database">
        \\
        \\Specialized data structures and algorithms designed to accelerate similarity search in high-dimensional vector spaces. These algorithms trade exact accuracy for significant speed improvements.
        \\
        \\**Major Categories:**
        \\
        \\**Tree-based:**
        \\- **KD-Tree**: Binary tree partitioning, effective in low dimensions
        \\- **Ball Tree**: Hypersphere partitioning, better for higher dimensions
        \\- **R-Tree**: Rectangle-based partitioning for spatial data
        \\
        \\**Hash-based:**
        \\- **LSH (Locality Sensitive Hashing)**: Hash similar items to same buckets
        \\- **Random Projection**: Reduce dimensionality while preserving distances
        \\- **Product Quantization**: Divide vectors into subvectors for compression
        \\
        \\**Graph-based:**
        \\- **HNSW**: Hierarchical navigable small world graphs
        \\- **NSW**: Navigable small world graphs
        \\- **SPTAG**: Space Partition Tree and Graph
        \\
        \\**Inverted File (IVF):**
        \\- **IVF-Flat**: Partition space into Voronoi cells
        \\- **IVF-PQ**: Combine IVF with product quantization
        \\- **IVF-SQ**: Combine IVF with scalar quantization
        \\
        \\</div>
        \\
        \\### HNSW (Hierarchical Navigable Small World)
        \\<div class="definition-card" data-category="algorithms database">
        \\
        \\A state-of-the-art graph-based indexing algorithm that builds a multi-layered network of connections between vectors. It provides excellent performance for approximate nearest neighbor search with logarithmic time complexity.
        \\
        \\**Architecture:**
        \\- **Layer 0 (bottom)**: Contains all vectors with short-range connections to immediate neighbors
        \\- **Upper layers**: Contain exponentially fewer vectors with long-range connections for fast navigation
        \\- **Entry point**: Top-layer node where search begins
        \\- **Greedy search**: Navigate from top to bottom, always moving to closer neighbors
        \\
        \\**Key Parameters:**
        \\- **M (max connections)**: Maximum edges per node (16-64 typical)
        \\  - Higher M: Better recall, more memory usage
        \\  - Lower M: Faster construction, potential recall degradation
        \\- **efConstruction**: Candidate set size during index construction (200-800 typical)
        \\  - Higher ef: Better index quality, slower construction
        \\- **efSearch**: Candidate set size during search (varies by recall requirements)
        \\  - Higher ef: Better recall, slower search
        \\- **ml (level multiplier)**: Controls layer distribution (1/ln(2) ≈ 1.44)
        \\
        \\**Performance Characteristics:**
        \\- **Search complexity**: O(log N) on average
        \\- **Construction complexity**: O(N log N) on average
        \\- **Memory usage**: O(M × N) for connections
        \\- **Recall**: 95-99% achievable with proper parameter tuning
        \\
        \\**Advantages:**
        \\- High recall with fast search speed
        \\- Supports dynamic insertions and deletions
        \\- Good performance across various distance metrics
        \\- Robust to different data distributions
        \\
        \\</div>
        \\
        \\## 🧠 Artificial Intelligence & Machine Learning {#ai}
        \\
        \\### Neural Network
        \\<div class="definition-card" data-category="ai">
        \\
        \\A computational model inspired by biological neural networks, consisting of interconnected processing units (neurons) organized in layers. Each connection has an associated weight that determines the strength and direction of signal transmission.
        \\
        \\**Architecture Components:**
        \\- **Input layer**: Receives raw feature data (images, text, audio, etc.)
        \\- **Hidden layers**: Process and transform input through weighted connections and activation functions
        \\- **Output layer**: Produces final predictions, classifications, or generated content
        \\- **Connections**: Weighted links between neurons that are learned during training
        \\
        \\**Common Architectures:**
        \\- **Feedforward**: Information flows in one direction from input to output
        \\- **Convolutional (CNN)**: Specialized for image and spatial data processing
        \\- **Recurrent (RNN/LSTM/GRU)**: Designed for sequential data with memory
        \\- **Transformer**: Attention-based architecture for sequence modeling
        \\- **Autoencoder**: Encoder-decoder structure for dimensionality reduction
        \\- **Generative Adversarial (GAN)**: Two networks competing to generate realistic data
        \\
        \\**Activation Functions:**
        \\- **ReLU**: f(x) = max(0, x) - most common, prevents vanishing gradients
        \\- **Sigmoid**: f(x) = 1/(1 + e^(-x)) - outputs between 0 and 1
        \\- **Tanh**: f(x) = tanh(x) - outputs between -1 and 1
        \\- **Softmax**: Converts logits to probability distribution
        \\- **Swish/SiLU**: f(x) = x × sigmoid(x) - smooth, self-gating
        \\
        \\</div>
        \\
        \\### Backpropagation
        \\<div class="definition-card" data-category="ai algorithms">
        \\
        \\The fundamental algorithm for training neural networks by computing gradients of the loss function with respect to each parameter. It efficiently propagates error signals backwards through the network layers.
        \\
        \\**Algorithm Steps:**
        \\1. **Forward pass**: Input data flows through network to produce output
        \\2. **Loss computation**: Compare network output to target using loss function
        \\3. **Backward pass**: Compute gradients by applying chain rule from output to input
        \\4. **Parameter update**: Adjust weights using gradients and learning rate
        \\
        \\**Mathematical Foundation:**
        \\- **Chain rule**: ∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
        \\- **Gradient computation**: Efficient recursive calculation of partial derivatives
        \\- **Dynamic programming**: Reuses intermediate computations to avoid redundancy
        \\
        \\**Common Issues:**
        \\- **Vanishing gradients**: Gradients become very small in deep networks
        \\- **Exploding gradients**: Gradients become very large, causing instability
        \\- **Dead neurons**: Neurons that always output zero (common with ReLU)
        \\
        \\**Solutions:**
        \\- **Gradient clipping**: Limit gradient magnitude to prevent explosion
        \\- **Normalization**: Batch norm, layer norm to stabilize training
        \\- **Skip connections**: ResNet-style shortcuts to help gradient flow
        \\- **Learning rate scheduling**: Adaptive learning rates during training
        \\
        \\</div>
        \\
        \\### Gradient Descent
        \\<div class="definition-card" data-category="ai algorithms">
        \\
        \\An iterative optimization algorithm that minimizes a loss function by moving in the direction of steepest descent. It's the foundation for training most machine learning models.
        \\
        \\**Variants:**
        \\- **Batch Gradient Descent**: Uses entire dataset for each update
        \\  - Pros: Stable convergence, deterministic
        \\  - Cons: Slow for large datasets, may get stuck in local minima
        \\- **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
        \\  - Pros: Fast updates, can escape local minima
        \\  - Cons: Noisy convergence, requires careful tuning
        \\- **Mini-batch Gradient Descent**: Uses small batches (32-256 samples)
        \\  - Pros: Good balance of speed and stability
        \\  - Cons: Still requires hyperparameter tuning
        \\
        \\**Advanced Optimizers:**
        \\- **Momentum**: Accumulates gradients to accelerate convergence
        \\- **AdaGrad**: Adapts learning rate based on historical gradients
        \\- **RMSprop**: Improves AdaGrad with exponential moving average
        \\- **Adam**: Combines momentum and adaptive learning rates
        \\- **AdamW**: Adam with decoupled weight decay
        \\
        \\**Hyperparameters:**
        \\- **Learning rate (α)**: Step size for parameter updates (1e-4 to 1e-1)
        \\- **Momentum (β)**: Exponential decay for gradient accumulation (0.9-0.99)
        \\- **Weight decay**: L2 regularization to prevent overfitting (1e-5 to 1e-3)
        \\- **Learning rate schedule**: Decay strategy over training epochs
        \\
        \\</div>
        \\
        \\### Transformer Architecture
        \\<div class="definition-card" data-category="ai">
        \\
        \\A neural network architecture based entirely on attention mechanisms, revolutionizing natural language processing and extending to computer vision and other domains.
        \\
        \\**Key Components:**
        \\- **Multi-Head Attention**: Parallel attention mechanisms focusing on different aspects
        \\- **Position Encoding**: Adds positional information since attention is permutation-invariant
        \\- **Feed-Forward Networks**: Point-wise fully connected layers
        \\- **Layer Normalization**: Stabilizes training and improves convergence
        \\- **Residual Connections**: Skip connections around each sub-layer
        \\
        \\**Attention Mechanism:**
        \\- **Query (Q)**: What information are we looking for?
        \\- **Key (K)**: What information is available?
        \\- **Value (V)**: The actual information content
        \\- **Attention(Q,K,V) = softmax(QK^T/√d_k)V**
        \\
        \\**Variants:**
        \\- **BERT**: Bidirectional encoder for understanding tasks
        \\- **GPT**: Autoregressive decoder for generation tasks
        \\- **T5**: Text-to-text transfer transformer
        \\- **Vision Transformer (ViT)**: Applies transformer to image patches
        \\- **CLIP**: Contrastive learning of text and image representations
        \\
        \\</div>
        \\
        \\### Large Language Models (LLMs)
        \\<div class="definition-card" data-category="ai">
        \\
        \\Neural networks with billions to trillions of parameters trained on vast text corpora to understand and generate human-like text. They demonstrate emergent capabilities as they scale.
        \\
        \\**Characteristics:**
        \\- **Scale**: 1B to 175B+ parameters (GPT-3 has 175B parameters)
        \\- **Training data**: Hundreds of gigabytes to terabytes of text
        \\- **Emergent abilities**: Few-shot learning, reasoning, code generation
        \\- **In-context learning**: Learning from examples in the prompt
        \\
        \\**Training Stages:**
        \\1. **Pre-training**: Unsupervised learning on large text corpus
        \\2. **Fine-tuning**: Supervised learning on specific tasks
        \\3. **RLHF**: Reinforcement Learning from Human Feedback
        \\4. **Constitutional AI**: Training for harmlessness and helpfulness
        \\
        \\**Capabilities:**
        \\- Text generation and completion
        \\- Question answering and reasoning
        \\- Code generation and debugging
        \\- Language translation
        \\- Summarization and analysis
        \\- Creative writing and ideation
        \\
        \\</div>
        \\
        \\### Agent-Based Systems
        \\<div class="definition-card" data-category="ai system">
        \\
        \\Autonomous software entities that perceive their environment, make decisions, and take actions to achieve specific goals. Modern AI agents often incorporate large language models and various tools.
        \\
        \\**Agent Components:**
        \\- **Perception**: Sensors and inputs to observe environment state
        \\- **Decision making**: Logic, rules, or learned policies to choose actions
        \\- **Action**: Effectors and outputs to modify the environment
        \\- **Memory**: Storage of experiences, knowledge, and learned behaviors
        \\- **Communication**: Ability to interact with other agents or humans
        \\
        \\**Agent Types:**
        \\- **Reactive agents**: Respond directly to current perceptions without internal state
        \\- **Deliberative agents**: Plan sequences of actions using internal world models
        \\- **Learning agents**: Improve performance through experience and feedback
        \\- **Hybrid agents**: Combine reactive and deliberative components
        \\
        \\**Modern AI Agents:**
        \\- **Tool-using agents**: LLMs that can use external tools and APIs
        \\- **Code agents**: Generate and execute code to solve problems
        \\- **Conversational agents**: Chatbots and virtual assistants
        \\- **Planning agents**: Decompose complex tasks into subtasks
        \\- **Multi-agent systems**: Coordination between multiple AI agents
        \\
        \\**Design Patterns:**
        \\- **ReAct**: Reasoning and Acting with language models
        \\- **Chain of Thought**: Step-by-step reasoning prompts
        \\- **Tree of Thoughts**: Exploring multiple reasoning paths
        \\- **Reflection**: Self-evaluation and improvement mechanisms
        \\
        \\</div>
        \\
        \\## ⚡ Performance & Optimization {#performance}
        \\
        \\### SIMD (Single Instruction, Multiple Data)
        \\<div class="definition-card" data-category="performance">
        \\
        \\A parallel computing technique where a single instruction operates on multiple data points simultaneously. Modern CPUs have dedicated SIMD units that can process multiple numbers in one clock cycle.
        \\
        \\**Instruction Sets:**
        \\- **SSE (128-bit)**: 4 × float32 or 2 × float64 operations per instruction
        \\- **AVX (256-bit)**: 8 × float32 or 4 × float64 operations per instruction
        \\- **AVX-512 (512-bit)**: 16 × float32 or 8 × float64 operations per instruction
        \\- **ARM NEON**: ARM's SIMD instruction set for mobile processors
        \\
        \\**Benefits:**
        \\- **Throughput**: 4-16x more operations per clock cycle
        \\- **Memory bandwidth**: More efficient use of memory bus
        \\- **Energy efficiency**: Better performance per watt
        \\- **Cache efficiency**: Process more data with same cache footprint
        \\
        \\**Applications in Vector Databases:**
        \\- Vector addition, subtraction, multiplication
        \\- Dot product and cosine similarity calculations
        \\- Distance metric computations (Euclidean, Manhattan)
        \\- Matrix operations for neural networks
        \\- Quantization and compression operations
        \\
        \\**Programming Considerations:**
        \\- **Alignment**: Data must be aligned to vector width boundaries
        \\- **Data layout**: Array of Structures vs Structure of Arrays
        \\- **Compiler intrinsics**: Direct use of SIMD instructions
        \\- **Auto-vectorization**: Compiler automatic SIMD optimization
        \\
        \\</div>
        \\
        \\### Memory Hierarchy & Optimization
        \\<div class="definition-card" data-category="performance system">
        \\
        \\The hierarchical organization of computer memory systems, from fast but small caches to large but slow storage, and techniques to optimize data access patterns.
        \\
        \\**Memory Hierarchy (fastest to slowest):**
        \\- **CPU Registers**: ~1 cycle access, 32-64 registers
        \\- **L1 Cache**: ~1-3 cycles, 32-64KB per core, separate instruction/data
        \\- **L2 Cache**: ~10-20 cycles, 256KB-1MB per core, unified
        \\- **L3 Cache**: ~30-50 cycles, 8-64MB shared across cores
        \\- **Main Memory (RAM)**: ~100-300 cycles, GBs to TBs
        \\- **SSD Storage**: ~10-100μs, TBs capacity
        \\- **HDD Storage**: ~1-10ms, TBs capacity
        \\
        \\**Cache Properties:**
        \\- **Cache line size**: Typically 64 bytes
        \\- **Associativity**: Direct-mapped, set-associative, fully-associative
        \\- **Replacement policies**: LRU, random, pseudo-LRU
        \\- **Write policies**: Write-through, write-back
        \\
        \\**Optimization Techniques:**
        \\- **Spatial locality**: Access nearby memory locations
        \\- **Temporal locality**: Reuse recently accessed data
        \\- **Prefetching**: Load data before it's needed
        \\- **Cache blocking**: Restructure algorithms for cache efficiency
        \\- **Memory alignment**: Align data structures to cache line boundaries
        \\
        \\</div>
        \\
        \\### Batch Processing
        \\<div class="definition-card" data-category="performance">
        \\
        \\The practice of grouping multiple operations together to improve throughput and reduce per-operation overhead. Essential for achieving high performance in vector databases and machine learning.
        \\
        \\**Benefits:**
        \\- **Amortized overhead**: Function call and setup costs spread across multiple items
        \\- **Better memory locality**: Sequential access patterns improve cache performance
        \\- **SIMD utilization**: Process multiple items with vector instructions
        \\- **Reduced context switching**: Fewer kernel calls and mode switches
        \\- **Pipeline efficiency**: Keep execution units busy with continuous work
        \\
        \\**Optimal Batch Sizes:**
        \\- **Database inserts**: 100-1000 vectors (balance memory and throughput)
        \\- **Neural network training**: 32-512 samples (GPU memory dependent)
        \\- **SIMD operations**: Multiples of vector width (4, 8, 16 elements)
        \\- **I/O operations**: Page size multiples (4KB, 64KB blocks)
        \\
        \\**Implementation Strategies:**
        \\- **Buffering**: Accumulate items before processing
        \\- **Pipelining**: Overlap different stages of processing
        \\- **Work stealing**: Dynamic load balancing across threads
        \\- **Adaptive batching**: Adjust batch size based on system conditions
        \\
        \\</div>
        \\
        \\### Quantization
        \\<div class="definition-card" data-category="performance ai">
        \\
        \\Techniques for reducing the precision of numerical representations while preserving essential information. Critical for reducing memory usage and improving performance in large-scale systems.
        \\
        \\**Types of Quantization:**
        \\- **Scalar quantization**: Map continuous values to discrete levels
        \\- **Vector quantization**: Group similar vectors and represent with centroids
        \\- **Product quantization**: Decompose vectors into subvectors, quantize separately
        \\- **Binary quantization**: Extreme compression to 1-bit representations
        \\
        \\**Precision Levels:**
        \\- **INT8**: 8-bit integers, 4x memory reduction from FP32
        \\- **INT4**: 4-bit integers, 8x memory reduction, requires careful calibration
        \\- **INT1 (Binary)**: 1-bit representations, 32x reduction, significant accuracy loss
        \\- **Mixed precision**: Different precisions for different layers/operations
        \\
        \\**Quantization Strategies:**
        \\- **Post-training quantization**: Quantize after training with calibration data
        \\- **Quantization-aware training**: Include quantization in training process
        \\- **Dynamic quantization**: Adjust quantization parameters during inference
        \\- **Learned quantization**: Use neural networks to optimize quantization
        \\
        \\**Trade-offs:**
        \\- **Memory**: 2-32x reduction in storage requirements
        \\- **Speed**: Faster integer operations, reduced memory bandwidth
        \\- **Accuracy**: Some loss in precision, especially for aggressive quantization
        \\- **Compatibility**: Requires specialized hardware or software support
        \\
        \\</div>
        \\
        \\## 📐 Distance Metrics & Similarity {#algorithms}
        \\
        \\### Euclidean Distance
        \\<div class="definition-card" data-category="algorithms">
        \\
        \\The straight-line distance between two points in multidimensional space, corresponding to our intuitive notion of distance in physical space.
        \\
        \\**Mathematical Definition:**
        \\- **Formula**: d(a,b) = √(Σᵢ(aᵢ - bᵢ)²)
        \\- **Squared Euclidean**: Often used to avoid expensive square root: Σᵢ(aᵢ - bᵢ)²
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞), where 0 indicates identical vectors
        \\- **Symmetry**: d(a,b) = d(b,a)
        \\- **Triangle inequality**: d(a,c) ≤ d(a,b) + d(b,c)
        \\- **Positive definiteness**: d(a,b) = 0 if and only if a = b
        \\
        \\**Best Use Cases:**
        \\- **Image features**: Pixel values, color histograms
        \\- **Continuous measurements**: Physical measurements, sensor data
        \\- **Dense embeddings**: When magnitude matters (e.g., word embeddings)
        \\- **Gaussian distributions**: When data follows normal distribution
        \\
        \\**Computational Complexity:**
        \\- **Time**: O(d) where d is vector dimension
        \\- **SIMD optimization**: Highly vectorizable operation
        \\- **Memory access**: Sequential, cache-friendly
        \\
        \\</div>
        \\
        \\### Cosine Similarity
        \\<div class="definition-card" data-category="algorithms">
        \\
        \\Measures the cosine of the angle between two vectors, focusing on direction rather than magnitude. Widely used in text analysis and recommendation systems.
        \\
        \\**Mathematical Definition:**
        \\- **Formula**: similarity(a,b) = (a·b) / (||a|| × ||b||)
        \\- **Cosine distance**: 1 - cosine_similarity(a,b)
        \\- **Dot product**: a·b = Σᵢ(aᵢ × bᵢ)
        \\- **Magnitude**: ||a|| = √(Σᵢaᵢ²)
        \\
        \\**Properties:**
        \\- **Range**: [-1, 1] where 1 = same direction, 0 = orthogonal, -1 = opposite
        \\- **Magnitude invariant**: Only considers direction, not length
        \\- **Normalized vectors**: For unit vectors, cosine similarity equals dot product
        \\- **Symmetry**: cosine_similarity(a,b) = cosine_similarity(b,a)
        \\
        \\**Best Use Cases:**
        \\- **Text embeddings**: TF-IDF vectors, word/sentence embeddings
        \\- **Sparse features**: High-dimensional sparse vectors
        \\- **Recommendation systems**: User-item preferences
        \\- **Document similarity**: When document length shouldn't matter
        \\
        \\**Optimization Techniques:**
        \\- **Pre-normalization**: Store normalized vectors to simplify computation
        \\- **SIMD dot product**: Vectorized multiplication and summation
        \\- **Approximate methods**: Random sampling for very high dimensions
        \\
        \\</div>
        \\
        \\### Manhattan Distance (L1 Norm)
        \\<div class="definition-card" data-category="algorithms">
        \\
        \\The sum of absolute differences between corresponding elements, named after Manhattan's grid-like street layout where you can only travel along perpendicular streets.
        \\
        \\**Mathematical Definition:**
        \\- **Formula**: d(a,b) = Σᵢ|aᵢ - bᵢ|
        \\- **Also known as**: L1 distance, taxicab distance, city block distance
        \\
        \\**Properties:**
        \\- **Range**: [0, ∞), where 0 indicates identical vectors
        \\- **Robustness**: Less sensitive to outliers than Euclidean distance
        \\- **Sparsity inducing**: Tends to produce sparse solutions in optimization
        \\- **Convex**: Forms diamond-shaped unit balls in 2D space
        \\
        \\**Best Use Cases:**
        \\- **Sparse data**: High-dimensional sparse vectors
        \\- **Robust statistics**: When outliers are present
        \\- **Feature selection**: L1 regularization promotes sparsity
        \\- **Discrete features**: Categorical or count data
        \\
        \\**Computational Advantages:**
        \\- **No squares**: Avoids expensive multiplication operations
        \\- **Integer arithmetic**: Can work with integer representations
        \\- **Bounded gradients**: Useful for optimization algorithms
        \\
        \\</div>
        \\
        \\### Hamming Distance
        \\<div class="definition-card" data-category="algorithms">
        \\
        \\The number of positions where corresponding elements differ, originally defined for binary strings but extended to other discrete alphabets.
        \\
        \\**Mathematical Definition:**
        \\- **Binary vectors**: Number of bit positions where vectors differ
        \\- **General case**: Number of positions where aᵢ ≠ bᵢ
        \\- **Normalized**: Divide by vector length for similarity score
        \\
        \\**Properties:**
        \\- **Range**: [0, d] where d is vector dimension
        \\- **Discrete**: Only integer values possible
        \\- **Symmetric**: Hamming(a,b) = Hamming(b,a)
        \\- **Triangle inequality**: Forms valid metric space
        \\
        \\**Applications:**
        \\- **Binary embeddings**: Locality sensitive hashing outputs
        \\- **Error correction**: Coding theory and data transmission
        \\- **Fingerprinting**: Perceptual hashing for duplicate detection
        \\- **Genetics**: DNA sequence comparison
        \\
        \\**Computational Efficiency:**
        \\- **Bit operations**: XOR followed by population count
        \\- **Hardware support**: Many CPUs have POPCNT instruction
        \\- **Parallel computation**: Highly parallelizable across bits
        \\
        \\</div>
        \\
        \\## 🏗️ System Architecture {#system}
        \\
        \\### Plugin Architecture
        \\<div class="definition-card" data-category="system">
        \\
        \\A software design pattern that enables extending core functionality through dynamically loaded, modular components. Plugins are independent units that implement well-defined interfaces.
        \\
        \\**Core Components:**
        \\- **Plugin interface**: Contract defining how plugins interact with the host
        \\- **Plugin manager**: Loads, unloads, and manages plugin lifecycle
        \\- **Host application**: Core system that provides plugin infrastructure
        \\- **Plugin registry**: Catalog of available plugins and their capabilities
        \\
        \\**Implementation Approaches:**
        \\- **Dynamic libraries**: Shared objects (.so, .dll, .dylib) loaded at runtime
        \\- **Process isolation**: Plugins run in separate processes with IPC
        \\- **Scripting engines**: Embed interpreters (Python, Lua, JavaScript)
        \\- **WebAssembly**: Sandboxed plugins with near-native performance
        \\- **Container-based**: Docker containers for maximum isolation
        \\
        \\**Benefits:**
        \\- **Modularity**: Keep core system lean, add features as needed
        \\- **Extensibility**: Third-party developers can add functionality
        \\- **Isolation**: Plugin failures don't crash the host system
        \\- **Hot-swapping**: Load/unload plugins without system restart
        \\- **Versioning**: Different plugin versions can coexist
        \\
        \\**Challenges:**
        \\- **Interface stability**: API changes can break existing plugins
        \\- **Security**: Malicious plugins can compromise system
        \\- **Performance**: Inter-plugin communication overhead
        \\- **Dependency management**: Complex dependency graphs
        \\
        \\</div>
        \\
        \\### Memory Management Strategies
        \\<div class="definition-card" data-category="system performance">
        \\
        \\Techniques for efficiently allocating, using, and deallocating memory in high-performance applications, crucial for vector databases handling large datasets.
        \\
        \\**Allocation Strategies:**
        \\- **Stack allocation**: Fast automatic cleanup, limited size, LIFO order
        \\- **Heap allocation**: Flexible size, manual management, fragmentation risk
        \\- **Pool allocation**: Pre-allocate fixed-size blocks, fast allocation/deallocation
        \\- **Arena allocation**: Bulk allocation with batch cleanup, minimal overhead
        \\- **Slab allocation**: Kernel-style allocator for objects of similar size
        \\
        \\**Memory Patterns:**
        \\- **RAII (Resource Acquisition Is Initialization)**: Tie resource lifetime to object scope
        \\- **Reference counting**: Automatic cleanup when no references remain
        \\- **Garbage collection**: Automatic memory management with performance trade-offs
        \\- **Copy-on-write**: Share memory until modification is needed
        \\
        \\**Optimization Techniques:**
        \\- **Memory pools**: Reduce allocation overhead for frequent operations
        \\- **Object recycling**: Reuse expensive-to-create objects
        \\- **Alignment**: Ensure data alignment for optimal access patterns
        \\- **Prefaulting**: Touch memory pages to ensure they're resident
        \\
        \\**Monitoring and Debugging:**
        \\- **Memory profiling**: Track allocation patterns and leaks
        \\- **Valgrind**: Memory error detection for C/C++ programs
        \\- **AddressSanitizer**: Runtime memory error detector
        \\- **Custom allocators**: Track application-specific memory usage
        \\
        \\</div>
        \\
        \\### Caching Strategies
        \\<div class="definition-card" data-category="system performance">
        \\
        \\Techniques for storing frequently accessed data in faster storage layers to improve system performance by exploiting temporal and spatial locality.
        \\
        \\**Cache Hierarchies:**
        \\- **CPU caches**: L1/L2/L3 hardware caches in processor
        \\- **Application caches**: In-memory data structures (hash tables, trees)
        \\- **Database caches**: Buffer pools for frequently accessed pages
        \\- **Web caches**: CDNs and reverse proxies for distributed systems
        \\- **Disk caches**: SSD tier for frequently accessed data
        \\
        \\**Replacement Policies:**
        \\- **LRU (Least Recently Used)**: Evict items not accessed recently
        \\- **LFU (Least Frequently Used)**: Evict items accessed infrequently
        \\- **FIFO (First In, First Out)**: Simple queue-based eviction
        \\- **Random**: Simple but often effective for uniform access patterns
        \\- **ARC (Adaptive Replacement Cache)**: Adapts between recency and frequency
        \\
        \\**Cache Strategies:**
        \\- **Write-through**: Immediately write to both cache and backing store
        \\- **Write-back**: Delay writes to backing store, better performance
        \\- **Write-around**: Skip cache for writes, avoid cache pollution
        \\- **Refresh-ahead**: Proactively refresh expired entries
        \\
        \\**Performance Considerations:**
        \\- **Hit ratio**: Percentage of requests served from cache
        \\- **Miss penalty**: Cost of loading data from slower storage
        \\- **Cache coherence**: Consistency across multiple cache instances
        \\- **Working set size**: Amount of data actively accessed
        \\
        \\</div>
        \\
        \\## 📊 Performance Metrics & Evaluation {#performance}
        \\
        \\### Throughput vs Latency
        \\<div class="definition-card" data-category="performance">
        \\
        \\Two fundamental performance metrics that often require trade-offs in system design. Understanding both is crucial for optimizing vector database performance.
        \\
        \\**Throughput:**
        \\- **Definition**: Number of operations completed per unit time
        \\- **Units**: Operations/second, requests/second, GB/second
        \\- **Optimization**: Batching, pipelining, parallelism
        \\- **Measurement**: Total operations / total time
        \\
        \\**Latency:**
        \\- **Definition**: Time required to complete a single operation
        \\- **Units**: Milliseconds, microseconds, nanoseconds
        \\- **Types**: Mean, median, P95, P99, tail latency
        \\- **Optimization**: Caching, indexing, algorithm optimization
        \\
        \\**Trade-offs:**
        \\- **High throughput**: May increase individual operation latency
        \\- **Low latency**: May reduce overall system throughput
        \\- **Batch processing**: Improves throughput at cost of latency
        \\- **Real-time systems**: Often prioritize latency over throughput
        \\
        \\**Little's Law:**
        \\- **Formula**: Average latency = Average queue length / Average throughput
        \\- **Application**: Helps understand system capacity and performance
        \\
        \\</div>
        \\
        \\### Recall and Precision in Vector Search
        \\<div class="definition-card" data-category="algorithms performance">
        \\
        \\Quality metrics for evaluating approximate nearest neighbor search algorithms, measuring how well they find relevant results compared to exact search.
        \\
        \\**Recall:**
        \\- **Definition**: Fraction of true nearest neighbors found by the algorithm
        \\- **Formula**: Recall = |Retrieved ∩ Relevant| / |Relevant|
        \\- **Range**: [0, 1] where 1 = perfect recall (found all true neighbors)
        \\- **Trade-off**: Higher recall usually requires more computation
        \\
        \\**Precision:**
        \\- **Definition**: Fraction of retrieved results that are true nearest neighbors
        \\- **Formula**: Precision = |Retrieved ∩ Relevant| / |Retrieved|
        \\- **Range**: [0, 1] where 1 = perfect precision (no false positives)
        \\- **Context**: Less commonly used in k-NN search (fixed k)
        \\
        \\**Evaluation Methodology:**
        \\- **Ground truth**: Exact k-NN results computed with brute force
        \\- **Test queries**: Representative sample of real-world queries
        \\- **Multiple k values**: Evaluate performance for different neighborhood sizes
        \\- **Parameter sweeps**: Test different algorithm configurations
        \\
        \\**Practical Considerations:**
        \\- **Acceptable recall**: Often 90-95% sufficient for most applications
        \\- **Speed-accuracy trade-off**: Balance recall against query latency
        \\- **Index parameters**: Tune to achieve target recall efficiently
        \\
        \\</div>
        \\
        \\---
        \\
        \\<style>
        \\.definition-search {
        \\  margin: 2rem 0;
        \\  padding: 1.5rem;
        \\  background: var(--color-canvas-subtle);
        \\  border-radius: 8px;
        \\  border: 1px solid var(--color-border-default);
        \\}
        \\
        \\.definition-search input {
        \\  width: 100%;
        \\  padding: 0.75rem 1rem;
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 6px;
        \\  font-size: 1rem;
        \\  margin-bottom: 1rem;
        \\}
        \\
        \\.definition-categories {
        \\  display: flex;
        \\  gap: 0.5rem;
        \\  flex-wrap: wrap;
        \\}
        \\
        \\.category-filter {
        \\  padding: 0.5rem 1rem;
        \\  border: 1px solid var(--color-border-default);
        \\  background: var(--color-canvas-default);
        \\  border-radius: 4px;
        \\  cursor: pointer;
        \\  transition: all 0.2s ease;
        \\}
        \\
        \\.category-filter:hover {
        \\  background: var(--color-canvas-subtle);
        \\}
        \\
        \\.category-filter.active {
        \\  background: var(--color-accent-emphasis);
        \\  color: white;
        \\  border-color: var(--color-accent-emphasis);
        \\}
        \\
        \\.definition-card {
        \\  margin: 1.5rem 0;
        \\  padding: 1.5rem;
        \\  border: 1px solid var(--color-border-default);
        \\  border-radius: 8px;
        \\  background: var(--color-canvas-default);
        \\  transition: box-shadow 0.2s ease;
        \\}
        \\
        \\.definition-card:hover {
        \\  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        \\}
        \\
        \\.definition-card h3 {
        \\  margin-top: 0;
        \\  color: var(--color-accent-fg);
        \\  border-bottom: 2px solid var(--color-accent-emphasis);
        \\  padding-bottom: 0.5rem;
        \\}
        \\
        \\.definition-card strong {
        \\  color: var(--color-fg-default);
        \\}
        \\
        \\.definition-card ul, .definition-card ol {
        \\  margin: 1rem 0;
        \\  padding-left: 1.5rem;
        \\}
        \\
        \\.definition-card li {
        \\  margin: 0.5rem 0;
        \\}
        \\
        \\.definition-card code {
        \\  background: var(--color-canvas-subtle);
        \\  padding: 0.2rem 0.4rem;
        \\  border-radius: 3px;
        \\  font-size: 0.9rem;
        \\}
        \\
        \\@media (max-width: 768px) {
        \\  .definition-categories {
        \\    flex-direction: column;
        \\  }
        \\  
        \\  .category-filter {
        \\    text-align: center;
        \\  }
        \\  
        \\  .definition-card {
        \\    margin: 1rem -1rem;
        \\    border-radius: 0;
        \\    border-left: none;
        \\    border-right: none;
        \\  }
        \\}
        \\</style>
        \\
        \\<script>
        \\document.addEventListener('DOMContentLoaded', function() {
        \\  const searchInput = document.getElementById('definition-search');
        \\  const categoryFilters = document.querySelectorAll('.category-filter');
        \\  const definitionCards = document.querySelectorAll('.definition-card');
        \\  
        \\  let currentCategory = 'all';
        \\  
        \\  // Search functionality
        \\  searchInput.addEventListener('input', function() {
        \\    const query = this.value.toLowerCase();
        \\    filterDefinitions(query, currentCategory);
        \\  });
        \\  
        \\  // Category filtering
        \\  categoryFilters.forEach(button => {
        \\    button.addEventListener('click', function() {
        \\      // Update active state
        \\      categoryFilters.forEach(b => b.classList.remove('active'));
        \\      this.classList.add('active');
        \\      
        \\      currentCategory = this.dataset.category;
        \\      const query = searchInput.value.toLowerCase();
        \\      filterDefinitions(query, currentCategory);
        \\    });
        \\  });
        \\  
        \\  function filterDefinitions(searchQuery, category) {
        \\    definitionCards.forEach(card => {
        \\      const text = card.textContent.toLowerCase();
        \\      const categories = card.dataset.category ? card.dataset.category.split(' ') : [];
        \\      
        \\      const matchesSearch = !searchQuery || text.includes(searchQuery);
        \\      const matchesCategory = category === 'all' || categories.includes(category);
        \\      
        \\      if (matchesSearch && matchesCategory) {
        \\        card.style.display = 'block';
        \\        // Highlight search terms
        \\        if (searchQuery) {
        \\          highlightSearchTerms(card, searchQuery);
        \\        }
        \\      } else {
        \\        card.style.display = 'none';
        \\      }
        \\    });
        \\  }
        \\  
        \\  function highlightSearchTerms(element, query) {
        \\    // Simple highlighting implementation
        \\    // In a real implementation, you'd want more sophisticated text highlighting
        \\    const textNodes = getTextNodes(element);
        \\    textNodes.forEach(node => {
        \\      if (node.textContent.toLowerCase().includes(query)) {
        \\        const parent = node.parentNode;
        \\        const regex = new RegExp(`(${query})`, 'gi');
        \\        const highlighted = node.textContent.replace(regex, '<mark>$1</mark>');
        \\        const wrapper = document.createElement('span');
        \\        wrapper.innerHTML = highlighted;
        \\        parent.replaceChild(wrapper, node);
        \\      }
        \\    });
        \\  }
        \\  
        \\  function getTextNodes(element) {
        \\    const textNodes = [];
        \\    const walker = document.createTreeWalker(
        \\      element,
        \\      NodeFilter.SHOW_TEXT,
        \\      null,
        \\      false
        \\    );
        \\    
        \\    let node;
        \\    while (node = walker.nextNode()) {
        \\      textNodes.push(node);
        \\    }
        \\    return textNodes;
        \\  }
        \\});
        \\</script>
        \\
    ;

    try file.writeAll(content);
}

/// Generate GitHub Actions workflow for automated documentation deployment
fn generateGitHubActionsWorkflow(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const file = try std.fs.cwd().createFile(".github/workflows/deploy_docs.yml", .{});
    defer file.close();

    const content =
        \\name: Deploy Documentation to GitHub Pages
        \\
        \\on:
        \\  push:
        \\    branches: [main, master]
        \\    paths:
        \\      - 'src/**'
        \\      - 'docs/**'
        \\      - 'tools/docs_generator.zig'
        \\      - '.github/workflows/deploy_docs.yml'
        \\  pull_request:
        \\    branches: [main, master]
        \\    paths:
        \\      - 'src/**'
        \\      - 'docs/**'
        \\      - 'tools/docs_generator.zig'
        \\  workflow_dispatch:
        \\
        \\permissions:
        \\  contents: read
        \\  pages: write
        \\  id-token: write
        \\
        \\# Allow only one concurrent deployment
        \\concurrency:
        \\  group: "pages"
        \\  cancel-in-progress: false
        \\
        \\jobs:
        \\  # Build documentation
        \\  build:
        \\    runs-on: ubuntu-latest
        \\    steps:
        \\      - name: Checkout repository
        \\        uses: actions/checkout@v4
        \\        with:
        \\          fetch-depth: 0
        \\
        \\      - name: Setup Zig
        \\        uses: goto-bus-stop/setup-zig@v2
        \\        with:
        \\          version: 0.12.0
        \\
        \\      - name: Cache Zig dependencies
        \\        uses: actions/cache@v3
        \\        with:
        \\          path: |
        \\            ~/.cache/zig
        \\            zig-cache
        \\          key: ${{ runner.os }}-zig-${{ hashFiles('build.zig.zon') }}
        \\          restore-keys: |
        \\            ${{ runner.os }}-zig-
        \\
        \\      - name: Build project
        \\        run: zig build
        \\
        \\      - name: Generate documentation
        \\        run: zig run tools/docs_generator.zig
        \\
        \\      - name: Setup Pages
        \\        uses: actions/configure-pages@v3
        \\
        \\      - name: Upload artifact
        \\        uses: actions/upload-pages-artifact@v2
        \\        with:
        \\          path: './docs'
        \\
        \\  # Deploy to GitHub Pages
        \\  deploy:
        \\    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master'
        \\    environment:
        \\      name: github-pages
        \\      url: ${{ steps.deployment.outputs.page_url }}
        \\    runs-on: ubuntu-latest
        \\    needs: build
        \\    steps:
        \\      - name: Deploy to GitHub Pages
        \\        id: deployment
        \\        uses: actions/deploy-pages@v2
        \\
    ;

    try file.writeAll(content);
}
