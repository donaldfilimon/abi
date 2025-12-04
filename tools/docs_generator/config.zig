const std = @import("std");

pub fn generateNoJekyll(_: std.mem.Allocator) !void {
    // Ensure GitHub Pages does not run Jekyll
    var file = try std.fs.cwd().createFile("docs/.nojekyll", .{ .truncate = true });
    defer file.close();
}

/// Generate Jekyll configuration for GitHub Pages
pub fn generateJekyllConfig(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_config.yml", .{ .truncate = true });
    defer file.close();

    const content =
        \\# ABI Documentation - Jekyll Configuration
        \\title: "ABI Documentation"
        \\description: "High-performance vector database with AI capabilities"
        \\url: "https://donaldfilimon.github.io"
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
pub fn generateGitHubPagesLayout(_: std.mem.Allocator) !void {
    const cwd = std.fs.cwd();
    const layout_path = "docs/_layouts/documentation.html";

    if (cwd.access(layout_path, .{})) |_| {
        return;
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    const file = try cwd.createFile(layout_path, .{ .truncate = true });
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
        \\  <script>window.__DOCS_BASEURL = {{ site.baseurl | default: '' | jsonify }};</script>
        \\  <script src="{{ '/assets/js/search.js' | relative_url }}" defer></script>
        \\</head>
        \\<body data-baseurl="{{ site.baseurl | default: '' }}">
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
pub fn generateNavigationData(_: std.mem.Allocator) !void {
    const file = try std.fs.cwd().createFile("docs/_data/navigation.yml", .{ .truncate = true });
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
pub fn generateSEOMetadata(_: std.mem.Allocator) !void {
    // Generate sitemap.xml
    const sitemap_file = try std.fs.cwd().createFile("docs/sitemap.xml", .{ .truncate = true });
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
    const robots_file = try std.fs.cwd().createFile("docs/robots.txt", .{ .truncate = true });
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

/// Generate CSS and JavaScript assets for GitHub Pages
pub fn generateGitHubPagesAssets(allocator: std.mem.Allocator) !void {
    _ = allocator;

    // Generate enhanced CSS
    const css_file = try std.fs.cwd().createFile("docs/assets/css/documentation.css", .{ .truncate = true });
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
    const js_file = try std.fs.cwd().createFile("docs/assets/js/documentation.js", .{ .truncate = true });
    defer js_file.close();

    const js_content =
        \\// Enhanced GitHub Pages Documentation JavaScript
        \\(function() {
        \\  'use strict';
        \\
        \\  const baseUrl = resolveBaseUrl();
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
        \\      const toc = document.getElementById('toc');
        \\      if (toc) toc.style.display = 'none';
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
        \\  function resolveBaseUrl() {
        \\    const fromWindow = typeof window !== 'undefined' && typeof window.__DOCS_BASEURL === 'string'
        \\      ? window.__DOCS_BASEURL
        \\      : '';
        \\    const fromBody = document.body ? (document.body.getAttribute('data-baseurl') || '') : '';
        \\    const raw = fromBody || fromWindow || '';
        \\    if (!raw || raw === '/') return '';
        \\    return raw.endsWith('/') ? raw.slice(0, -1) : raw;
        \\  }
        \\
        \\  function buildUrl(path) {
        \\    if (!path) return baseUrl || '';
        \\    const normalized = path.startsWith('/') ? path : `/${path}`;
        \\    return `${baseUrl}${normalized}`;
        \\  }
        \\
        \\  function escapeHtml(text) {
        \\    return String(text)
        \\      .replace(/&/g, '&amp;')
        \\      .replace(/</g, '&lt;')
        \\      .replace(/>/g, '&gt;')
        \\      .replace(/"/g, '&quot;')
        \\      .replace(/'/g, '&#39;');
        \\  }
        \\
        \\  function escapeAttribute(text) {
        \\    return escapeHtml(text).replace(/`/g, '&#96;');
        \\  }
        \\
        \\  function escapeRegExp(text) {
        \\    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
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
        \\    if (typeof window !== 'undefined' && Array.isArray(window.__ABI_SEARCH_DATA)) {
        \\      searchData = window.__ABI_SEARCH_DATA.slice();
        \\    } else {
        \\      fetch(buildUrl('/generated/search_index.json'))
        \\        .then(response => response.json())
        \\        .then(data => {
        \\          if (Array.isArray(data)) {
        \\            searchData = data;
        \\            window.__ABI_SEARCH_DATA = data;
        \\          }
        \\        })
        \\        .catch(error => {
        \\          console.warn('Search index not available:', error);
        \\        });
        \\    }
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
        \\    searchResults.addEventListener('click', function(event) {
        \\      const target = event.target.closest('.search-result-item');
        \\      if (!target) return;
        \\
        \\      if (target.dataset.file) {
        \\        navigateToPage(target.dataset.file);
        \\        searchResults.classList.add('hidden');
        \\        return;
        \\      }
        \\
        \\      if (target.dataset.suggestion) {
        \\        applySuggestion(target.dataset.suggestion, searchInput);
        \\        searchResults.classList.add('hidden');
        \\      }
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
        \\      searchResults.innerHTML = '<div class="search-result-item" data-empty="true">No results found</div>';
        \\    } else {
        \\      const safeQuery = query ? escapeRegExp(query) : '';
        \\      searchResults.innerHTML = results.map(result => {
        \\        const file = escapeAttribute(result.file);
        \\        const title = highlightText(result.title, safeQuery);
        \\        const excerpt = highlightText(result.excerpt, safeQuery);
        \\        return `
        \\          <div class="search-result-item" data-file="${file}">
        \\            <div class="search-result-title">${title}</div>
        \\            <div class="search-result-excerpt">${excerpt}</div>
        \\          </div>
        \\        `;
        \\      }).join('');
        \\    }
        \\
        \\    searchResults.classList.remove('hidden');
        \\  }
        \\
        \\  function highlightText(text, escapedQuery) {
        \\    const safeText = escapeHtml(text);
        \\    if (!escapedQuery) return safeText;
        \\    const regex = new RegExp(`(${escapedQuery})`, 'gi');
        \\    return safeText.replace(regex, '<strong>$1</strong>');
        \\  }
        \\
        \\  function navigateToPage(file) {
        \\    if (!file) return;
        \\    window.location.href = buildUrl(file);
        \\  }
        \\
        \\  function applySuggestion(suggestion, input) {
        \\    if (!input) return;
        \\    input.value = suggestion;
        \\    input.dispatchEvent(new Event('input', { bubbles: true }));
        \\    input.focus();
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
        \\          const entries = performance.getEntriesByType('navigation');
        \\          if (!entries || entries.length === 0) return;
        \\          const perfData = entries[0];
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
        \\    const performanceMarkers = Array.from(document.querySelectorAll('code')).filter(code => {
        \\      const text = code.textContent || '';
        \\      return text.includes('~') || text.includes('ms') || text.includes('μs');
        \\    });
        \\
        \\    performanceMarkers.forEach(function(marker) {
        \\      if (marker.textContent.includes('~')) {
        \\        const badge = document.createElement('span');
        \\        badge.className = 'performance-badge';
        \\        badge.textContent = 'PERF';
        \\        marker.parentElement.insertBefore(badge, marker.nextSibling);
        \\      }
        \\    });
        \\
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
        \\n
    ;

    try js_file.writeAll(js_content);

    // Generate search JavaScript
    const search_js_file = try std.fs.cwd().createFile("docs/assets/js/search.js", .{ .truncate = true });
    defer search_js_file.close();

    const search_js_content =
        \\// Advanced search functionality for GitHub Pages
        \\(function() {
        \\  'use strict';
        \\
        \\  const baseUrl = resolveBaseUrl();
        \\  let searchIndex = [];
        \\
        \\  function resolveBaseUrl() {
        \\    const fromWindow = typeof window !== 'undefined' && typeof window.__DOCS_BASEURL === 'string'
        \\      ? window.__DOCS_BASEURL
        \\      : '';
        \\    const fromBody = document.body ? (document.body.getAttribute('data-baseurl') || '') : '';
        \\    const raw = fromBody || fromWindow || '';
        \\    if (!raw || raw === '/') return '';
        \\    return raw.endsWith('/') ? raw.slice(0, -1) : raw;
        \\  }
        \\
        \\  function buildUrl(path) {
        \\    if (!path) return baseUrl || '';
        \\    const normalized = path.startsWith('/') ? path : `/${path}`;
        \\    return `${baseUrl}${normalized}`;
        \\  }
        \\
        \\  function initializeAdvancedSearch() {
        \\    const existingData = Array.isArray(window.__ABI_SEARCH_DATA) ? window.__ABI_SEARCH_DATA : null;
        \\    if (existingData) {
        \\      searchIndex = existingData;
        \\      setupSearchInterface();
        \\      return;
        \\    }
        \\
        \\    fetch(buildUrl('/generated/search_index.json'))
        \\      .then(response => response.json())
        \\      .then(data => {
        \\        searchIndex = Array.isArray(data) ? data : [];
        \\        if (searchIndex.length > 0) {
        \\          window.__ABI_SEARCH_DATA = searchIndex;
        \\        }
        \\        setupSearchInterface();
        \\      })
        \\      .catch(error => {
        \\        console.warn('Search functionality unavailable:', error);
        \\        setupSearchInterface();
        \\      });
        \\  }
        \\
        \\  function setupSearchInterface() {
        \\    const searchInput = document.getElementById('search-input');
        \\    const searchResults = document.getElementById('search-results');
        \\    if (!searchInput || !searchResults) return;
        \\
        \\    document.addEventListener('keydown', function(e) {
        \\      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        \\        e.preventDefault();
        \\        searchInput.focus();
        \\        searchInput.select();
        \\      }
        \\
        \\      if (e.key === 'Escape' && document.activeElement === searchInput) {
        \\        searchInput.value = '';
        \\        hideSearchResults(searchResults);
        \\      }
        \\    });
        \\
        \\    searchInput.addEventListener('focus', function() {
        \\      if (this.value.trim() === '') {
        \\        showSearchSuggestions(searchResults);
        \\      }
        \\    });
        \\
        \\    searchInput.addEventListener('input', function() {
        \\      if (this.value.trim() === '') {
        \\        showSearchSuggestions(searchResults);
        \\      }
        \\    });
        \\
        \\    searchResults.addEventListener('mousedown', function(event) {
        \\      if (event.target.closest('.search-result-item')) {
        \\        event.preventDefault();
        \\      }
        \\    });
        \\  }
        \\
        \\  function hideSearchResults(container) {
        \\    if (container) {
        \\      container.classList.add('hidden');
        \\    }
        \\  }
        \\
        \\  function showSearchSuggestions(container) {
        \\    if (!container) return;
        \\
        \\    const suggestions = buildSuggestionList();
        \\    container.innerHTML = suggestions.map(suggestion => `
        \\      <div class="search-result-item suggestion" data-suggestion="${escapeHtml(suggestion)}">
        \\        <div class="search-result-title">💡 ${escapeHtml(suggestion)}</div>
        \\        <div class="search-result-excerpt">Press Enter to search</div>
        \\      </div>
        \\    `).join('');
        \\
        \\    container.classList.remove('hidden');
        \\  }
        \\
        \\  function buildSuggestionList() {
        \\    if (!Array.isArray(searchIndex) || searchIndex.length === 0) {
        \\      return [
        \\        'database API',
        \\        'neural networks',
        \\        'SIMD operations',
        \\        'performance guide',
        \\        'plugin system',
        \\        'vector search',
        \\        'machine learning'
        \\      ];
        \\    }
        \\
        \\    const titles = [];
        \\    for (const item of searchIndex) {
        \\      if (item && item.title && !titles.includes(item.title)) {
        \\        titles.push(item.title);
        \\      }
        \\      if (titles.length >= 7) break;
        \\    }
        \\    return titles;
        \\  }
        \\
        \\  function escapeHtml(text) {
        \\    return String(text)
        \\      .replace(/&/g, '&amp;')
        \\      .replace(/</g, '&lt;')
        \\      .replace(/>/g, '&gt;')
        \\      .replace(/"/g, '&quot;')
        \\      .replace(/'/g, '&#39;');
        \\  }
        \\
        \\  if (document.readyState === 'loading') {
        \\    document.addEventListener('DOMContentLoaded', initializeAdvancedSearch);
        \\  } else {
        \\    initializeAdvancedSearch();
        \\  }
        \\
        \\})();
        \\n
    ;

    try search_js_file.writeAll(search_js_content);
}

/// Generate GitHub Actions workflow for automated documentation deployment
pub fn generateGitHubActionsWorkflow(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const file = try std.fs.cwd().createFile(".github/workflows/deploy_docs.yml", .{ .truncate = true });
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
        \\      - 'src/tools/docs_generator/**'
        \\      - '.github/workflows/deploy_docs.yml'
        \\  pull_request:
        \\    branches: [main, master]
        \\    paths:
        \\      - 'src/**'
        \\      - 'docs/**'
        \\      - 'src/tools/docs_generator/**'
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
        \\        uses: mlugg/setup-zig@v2
        \\        with:
        \\          version: 0.16.0-dev.254+6dd0270a1
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
        \\        run: zig run src/tools/docs_generator/main.zig
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
