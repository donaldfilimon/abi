#!/usr/bin/env python3
"""
Generate static HTML documentation from markdown files.
"""

import os
import re
from pathlib import Path
from datetime import datetime


# Simple markdown parser
def parse_markdown(content, title=None):
    """Parse markdown content to HTML."""
    html = content

    # Headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Code blocks
    html = re.sub(
        r"```(\w*)\n(.+?)\n```",
        r'<pre><code class="language-\1">\2</code></pre>',
        html,
        flags=re.DOTALL,
    )

    # Inline code
    html = re.sub(r"`([^`]+?)`", r"<code>\1</code>", html)

    # Links
    html = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', html)

    # Lists (simple)
    html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    html = re.sub(r"(<li>.*</li>\n?)+", lambda m: f"<ul>{m.group(0)}</ul>", html)

    # Paragraphs
    html = re.sub(r"\n\n([^<\n].+?)\n\n", r"\n<p>\1</p>\n", html)

    # Horizontal rules
    html = re.sub(r"^-{3,}$", "<hr>", html, flags=re.MULTILINE)

    # Alerts/notes (simple pattern matching)
    html = re.sub(
        r"> \[!NOTE\]",
        '<div class="alert alert-note"><div class="alert-title">ℹ️ Note</div>',
        html,
    )
    html = re.sub(
        r"> \[!WARNING\]",
        '<div class="alert alert-warning"><div class="alert-title">⚠️ Warning</div>',
        html,
    )

    return html


def generate_html_template(title, content, sidebar_active=None):
    """Generate HTML page with template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="ABI Framework - {title}">
    <title>{title} | ABI Framework</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="canonical" href="https://donaldfilimon.github.io/abi/">
</head>
<body>
    <div class="container">
        <!-- Sidebar Navigation -->
        <nav class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <h1>
                    <span>ABI</span>
                    <span class="version">v0.1.0</span>
                </h1>
            </div>

            <div class="search-container">
                <input
                    type="search"
                    id="search-input"
                    class="search-input"
                    placeholder="Search documentation... (Ctrl+K)"
                    aria-label="Search documentation"
                >
                <div class="search-results" id="search-results"></div>
            </div>

            <div class="sidebar-content">
                <div class="sidebar-section">
                    <h3>Getting Started</h3>
                    <a href="../index.html" class="sidebar-link">Home</a>
                    <a href="quickstart.html" class="sidebar-link {"active" if sidebar_active == "quickstart" else ""}>Quick Start</a>
                    <a href="build-test.html" class="sidebar-link {"active" if sidebar_active == "build-test" else ""}>Build & Test</a>
                    <a href="intro.html" class="sidebar-link {"active" if sidebar_active == "intro" else ""}>Introduction</a>
                </div>

                <div class="sidebar-section">
                    <h3>Developer Guide</h3>
                    <a href="agents.html" class="sidebar-link {"active" if sidebar_active == "agents" else ""}>Agent Guide</a>
                    <a href="../../AGENTS.md" class="sidebar-link">AGENTS.md (Raw)</a>
                    <a href="../../CLAUDE.md" class="sidebar-link">CLAUDE.md (Raw)</a>
                </div>

                <div class="sidebar-section">
                    <h3>Features</h3>
                    <a href="ai.html" class="sidebar-link {"active" if sidebar_active == "ai" else ""}>AI & Agents</a>
                    <a href="compute.html" class="sidebar-link {"active" if sidebar_active == "compute" else ""}>Compute Engine</a>
                    <a href="gpu.html" class="sidebar-link {"active" if sidebar_active == "gpu" else ""}>GPU Acceleration</a>
                    <a href="database.html" class="sidebar-link {"active" if sidebar_active == "database" else ""}>Database</a>
                    <a href="network.html" class="sidebar-link {"active" if sidebar_active == "network" else ""}>Network</a>
                    <a href="monitoring.html" class="sidebar-link {"active" if sidebar_active == "monitoring" else ""}>Monitoring</a>
                </div>

                <div class="sidebar-section">
                    <h3>Reference</h3>
                    <a href="api.html" class="sidebar-link {"active" if sidebar_active == "api" else ""}>API Reference</a>
                    <a href="cli.html" class="sidebar-link {"active" if sidebar_active == "cli" else ""}>CLI Commands</a>
                    <a href="migration.html" class="sidebar-link {"active" if sidebar_active == "migration" else ""}>Zig 0.16 Migration</a>
                </div>

                <div class="sidebar-section">
                    <h3>Support</h3>
                    <a href="troubleshooting.html" class="sidebar-link {"active" if sidebar_active == "troubleshooting" else ""}>Troubleshooting</a>
                    <a href="../../TODO.md" class="sidebar-link">TODO List</a>
                    <a href="../../ROADMAP.md" class="sidebar-link">Roadmap</a>
                </div>

                <div class="sidebar-section">
                    <h3>External</h3>
                    <a href="../../README.md" class="sidebar-link">README</a>
                    <a href="https://github.com/donaldfilimon/abi" class="sidebar-link" target="_blank" rel="noopener">
                        GitHub Repository ↗
                    </a>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="main-content">
            <div class="content-wrapper fade-in">
{content}

                <footer style="margin-top: 3rem; border-top: 1px solid var(--border); padding-top: 2rem;">
                    <div class="text-center">
                        <p class="text-muted">
                            Built with ❤️ using Zig 0.16.x •
                            <a href="https://github.com/donaldfilimon/abi" target="_blank" rel="noopener">
                                GitHub Repository
                            </a> •
                            <a href="https://github.com/donaldfilimon/abi/issues" target="_blank" rel="noopener">
                                Report Issue
                            </a>
                        </p>
                        <p class="text-muted" style="margin-top: 0.5rem; font-size: 0.85rem;">
                            MIT License • © 2025-2026 ABI Framework Contributors
                        </p>
                    </div>
                </footer>
            </div>
        </main>

        <button class="menu-toggle" id="menu-toggle">☰</button>
    </div>

    <script src="../assets/js/search.js"></script>
</body>
</html>"""


# Mapping of markdown files to HTML pages
PAGES = [
    ("../QUICKSTART.md", "quickstart.html", "Quick Start", "quickstart"),
    ("../docs/intro.md", "intro.html", "Introduction", "intro"),
    ("../AGENTS.md", "agents.html", "Agent Guide", "agents"),
    ("../API_REFERENCE.md", "api.html", "API Reference", "api"),
    ("../docs/ai.md", "ai.html", "AI & Agents", "ai"),
    ("../docs/compute.md", "compute.html", "Compute Engine", "compute"),
    ("../docs/gpu.md", "gpu.html", "GPU Acceleration", "gpu"),
    ("../docs/database.md", "database.html", "Database", "database"),
    ("../docs/network.md", "network.html", "Network", "network"),
    ("../docs/monitoring.md", "monitoring.html", "Monitoring", "monitoring"),
    ("../docs/framework.md", "framework.html", "Framework", "framework"),
    (
        "../docs/troubleshooting.md",
        "troubleshooting.html",
        "Troubleshooting",
        "troubleshooting",
    ),
    (
        "../docs/migration/zig-0.16-migration.md",
        "migration.html",
        "Zig 0.16 Migration",
        "migration",
    ),
]


def main():
    """Generate all HTML pages."""
    pages_dir = Path("pages")
    pages_dir.mkdir(exist_ok=True)

    print("🚀 Generating ABI Framework Documentation...")
    print()

    for md_path, html_name, title, active_id in PAGES:
        md_file = Path(md_path)
        if not md_file.exists():
            print(f"⚠️  Skipping {md_path} (not found)")
            continue

        print(f"📄 Processing {title}...")

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse markdown to HTML
        html_content = parse_markdown(content, title=title)

        # Generate full HTML page
        full_html = generate_html_template(
            title, html_content, sidebar_active=active_id
        )

        # Write to file
        output_path = pages_dir / html_name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"   ✅ Generated {html_name}")

    print()
    print("✨ Documentation generation complete!")
    print()
    print("📊 Statistics:")
    print(f"   - {len(PAGES)} pages generated")
    print(f"   - Output directory: {pages_dir.absolute()}")
    print()
    print("🚀 To view locally:")
    print(f"   python -m http.server 8000")
    print(f"   Then open: http://localhost:8000/docs-site/")


if __name__ == "__main__":
    main()
