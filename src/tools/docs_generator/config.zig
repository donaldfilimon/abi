const std = @import("std");

fn writeFile(path: []const u8, contents: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    if (contents.len > 0) try file.writeAll(contents);
}

pub fn generateNoJekyll(_: std.mem.Allocator) !void {
    try writeFile("docs/.nojekyll", "");
}

pub fn generateJekyllConfig(_: std.mem.Allocator) !void {
    const config =
        \\title: "ABI Documentation"\n
        \\description: "Developer documentation for the ABI project"\n
        \\theme: minima\n
        \\baseurl: "/abi"\n
        \\plugins:\n
        \\  - jekyll-feed\n
        \\  - jekyll-seo-tag\n
        \\navigation:\n
        \\  - title: "API Reference"\n
        \\    url: /generated/API_REFERENCE.md\n
        \\  - title: "Module Reference"\n
        \\    url: /generated/MODULE_REFERENCE.md\n
        \\  - title: "Examples"\n
        \\    url: /generated/EXAMPLES.md\n
        \\  - title: "Performance Guide"\n
        \\    url: /generated/PERFORMANCE_GUIDE.md\n
    ;
    try writeFile("docs/_config.yml", config);
}

pub fn generateGitHubPagesLayout(_: std.mem.Allocator) !void {
    const html =
        \\---\n
        \\layout: default\n
        \\---\n
        \\<!DOCTYPE html>\n
        \\<html lang="en">\n
        \\  <head>\n
        \\    <meta charset="utf-8">\n
        \\    <meta name="viewport" content="width=device-width, initial-scale=1">\n
        \\    <link rel="stylesheet" href="{{ '/assets/css/documentation.css' | relative_url }}">\n
        \\  </head>\n
        \\  <body>\n
        \\    <main class="documentation">\n
        \\      <nav>\n
        \\        <ul>\n
        \\          {% for nav in site.navigation %}\n
        \\          <li><a href="{{ nav.url | relative_url }}">{{ nav.title }}</a></li>\n
        \\          {% endfor %}\n
        \\        </ul>\n
        \\      </nav>\n
        \\      <article>\n
        \\        {{ content }}\n
        \\      </article>\n
        \\    </main>\n
        \\    <script src="{{ '/assets/js/search.js' | relative_url }}"></script>\n
        \\  </body>\n
        \\</html>\n
    ;
    try writeFile("docs/_layouts/documentation.html", html);
}

pub fn generateNavigationData(_: std.mem.Allocator) !void {
    const data =
        \\items:\n
        \\  - title: API Reference\n
        \\    href: generated/API_REFERENCE.md\n
        \\  - title: Module Reference\n
        \\    href: generated/MODULE_REFERENCE.md\n
        \\  - title: Examples\n
        \\    href: generated/EXAMPLES.md\n
        \\  - title: Performance Guide\n
        \\    href: generated/PERFORMANCE_GUIDE.md\n
        \\  - title: Definitions\n
        \\    href: generated/DEFINITIONS_REFERENCE.md\n
    ;
    try writeFile("docs/_data/navigation.yml", data);
}

pub fn generateSEOMetadata(_: std.mem.Allocator) !void {
    const seo =
        \\title: ABI Documentation\n
        \\description: Reference material for the ABI project\n
        \\keywords:\n
        \\  - zig\n
        \\  - abi\n
        \\  - documentation\n
        \\twitter:\n
        \\  creator: '@abi_project'\n
        \\open_graph:\n
        \\  type: website\n
    ;
    try writeFile("docs/_data/seo.yml", seo);
}

pub fn generateGitHubPagesAssets(_: std.mem.Allocator) !void {
    const css =
        \\body { font-family: system-ui, sans-serif; margin: 0; padding: 0; }\n
        \\main.documentation { display: flex; gap: 2rem; padding: 2rem; }\n
        \\main.documentation nav { width: 16rem; }\n
        \\main.documentation article { flex: 1; min-width: 0; }\n
        \\pre { background: #1f2430; color: #f2f5fa; padding: 1rem; overflow-x: auto; }\n
    ;
    try writeFile("docs/assets/css/documentation.css", css);

    const js =
        \\(function() {\n
        \\  const input = document.querySelector('[data-search-input]') || document.getElementById('search-input');\n
        \\  if (!input) return;\n
        \\  fetch('{{ '/generated/search_index.json' | relative_url }}').then(function(res) { return res.json(); }).then(function(index) {\n
        \\    input.addEventListener('input', function() {\n
        \\      const query = this.value.trim().toLowerCase();\n
        \\      const results = index.filter(function(entry) {\n
        \\        return entry.title.toLowerCase().includes(query);\n
        \\      }).slice(0, 5);\n
        \\      const list = document.getElementById('search-results');\n
        \\      if (!list) return;\n
        \\      list.innerHTML = results.map(function(entry) {\n
        \\        return '<li><a href="' + entry.file + '">' + entry.title + '</a></li>';\n
        \\      }).join('');\n
        \\    });\n
        \\  }).catch(function(err) {\n
        \\    console.warn('Failed to load search index', err);\n
        \\  });\n
        \\})();\n
    ;
    try writeFile("docs/assets/js/search.js", js);
}

pub fn generateGitHubActionsWorkflow(_: std.mem.Allocator) !void {
    const workflow =
        \\name: Build documentation\n
        \\on:\n
        \\  push:\n
        \\    branches: [ main ]\n
        \\  pull_request:\n
        \\jobs:\n
        \\  docs:\n
        \\    runs-on: ubuntu-latest\n
        \\    steps:\n
        \\      - uses: actions/checkout@v4\n
        \\      - uses: goto-bus-stop/setup-zig@v2\n
        \\        with:\n
        \\          version: 0.11.0\n
        \\      - run: zig build docgen\n
        \\      - uses: actions/upload-pages-artifact@v3\n
        \\        with:\n
        \\          path: docs\n
    ;
    try writeFile(".github/workflows/docs.yml", workflow);
}
