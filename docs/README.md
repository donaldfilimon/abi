---
layout: documentation
title: "ABI Documentation"
description: "High-performance vector database with AI capabilities - Complete documentation"
permalink: /
---

# ABI Documentation

Welcome to the comprehensive documentation for ABI, a high-performance vector database with integrated AI capabilities.

## 🚀 Quick Navigation

<div class="quick-nav">
  <div class="nav-card">
    <h3><a href="{{ '/generated/API_REFERENCE/' | relative_url }}">📘 API Reference</a></h3>
    <p>Complete API documentation with examples and detailed function signatures.</p>
  </div>

  <div class="nav-card">
    <h3><a href="{{ '/generated/EXAMPLES/' | relative_url }}">💡 Examples</a></h3>
    <p>Practical examples and tutorials to get you started quickly.</p>
  </div>

  <div class="nav-card">
    <h3><a href="{{ '/generated/MODULE_REFERENCE/' | relative_url }}">📦 Module Reference</a></h3>
    <p>Detailed module documentation and architecture overview.</p>
  </div>

  <div class="nav-card">
    <h3><a href="{{ '/generated/PERFORMANCE_GUIDE/' | relative_url }}">⚡ Performance Guide</a></h3>
    <p>Optimization tips, benchmarks, and performance best practices.</p>
  </div>
</div>

## 📖 What's Inside

### Core Documentation
- **[API Reference]({{ '/generated/API_REFERENCE/' | relative_url }})** - Complete function and type documentation
- **[Module Reference]({{ '/generated/MODULE_REFERENCE/' | relative_url }})** - Module structure and relationships
- **[Examples]({{ '/generated/EXAMPLES/' | relative_url }})** - Practical usage examples and tutorials
- **[Performance Guide]({{ '/generated/PERFORMANCE_GUIDE/' | relative_url }})** - Optimization and benchmarking
- **[Definitions]({{ '/generated/DEFINITIONS_REFERENCE/' | relative_url }})** - Comprehensive glossary and concepts

### Developer Resources
- **[Code Index]({{ '/generated/CODE_API_INDEX/' | relative_url }})** - Auto-generated API index from source
- **[Native Docs]({{ '/zig-docs/' | relative_url }})** - Zig compiler-generated documentation
- **[Windows Zig std Troubleshooting]({{ '/ZIG_STD_WINDOWS_FIX.html' | relative_url }})** - Workarounds when the stdlib docs fail to load on Windows
- **[Search]({{ '/index.html' | relative_url }})** - Interactive documentation browser

## 🔍 Features

- **🚄 High Performance**: Optimized vector operations with SIMD support
- **🧠 AI Integration**: Built-in neural networks and machine learning
- **🗄️ Vector Database**: Efficient storage and similarity search
- **🔌 Plugin System**: Extensible architecture for custom functionality
- **📊 Analytics**: Performance monitoring and optimization tools

## 🛠️ Getting Started

1. **Installation**: Check the [Examples]({{ '/generated/EXAMPLES/' | relative_url }}) for setup instructions
2. **Quick Start**: Follow the [basic usage examples]({{ '/generated/EXAMPLES/' | relative_url }}#quick-start)
3. **API Learning**: Explore the [API Reference]({{ '/generated/API_REFERENCE/' | relative_url }}) for detailed function documentation
4. **Optimization**: Read the [Performance Guide]({{ '/generated/PERFORMANCE_GUIDE/' | relative_url }}) for best practices

## 📚 Documentation Types

This documentation is generated using multiple approaches:

### Manual Documentation
- Curated guides and examples
- Performance analysis and optimization tips
- Comprehensive concept explanations
- Best practices and design patterns

### Auto-Generated Documentation
- Source code scanning for public APIs
- Zig compiler documentation extraction
- Type information and signatures
- Cross-references and relationships

```bash
zig build docs
```

Run the command above to regenerate the complete documentation portal, including the Zig-native
reference and the markdown API catalog, ensuring the published site always reflects the current
source tree.

## 🔗 External Resources

- **[GitHub Repository](https://github.com/donaldfilimon/abi/)** - Source code and issues
- **[Zig Language](https://ziglang.org/)** - Learn about the Zig programming language
- **[Vector Databases]({{ '/generated/DEFINITIONS_REFERENCE/' | relative_url }}#vector-database)** - Learn about vector database concepts

## 📧 Support

- **Issues**: [Report bugs or request features](https://github.com/donaldfilimon/abi/issues)
- **Discussions**: [Join community discussions](https://github.com/donaldfilimon/abi/discussions)
- **Documentation**: [Improve documentation](https://github.com/donaldfilimon/abi/issues/new?title=Documentation%20Improvement)

---

<style>
.quick-nav {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.nav-card {
  border: 1px solid #e1e4e8;
  border-radius: 8px;
  padding: 1.5rem;
  background: #f6f8fa;
}

.nav-card h3 {
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.nav-card h3 a {
  text-decoration: none;
  color: #0366d6;
}

.nav-card p {
  margin-bottom: 0;
  color: #586069;
  font-size: 0.9rem;
}

@media (prefers-color-scheme: dark) {
  .nav-card {
    border-color: #30363d;
    background: #21262d;
  }
  
  .nav-card h3 a {
    color: #58a6ff;
  }
  
  .nav-card p {
    color: #8b949e;
  }
}
</style>
