//! Features Overview

The **features** directory groups optional capabilities that can be toggled via the `build_options` flags in `build.zig`. Each feature lives in its own subdirectory and provides a `mod.zig` for public exposure.

| Feature | Flag | Description |
|---------|------|-------------|
| AI | `enable‑ai` | Machine‑learning utilities, model loading, inference pipelines |
| Database | `enable‑database` | Database drivers, connection pooling, query abstractions |
| GPU | `enable‑gpu` | GPU compute back‑ends, shader compilation, device management |
| Monitoring | `enable‑monitoring` | Metrics collection, health checks, alerts |
| Network | `enable‑network` | Advanced networking stacks, protocols, security layers |
| Web | `enable‑web` | HTTP server framework, routing, templating |

Each feature may depend on core or compute modules. The modular layout lets developers compile only the needed pieces, keeping binary size minimal.

