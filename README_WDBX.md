# Abi AI Framework - WDBX Vector Database

High-performance AI framework with advanced vector database capabilities. Features HNSW indexing, SIMD acceleration, and comprehensive APIs for modern AI applications.

## 🚀 Features

- **HNSW Indexing**: Sub-12ms search on 10k vectors (81.96 ops/sec)
- **SIMD Acceleration**: 2,777+ ops/sec sustained throughput
- **Multiple APIs**: CLI, HTTP REST, TCP binary, WebSocket
- **Security**: JWT auth, rate limiting, secure credentials
- **Monitoring**: Performance profiling and metrics collection
- **Cross-Platform**: Windows, Linux, macOS with optimizations

## 🏗️ Architecture

- **Core Framework**: Foundation utilities and cross-platform abstractions
- **Vector Database**: HNSW indexing and high-performance storage
- **Networking**: HTTP/TCP/WebSocket servers and clients
- **AI/ML**: Neural networks and machine learning algorithms
- **Performance**: Comprehensive profiling and monitoring
- **GPU Computing**: SIMD and GPU-accelerated operations

## 📦 Installation

**Prerequisites**: Zig 0.16.0-dev, Git

```bash
git clone <repository-url>
cd abi
zig build
# Executables: zig-out/bin/abi, abi-http, abi-demo
```

## 🛠️ Usage

### CLI Commands

```bash
zig build run                    # Start CLI
./zig-out/bin/abi knn "1,2,3,4,5,6,7,8" 5    # K-nearest neighbors
./zig-out/bin/abi query "1,2,3,4,5,6,7,8"   # Nearest neighbor
./zig-out/bin/abi add "1,2,3,4,5,6,7,8"     # Add vector
./zig-out/bin/abi stats                       # Database stats
zig build run-server                         # Start HTTP server
```

### HTTP REST API

```bash
zig build run-server              # Start server (port 8080)
curl http://localhost:8080/health # Health check
curl http://localhost:8080/stats  # Database stats
curl "http://localhost:8080/query?vec=1,2,3,4,5,6,7,8"  # Query
```

## 🔧 Configuration

```bash
# Build options
zig build -Doptimize=ReleaseFast    # Optimized build
zig build -Dsimd=true              # Enable SIMD

# Runtime options
./zig-out/bin/abi --db vectors.wdbx stats    # Custom database
zig build run-server -- --port 9000         # Custom port
./zig-out/bin/abi --format json stats       # JSON output
```

## 📊 Database Format

**WDBX Binary Format:**
- Header (4KB): Metadata, version, dimensions
- Records: Packed float32 vectors
- I/O: Page-aligned, memory-mapped

## 🔒 Security

- JWT authentication and rate limiting
- CORS support and input validation
- Production-ready security features

## 🧪 Testing

```bash
zig build test              # All tests
zig build benchmark         # Performance tests
zig build smoke-http        # HTTP smoke tests
```

## 🚀 Production Deployment

**Requirements:**
- 2GB+ RAM, SSD storage
- Low-latency network

**Security:**
- Change default JWT secret
- Use HTTPS in production
- Configure firewall rules

**Performance:**
- Adjust rate limiting as needed
- Monitor memory usage
- Tune vector dimensions

## 🔮 Future

- GPU acceleration (CUDA/OpenCL)
- Distributed storage
- Advanced indexing algorithms
- Real-time streaming

## 🤝 Contributing

1. Fork and create feature branch
2. Add tests for new functionality
3. Follow Zig code style
4. Submit PR with description

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 📚 Resources

- [Zig Language](https://ziglang.org/)
- [Framework Docs](docs/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
