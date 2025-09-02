# WDBX Vector Database

A high-performance, feature-rich vector database system designed for AI applications, offering efficient storage, indexing, and retrieval of high-dimensional vector embeddings. Implemented in Zig, a modern systems programming language known for its performance and safety features.

## 🚀 Features

- **High Performance**: SIMD-optimized vector operations and efficient file I/O
- **Vector Operations**: Add, query, and k-nearest neighbor search
- **Multiple APIs**: Command-line interface, HTTP REST API, TCP binary protocol, WebSocket
- **Security**: JWT authentication and rate limiting
- **Monitoring**: Comprehensive statistics and performance metrics
- **Production Ready**: Error handling, graceful degradation, and comprehensive testing

## 🏗️ Architecture

The WDBX system consists of several key components:

- **Core Database** (`src/database.zig`): Binary file format with efficient vector storage
- **CLI Interface** (`src/wdbx_cli.zig`): Command-line tool for database operations
- **HTTP Server** (`src/wdbx_http_server.zig`): REST API server with authentication
- **Build System** (`build_wdbx.zig`): Zig build configuration

## 📦 Installation

### Prerequisites

- [Zig](https://ziglang.org/) 0.11.0 or later
- Git

### Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd abi

# Build the WDBX executable
zig build -f build_wdbx.zig

# The executable will be available at zig-out/bin/wdbx
```

## 🛠️ Usage

### Command Line Interface

The WDBX CLI provides comprehensive database management:

```bash
# Query k-nearest neighbors
./zig-out/bin/wdbx knn "1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1" 5

# Query nearest neighbor
./zig-out/bin/wdbx query "1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1"

# Add vector to database
./zig-out/bin/wdbx add "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"

# Show database statistics
./zig-out/bin/wdbx stats

# Start HTTP REST API server
./zig-out/bin/wdbx http 8080

# Generate JWT authentication token
./zig-out/bin/wdbx gen_token admin
```

### HTTP REST API

The HTTP server provides a RESTful interface for vector operations:

#### Start Server

```bash
./zig-out/bin/wdbx http 8080
```

#### API Endpoints

**Health Check**

```bash
curl http://localhost:8080/health
```

**Database Statistics**

```bash
curl http://localhost:8080/stats
```

**Add Vector** (requires admin token)

```bash
# Get JWT token first
TOKEN=$(./zig-out/bin/wdbx gen_token admin | grep "JWT:" | cut -d' ' -f2)

# Add vector
curl -X POST "http://localhost:8080/add" \
     -H "Authorization: Bearer $TOKEN" \
     -d "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"
```

**Query Nearest Neighbor**

```bash
curl "http://localhost:8080/query?vec=1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1"
```

**Query K-Nearest Neighbors**

```bash
curl "http://localhost:8080/knn?vec=1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1&k=5"
```

**Performance Monitoring**

```bash
curl http://localhost:8080/monitor
```

## 🔧 Configuration

The system can be configured via command-line options:

```bash
# Use custom database file
./zig-out/bin/wdbx --db my_vectors.wdbx stats

# Custom server host and port
./zig-out/bin/wdbx --host 0.0.0.0 --port 9000 http

# Output format
./zig-out/bin/wdbx --format json stats

# Verbose output
./zig-out/bin/wdbx --verbose knn "1.0,2.0,3.0" 3
```

## 📊 Database Format

WDBX uses a custom binary format optimized for vector operations:

- **Header (4KB)**: Metadata including magic bytes, version, row count, and dimensionality
- **Records Section**: Densely packed float32 vectors
- **Efficient I/O**: Page-aligned operations and memory mapping support

## 🔒 Security Features

- **JWT Authentication**: Industry-standard token-based authentication
- **Rate Limiting**: Protection against DoS attacks
- **CORS Support**: Configurable cross-origin resource sharing
- **Input Validation**: Comprehensive vector format validation

## 📈 Performance Features

- **SIMD Optimization**: Hand-optimized distance calculations
- **Efficient Storage**: Binary format with minimal overhead
- **Memory Mapping**: Zero-copy file I/O for large datasets
- **Batch Operations**: Support for bulk vector operations

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
zig build -f build_wdbx.zig test

# Run specific test categories
zig build -f build_wdbx.zig test-db      # Database tests
zig build -f build_wdbx.zig test-http    # HTTP server tests
```

## 🚀 Production Deployment

### System Requirements

- **Memory**: 2GB+ RAM recommended
- **Storage**: SSD recommended for high-performance I/O
- **Network**: Low-latency network for distributed deployments

### Security Considerations

- Change default JWT secret in production
- Configure proper firewall rules
- Use HTTPS in production environments
- Implement proper logging and monitoring

### Performance Tuning

- Adjust rate limiting based on system capacity
- Configure appropriate cache sizes
- Monitor memory usage and adjust accordingly
- Use appropriate vector dimensions for your use case

## 🔮 Future Enhancements

- **GPU Acceleration**: CUDA/OpenCL support for distance calculations
- **Distributed Storage**: Multi-node cluster support
- **Advanced Indexing**: HNSW, IVF, and other ANN algorithms
- **Real-time Updates**: Streaming vector ingestion
- **Machine Learning**: Automatic hyperparameter optimization

## 🤝 Contributing

Contributions are welcome! Please see the main project's contributing guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the same terms as the main Abi AI Framework.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Include system information and error messages

## 📚 Additional Resources

- [Zig Programming Language](https://ziglang.org/)
- [Vector Similarity Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
- [JWT Authentication](https://jwt.io/)
- [REST API Design](https://restfulapi.net/)

---

**WDBX Vector Database** - High-performance vector similarity search for AI applications.
