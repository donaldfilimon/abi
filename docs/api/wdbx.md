# WDBX Utilities API

This document provides comprehensive API documentation for the `wdbx` module.

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

WDBX utilities for database management and operations.

### CLI Commands

- `wdbx stats`: Show database statistics
- `wdbx add <vector>`: Add vector to database
- `wdbx query <vector>`: Find nearest neighbor
- `wdbx knn <vector> <k>`: Find k-nearest neighbors
- `wdbx http [--host <ip>] [--port <port>]`: Start the built-in HTTP server

### Running the HTTP server

```bash
# Start the HTTP server on the default loopback interface and port 8080
wdbx http

# Expose the server on all interfaces on port 9090
wdbx http --host 0.0.0.0 --port 9090
```

The CLI binds directly to the lightweight in-memory WDBX service. The server
responds to REST-style requests such as:

- `GET /health` – service health probe returning host and port metadata.
- `GET /stats` – database/vector statistics.
- `GET /query?vec=1,2,3&k=5` – nearest-neighbour search for a CSV vector.
- `POST /add` – add a vector by sending a JSON body `{ "vector": [...] }`.

