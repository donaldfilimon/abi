# Production Deployment Guide

## Networking and HTTP Clients

- Prefer bounded reads with streaming over readAllAlloc for untrusted endpoints.
- Use keep-alive and connection pooling for servers with many small requests.
- Validate all JSON responses and enforce maximum payload sizes.
- Configure timeouts: the weather client enforces request timeouts and max-bytes.

