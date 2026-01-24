---
title: Personas API Reference
description: Complete API reference for the Multi-Persona AI Assistant system
category: api
---

# Personas API Reference

Complete API documentation for the Multi-Persona AI Assistant system.

## Overview

The Personas API provides endpoints for interacting with three specialized AI personas:

| Persona | Type | Description |
|---------|------|-------------|
| **Abi** | Router/Moderator | Content moderation, sentiment analysis, routing decisions |
| **Abbey** | Empathetic Polymath | Supportive responses with emotional intelligence |
| **Aviva** | Direct Expert | Concise, factual, technically accurate responses |

## Base URL

```
/api/v1
```

## Authentication

All endpoints require authentication via API key in the `Authorization` header:

```
Authorization: Bearer <api_key>
```

---

## Endpoints

### Chat Endpoints

#### POST /chat

Route a message through automatic persona selection.

**Request Body:**

```json
{
  "message": "string",
  "session_id": "string (optional)",
  "context": {
    "previous_messages": ["string"],
    "user_preferences": {}
  },
  "options": {
    "max_tokens": 2048,
    "temperature": 0.7,
    "stream": false
  }
}
```

**Response:**

```json
{
  "response": "string",
  "persona_used": "abbey | aviva | abi",
  "routing_info": {
    "scores": {
      "abbey": 0.85,
      "aviva": 0.65,
      "abi": 0.30
    },
    "decision_reason": "string",
    "latency_ms": 45
  },
  "metadata": {
    "session_id": "string",
    "request_id": "string",
    "timestamp": "ISO8601"
  }
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request body |
| 401 | Authentication required |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (all personas unhealthy) |

---

#### POST /chat/abbey

Force routing to Abbey (empathetic polymath).

**Request Body:**

```json
{
  "message": "string",
  "session_id": "string (optional)",
  "emotional_context": {
    "detected_emotion": "frustrated | sad | happy | curious | neutral",
    "intensity": 0.0-1.0
  },
  "options": {
    "max_tokens": 2048,
    "temperature": 0.7,
    "empathy_level": 0.8,
    "include_reasoning": true
  }
}
```

**Response:**

```json
{
  "response": "string",
  "emotional_analysis": {
    "detected_emotion": "frustrated",
    "intensity": 0.75,
    "suggested_tone": "supportive",
    "empathy_applied": 0.85
  },
  "reasoning_chain": [
    {
      "step": 1,
      "type": "understanding",
      "content": "string",
      "confidence": 0.9
    }
  ],
  "metadata": {
    "session_id": "string",
    "request_id": "string",
    "timestamp": "ISO8601"
  }
}
```

---

#### POST /chat/aviva

Force routing to Aviva (direct expert).

**Request Body:**

```json
{
  "message": "string",
  "session_id": "string (optional)",
  "query_type": "code_request | factual_query | explanation | debugging (optional)",
  "options": {
    "max_tokens": 2048,
    "temperature": 0.2,
    "include_sources": true,
    "code_language": "zig | python | rust | go | javascript"
  }
}
```

**Response:**

```json
{
  "response": "string",
  "classification": {
    "query_type": "code_request",
    "language": "zig",
    "domain": "systems_programming",
    "confidence": 0.95
  },
  "fact_check": {
    "overall_confidence": 0.92,
    "claims": [
      {
        "claim": "string",
        "type": "definition | numerical | causal",
        "confidence": 0.95,
        "source": "string (optional)"
      }
    ],
    "qualifications": ["string"]
  },
  "code_blocks": [
    {
      "language": "zig",
      "code": "string",
      "explanation": "string (optional)"
    }
  ],
  "metadata": {
    "session_id": "string",
    "request_id": "string",
    "timestamp": "ISO8601"
  }
}
```

---

### Persona Management

#### GET /personas

List all available personas with their current status.

**Response:**

```json
{
  "personas": [
    {
      "type": "abbey",
      "name": "Abbey",
      "description": "Empathetic polymath for supportive, thorough responses",
      "status": "healthy | degraded | unhealthy",
      "health": {
        "score": 0.95,
        "latency_ms": 120,
        "success_rate": 0.98,
        "circuit_breaker": "closed | open | half_open"
      },
      "capabilities": ["emotion_detection", "empathy_injection", "reasoning_chains"],
      "config": {
        "temperature": 0.7,
        "max_tokens": 2048
      }
    }
  ],
  "routing_config": {
    "algorithm": "weighted_round_robin",
    "fallback_enabled": true,
    "health_check_interval_ms": 5000
  }
}
```

---

#### GET /personas/{type}

Get detailed information about a specific persona.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| type | string | Persona type: `abi`, `abbey`, or `aviva` |

**Response:**

```json
{
  "type": "abbey",
  "name": "Abbey",
  "description": "Empathetic polymath for supportive, thorough responses",
  "status": "healthy",
  "health": {
    "score": 0.95,
    "latency_p50_ms": 80,
    "latency_p90_ms": 150,
    "latency_p99_ms": 300,
    "success_rate": 0.98,
    "error_rate": 0.02,
    "circuit_breaker": {
      "state": "closed",
      "failure_count": 2,
      "last_failure": "ISO8601",
      "last_success": "ISO8601"
    }
  },
  "capabilities": {
    "emotion_detection": {
      "enabled": true,
      "emotions": ["frustrated", "sad", "happy", "curious", "angry", "anxious", "excited"]
    },
    "empathy_injection": {
      "enabled": true,
      "templates_count": 15
    },
    "reasoning_chains": {
      "enabled": true,
      "max_steps": 10
    }
  },
  "statistics": {
    "total_requests": 15420,
    "avg_response_time_ms": 95,
    "requests_today": 234,
    "tokens_processed": 1542000
  }
}
```

---

### Metrics & Monitoring

#### GET /personas/metrics

Get aggregated metrics for all personas.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| window | string | 1h | Time window: `5m`, `1h`, `24h`, `7d` |
| format | string | json | Response format: `json`, `prometheus` |

**Response (JSON):**

```json
{
  "timestamp": "ISO8601",
  "window": "1h",
  "aggregate": {
    "total_requests": 5420,
    "success_rate": 0.97,
    "avg_latency_ms": 105,
    "p50_latency_ms": 85,
    "p90_latency_ms": 180,
    "p95_latency_ms": 250,
    "p99_latency_ms": 450
  },
  "by_persona": {
    "abbey": {
      "requests": 2100,
      "success_rate": 0.98,
      "avg_latency_ms": 120,
      "routing_score_avg": 0.82
    },
    "aviva": {
      "requests": 2800,
      "success_rate": 0.96,
      "avg_latency_ms": 90,
      "routing_score_avg": 0.78
    },
    "abi": {
      "requests": 520,
      "success_rate": 0.99,
      "avg_latency_ms": 45,
      "routing_score_avg": 0.35
    }
  },
  "routing": {
    "decisions": 5420,
    "avg_decision_time_ms": 12,
    "fallback_count": 15,
    "circuit_breaker_trips": 2
  },
  "alerts": {
    "active": 0,
    "triggered_in_window": 1
  }
}
```

**Response (Prometheus):**

```prometheus
# HELP persona_requests_total Total requests per persona
# TYPE persona_requests_total counter
persona_requests_total{persona="abbey"} 2100
persona_requests_total{persona="aviva"} 2800
persona_requests_total{persona="abi"} 520

# HELP persona_latency_seconds Request latency in seconds
# TYPE persona_latency_seconds histogram
persona_latency_seconds_bucket{persona="abbey",le="0.1"} 1500
persona_latency_seconds_bucket{persona="abbey",le="0.25"} 1950
persona_latency_seconds_bucket{persona="abbey",le="0.5"} 2050
persona_latency_seconds_bucket{persona="abbey",le="+Inf"} 2100

# HELP persona_health_score Current health score (0-1)
# TYPE persona_health_score gauge
persona_health_score{persona="abbey"} 0.95
persona_health_score{persona="aviva"} 0.92
persona_health_score{persona="abi"} 0.98
```

---

#### GET /personas/health

Get health status for all personas.

**Response:**

```json
{
  "status": "healthy | degraded | unhealthy",
  "timestamp": "ISO8601",
  "personas": {
    "abbey": {
      "status": "healthy",
      "score": 0.95,
      "checks": {
        "latency": "pass",
        "error_rate": "pass",
        "circuit_breaker": "pass",
        "resource_usage": "pass"
      },
      "last_check": "ISO8601"
    },
    "aviva": {
      "status": "healthy",
      "score": 0.92,
      "checks": {
        "latency": "pass",
        "error_rate": "pass",
        "circuit_breaker": "pass",
        "resource_usage": "pass"
      },
      "last_check": "ISO8601"
    },
    "abi": {
      "status": "healthy",
      "score": 0.98,
      "checks": {
        "latency": "pass",
        "error_rate": "pass",
        "circuit_breaker": "pass",
        "resource_usage": "pass"
      },
      "last_check": "ISO8601"
    }
  },
  "system": {
    "memory_usage_percent": 45.2,
    "cpu_usage_percent": 32.1,
    "active_connections": 12,
    "pending_requests": 3
  }
}
```

---

### WebSocket Streaming

#### WS /chat/stream

Stream responses in real-time via WebSocket.

**Connection:**

```javascript
const ws = new WebSocket('wss://api.example.com/api/v1/chat/stream');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your_api_key'
  }));
};
```

**Request Message:**

```json
{
  "type": "chat",
  "message": "string",
  "session_id": "string (optional)",
  "persona": "auto | abbey | aviva (optional)",
  "options": {
    "max_tokens": 2048,
    "temperature": 0.7
  }
}
```

**Response Messages:**

```json
// Routing decision
{
  "type": "routing",
  "persona_selected": "abbey",
  "scores": {
    "abbey": 0.85,
    "aviva": 0.65
  }
}

// Token stream
{
  "type": "token",
  "content": "string",
  "index": 0
}

// Completion
{
  "type": "complete",
  "total_tokens": 150,
  "latency_ms": 1200,
  "metadata": {}
}

// Error
{
  "type": "error",
  "code": "string",
  "message": "string"
}
```

---

## Data Types

### PersonaType

```typescript
type PersonaType = "abi" | "abbey" | "aviva";
```

### HealthStatus

```typescript
type HealthStatus = "healthy" | "degraded" | "unhealthy";
```

### CircuitBreakerState

```typescript
type CircuitBreakerState = "closed" | "open" | "half_open";
```

### EmotionType

```typescript
type EmotionType =
  | "neutral"
  | "happy"
  | "sad"
  | "frustrated"
  | "angry"
  | "anxious"
  | "curious"
  | "excited"
  | "confused"
  | "disappointed";
```

### QueryType

```typescript
type QueryType =
  | "code_request"
  | "factual_query"
  | "explanation"
  | "documentation"
  | "debugging"
  | "general";
```

### ToneStyle

```typescript
type ToneStyle =
  | "neutral"
  | "supportive"
  | "empathetic"
  | "encouraging"
  | "patient"
  | "enthusiastic";
```

### Language

```typescript
type Language =
  | "zig"
  | "python"
  | "rust"
  | "go"
  | "javascript"
  | "typescript"
  | "c"
  | "cpp"
  | "java"
  | "unknown";
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Malformed request body |
| `missing_field` | 400 | Required field missing |
| `invalid_persona` | 400 | Unknown persona type |
| `unauthorized` | 401 | Invalid or missing API key |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limited` | 429 | Too many requests |
| `persona_unavailable` | 503 | Requested persona unhealthy |
| `all_personas_unavailable` | 503 | All personas unhealthy |
| `internal_error` | 500 | Internal server error |

**Error Response Format:**

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Human-readable error message",
    "details": {
      "field": "message",
      "reason": "Field is required"
    },
    "request_id": "string"
  }
}
```

---

## Rate Limits

| Tier | Requests/min | Tokens/min |
|------|--------------|------------|
| Free | 20 | 40,000 |
| Basic | 60 | 150,000 |
| Pro | 300 | 1,000,000 |
| Enterprise | Custom | Custom |

Rate limit headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640000000
```

---

## SDK Examples

### Zig

```zig
const personas = @import("abi").ai.personas;

// Initialize
var orchestrator = try personas.PersonaOrchestrator.init(allocator, .{
    .enable_abbey = true,
    .enable_aviva = true,
});
defer orchestrator.deinit();

// Route automatically
const response = try orchestrator.process(.{
    .content = "How do I implement a hash table in Zig?",
    .session_id = "user-123",
});

std.debug.print("Response from {s}: {s}\n", .{
    @tagName(response.persona_used),
    response.content,
});
```

### Python

```python
from abi import PersonaClient

client = PersonaClient(api_key="your_api_key")

# Auto-routing
response = client.chat("How do I implement a hash table?")
print(f"Response from {response.persona}: {response.content}")

# Force Abbey
response = client.chat(
    "I'm struggling with this concept...",
    persona="abbey",
    options={"empathy_level": 0.8}
)

# Streaming
for chunk in client.chat_stream("Explain recursion"):
    print(chunk.content, end="", flush=True)
```

### JavaScript/TypeScript

```typescript
import { PersonaClient } from '@abi/personas';

const client = new PersonaClient({ apiKey: 'your_api_key' });

// Auto-routing
const response = await client.chat({
  message: 'How do I implement a hash table?'
});
console.log(`Response from ${response.personaUsed}: ${response.content}`);

// WebSocket streaming
const ws = client.stream();
ws.onMessage((msg) => {
  if (msg.type === 'token') {
    process.stdout.write(msg.content);
  }
});
ws.send({ message: 'Explain recursion step by step' });
```

---

## See Also

- [AI Module Guide](../ai.md) - Overview of the AI module
- [Getting Started Tutorial](../tutorials/personas.md) - Step-by-step tutorial
- [Architecture Overview](../architecture/multi-persona-roadmap.md) - System design
