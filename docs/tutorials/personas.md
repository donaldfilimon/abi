---
title: Getting Started with Personas
description: Step-by-step guide to using the Multi-Persona AI Assistant
category: tutorials
---

# Getting Started with Personas

This tutorial walks you through setting up and using the Multi-Persona AI Assistant system. You'll learn how to:

1. Initialize the persona system
2. Route messages automatically
3. Use specific personas directly
4. Monitor health and metrics
5. Handle errors gracefully

## Prerequisites

- Zig 0.16 or later
- ABI framework built with AI enabled: `zig build -Denable-ai=true`

## Quick Start

### 1. Basic Setup

```zig
const std = @import("std");
const abi = @import("abi");
const personas = abi.ai.personas;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the persona orchestrator
    var orchestrator = try personas.PersonaOrchestrator.init(allocator, .{
        .enable_abbey = true,
        .enable_aviva = true,
        .routing_strategy = .adaptive,
    });
    defer orchestrator.deinit();

    // Process a message with automatic routing
    const response = try orchestrator.process(.{
        .content = "How do I implement a linked list in Zig?",
    });

    std.debug.print("Persona: {s}\n", .{@tagName(response.persona_used)});
    std.debug.print("Response: {s}\n", .{response.content});
}
```

### 2. Understanding Persona Selection

The system automatically routes messages based on content analysis:

| Content Type | Selected Persona | Why |
|--------------|------------------|-----|
| Technical questions | Aviva | Direct, accurate answers |
| Emotional messages | Abbey | Empathetic support |
| Policy violations | Abi | Content moderation |
| Ambiguous queries | Abbey/Aviva | Based on scoring |

```zig
// Technical question -> Routes to Aviva
const tech = try orchestrator.process(.{
    .content = "What's the time complexity of binary search?",
});
// tech.persona_used == .aviva

// Emotional message -> Routes to Abbey
const emotional = try orchestrator.process(.{
    .content = "I'm really frustrated with this bug...",
});
// emotional.persona_used == .abbey
```

## Working with Abbey (Empathetic Polymath)

Abbey excels at supportive, thorough responses with emotional intelligence.

### When to Use Abbey

- User expresses frustration or confusion
- Complex topics requiring step-by-step explanation
- Situations needing empathy and encouragement

### Direct Abbey Requests

```zig
const abbey = orchestrator.getPersona(.abbey);

// Process with emotional context
const response = try abbey.process(.{
    .content = "I've been stuck on this for hours and nothing works!",
    .emotional_context = .{
        .detected_emotion = .frustrated,
        .intensity = 0.8,
    },
});

// Response includes empathetic acknowledgment
// "I understand how frustrating that can be when you've put in so much effort..."
```

### Emotion Detection

Abbey automatically detects emotions in user messages:

```zig
const emotion_processor = abbey.getEmotionProcessor();
const result = try emotion_processor.process(
    "This is so confusing, I don't understand anything!",
    .{}, // empty prior context
);

std.debug.print("Detected emotion: {s}\n", .{@tagName(result.detected_emotion)});
std.debug.print("Intensity: {d:.2}\n", .{result.intensity});
std.debug.print("Suggested tone: {s}\n", .{@tagName(result.suggested_tone)});
```

### Reasoning Chains

For complex questions, Abbey generates step-by-step reasoning:

```zig
const reasoning_engine = abbey.getReasoningEngine();
const chain = try reasoning_engine.reason(
    "How do I debug a memory leak in my Zig program?",
    .{}, // memory context
    null, // no emotional context
);
defer chain.deinit();

for (chain.steps.items, 0..) |step, i| {
    std.debug.print("Step {d}: {s}\n", .{ i + 1, step.content });
    std.debug.print("  Confidence: {d:.2}\n", .{step.confidence});
}
```

## Working with Aviva (Direct Expert)

Aviva provides concise, factual, technically accurate responses.

### When to Use Aviva

- Code generation requests
- Factual questions requiring accuracy
- Documentation lookups
- Debugging assistance

### Direct Aviva Requests

```zig
const aviva = orchestrator.getPersona(.aviva);

const response = try aviva.process(.{
    .content = "Write a function to check if a string is a palindrome",
    .options = .{
        .include_code = true,
        .language = .zig,
    },
});
```

### Query Classification

Aviva classifies queries to optimize response format:

```zig
const classifier = aviva.getClassifier();
const classification = classifier.classify(
    "Implement a binary search tree in Zig",
);

std.debug.print("Query type: {s}\n", .{@tagName(classification.query_type)});
std.debug.print("Language: {s}\n", .{@tagName(classification.language)});
std.debug.print("Confidence: {d:.2}\n", .{classification.confidence});
```

### Code Generation

```zig
const code_generator = aviva.getCodeGenerator();

const params = [_][]const u8{ "haystack: []const u8", "needle: []const u8" };
const template = try code_generator.generateFunctionTemplate(
    "contains",
    &params,
    "bool",
    .zig,
    null,
);

std.debug.print("Generated code:\n{s}\n", .{template.code});
```

### Fact Checking

Aviva verifies claims in responses:

```zig
const fact_checker = aviva.getFactChecker();
var result = try fact_checker.check(
    "Zig was created by Andrew Kelley and first released in 2016.",
);
defer result.deinit();

std.debug.print("Overall confidence: {d:.2}\n", .{result.overall_confidence});
for (result.claims.items) |claim| {
    std.debug.print("  Claim: {s}\n", .{claim.text});
    std.debug.print("  Confidence: {d:.2}\n", .{claim.confidence});
}
```

## Routing and Load Balancing

### Understanding Routing Scores

The router evaluates each message and assigns scores:

```zig
const router = orchestrator.getRouter();
const scores = try router.evaluate(.{
    .content = "I'm confused about how slices work in Zig",
});

std.debug.print("Routing scores:\n", .{});
std.debug.print("  Abbey: {d:.2}\n", .{scores.getScore(.abbey) orelse 0});
std.debug.print("  Aviva: {d:.2}\n", .{scores.getScore(.aviva) orelse 0});
```

### Custom Routing Rules

Add custom rules to influence routing:

```zig
var engine = orchestrator.getRulesEngine();

// Boost Abbey for learning-related queries
try engine.addRule(.{
    .name = "learning_support",
    .condition = .{ .contains_keywords = &.{ "learn", "understand", "confused" } },
    .persona_boost = .{ .abbey = 0.2 },
    .priority = 5,
});
```

### Health-Weighted Routing

The load balancer considers persona health:

```zig
const lb = orchestrator.getLoadBalancer();

// Get current health status
const health = lb.getHealthStatus();
for (health) |h| {
    std.debug.print("{s}: {s} (score: {d:.2})\n", .{
        @tagName(h.persona_type),
        @tagName(h.status),
        h.score,
    });
}
```

## Monitoring and Metrics

### Accessing Metrics

```zig
const metrics = orchestrator.getMetrics();

// Get latency percentiles
const percentiles = metrics.getPercentiles(.abbey);
std.debug.print("Abbey latency - p50: {d}ms, p99: {d}ms\n", .{
    percentiles.p50 / 1_000_000,
    percentiles.p99 / 1_000_000,
});

// Get success rates
const success_rate = metrics.getSuccessRate(.aviva);
std.debug.print("Aviva success rate: {d:.2}%\n", .{success_rate * 100});
```

### Setting Up Alerts

```zig
const alerts = orchestrator.getAlerts();

// Add custom alert
try alerts.addRule(.{
    .name = "high_latency_warning",
    .condition = .{ .latency_exceeds_ms = 2000 },
    .severity = .warning,
    .personas = &.{ .abbey, .aviva },
});

// Check for active alerts
const active = alerts.getActive();
for (active) |alert| {
    std.debug.print("Alert: {s} ({s})\n", .{
        alert.name,
        @tagName(alert.severity),
    });
}
```

### Health Checks

```zig
const health_checker = orchestrator.getHealthChecker();

// Check all personas
const results = try health_checker.checkAll();
defer allocator.free(results);

for (results) |result| {
    std.debug.print("{s}: {s}\n", .{
        @tagName(result.persona_type),
        @tagName(result.status),
    });
    std.debug.print("  Latency check: {s}\n", .{
        if (result.checks.latency) "pass" else "fail",
    });
    std.debug.print("  Error rate: {s}\n", .{
        if (result.checks.error_rate) "pass" else "fail",
    });
}
```

## Error Handling

### Graceful Degradation

The system handles failures with fallback:

```zig
const response = orchestrator.process(.{
    .content = "Help me with this code",
}) catch |err| switch (err) {
    error.PersonaUnavailable => {
        // Primary persona unhealthy, try fallback
        return orchestrator.processWithFallback(.{
            .content = "Help me with this code",
        });
    },
    error.AllPersonasUnavailable => {
        // All personas down, return cached or error
        return error.ServiceUnavailable;
    },
    else => return err,
};
```

### Circuit Breaker States

```zig
const cb_state = orchestrator.getCircuitBreakerState(.abbey);
switch (cb_state) {
    .closed => std.debug.print("Abbey healthy\n", .{}),
    .open => std.debug.print("Abbey circuit open - not accepting requests\n", .{}),
    .half_open => std.debug.print("Abbey recovering - testing with limited requests\n", .{}),
}
```

## HTTP API Usage

### Starting the API Server

```zig
const web = abi.web;

var server = try web.Server.init(allocator, .{
    .port = 8080,
});
defer server.deinit();

// Register persona routes
const router = web.routes.personas.Router.init(orchestrator);
try server.registerRoutes(router.getRoutes());

std.debug.print("Server listening on http://localhost:8080\n", .{});
try server.run();
```

### API Examples with curl

```bash
# Auto-routing chat
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"message": "How do I use slices in Zig?"}'

# Force Abbey
curl -X POST http://localhost:8080/api/v1/chat/abbey \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"message": "Im struggling with this concept..."}'

# Get metrics
curl http://localhost:8080/api/v1/personas/metrics \
  -H "Authorization: Bearer $API_KEY"

# Health check
curl http://localhost:8080/api/v1/personas/health \
  -H "Authorization: Bearer $API_KEY"
```

## Best Practices

### 1. Let the Router Decide

In most cases, let automatic routing handle persona selection:

```zig
// Good - let router decide
const response = try orchestrator.process(.{ .content = user_message });

// Only force persona when you have specific requirements
const abbey_response = try orchestrator.processWithPersona(.abbey, .{
    .content = user_message,
});
```

### 2. Provide Context

Include session context for better routing:

```zig
const response = try orchestrator.process(.{
    .content = "Can you explain that again?",
    .session_id = session_id,
    .context = .{
        .previous_messages = &previous_messages,
    },
});
```

### 3. Handle Metrics

Regularly check metrics for performance issues:

```zig
// In your monitoring loop
const aggregate = orchestrator.getMetrics().getAggregate();
if (aggregate.p99_latency_ms > 2000) {
    log.warn("High p99 latency: {}ms", .{aggregate.p99_latency_ms});
}
if (aggregate.success_rate < 0.95) {
    log.warn("Low success rate: {d:.2}%", .{aggregate.success_rate * 100});
}
```

### 4. Configure Appropriate Timeouts

```zig
var orchestrator = try personas.PersonaOrchestrator.init(allocator, .{
    .request_timeout_ms = 30_000,
    .health_check_interval_ms = 5_000,
    .circuit_breaker = .{
        .failure_threshold = 5,
        .reset_timeout_ms = 60_000,
    },
});
```

## Next Steps

- [API Reference](../api/personas.md) - Complete API documentation
- [Architecture Overview](../architecture/multi-persona-roadmap.md) - System design details
- [AI Module Guide](../ai.md) - Broader AI capabilities

## Troubleshooting

### Common Issues

**"PersonaUnavailable" error**
- Check health status: `orchestrator.getHealthChecker().checkAll()`
- Verify circuit breaker state
- Check logs for underlying errors

**High latency responses**
- Monitor p99 latency in metrics
- Consider enabling request caching
- Check LLM backend performance

**Inconsistent routing decisions**
- Review custom routing rules
- Check sentiment analysis accuracy
- Verify emotional context is being passed

**Memory issues**
- Ensure proper `deinit()` calls
- Use `GeneralPurposeAllocator` with leak detection
- Monitor memory metrics in health checks
