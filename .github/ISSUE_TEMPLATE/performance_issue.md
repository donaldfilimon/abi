---
name: Performance issue
about: Report performance problems or regressions
title: '[PERF] '
labels: performance
assignees: ''
---

**Performance Issue Description**
A clear description of the performance problem you're experiencing.

**Expected Performance**
What performance characteristics did you expect?

**Actual Performance**
What performance are you actually seeing?

**Environment**
 - OS: [e.g. Windows 10, Ubuntu 22.04, macOS 13.0]
 - Zig version: [e.g. 0.15.1]
 - ABI version: [e.g. 0.1.0]
 - Architecture: [e.g. x86_64, aarch64]
 - Hardware: [e.g. CPU model, RAM amount]

**Benchmark Results**
If you have benchmark results, please include them:

```bash
# Run benchmarks and paste results
zig build bench-all
```

**Performance Profiling**
If you've run profiling, include the results:

```bash
# Run profiler and paste relevant output
zig build profile
```

**Code Example**
```zig
// Minimal code that demonstrates the performance issue
```

**Additional Context**
- Is this a regression from a previous version?
- Does this affect all platforms or specific ones?
- Any workarounds you've found?

**Performance Metrics**
- [ ] CPU usage
- [ ] Memory usage
- [ ] Execution time
- [ ] Throughput
- [ ] Latency
- [ ] Other: ___________
