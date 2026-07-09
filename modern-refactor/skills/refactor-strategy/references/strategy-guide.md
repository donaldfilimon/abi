# Refactor Strategy Guide

## Strategy Selection Matrix

| Situation                    | Recommended Strategy     | Rationale |
|-----------------------------|--------------------------|---------|
| Small isolated module, good tests | Direct clean rewrite    | Low risk, high payoff |
| Core domain with many dependents | Phased (Strangler Fig)  | Preserve behavior for consumers |
| Legacy with poor tests      | Analysis first + parallel implementation | Reduce risk |
| Performance critical path   | Measure → target design → incremental migration | Protect invariants |

## Clean Slate Questions

For any module ask:
1. What is the single responsibility in modern terms?
2. What are the ideal error and success paths?
3. Which current types/structures are historical accidents?
4. How would I test this if writing it fresh?
5. What modern language features remove the need for previous workarounds?

## Example Phased Plan Outline

1. **Discovery** (1-2 days)
   - Extract contracts
   - Identify modernization targets

2. **Parallel Modern Impl** (core module)
   - Write new version beside old
   - 100% parity test suite

3. **Strangler Introduction**
   - Route 5% traffic / calls
   - Monitor + expand

4. **Cutover + Deletion**
   - Remove old after confidence gates

See the main SKILL.md for the high-level process.
