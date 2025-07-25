# Cell Language Specification (Draft)

This document outlines a minimal prototype of the **Cell** programming language.
It incorporates experimental features discussed previously such as scoped error
handling with `error_scope` and `raise`.

## Syntax Overview

```
let x = 1 + 2;
print x;

error_scope {
    let y = x + 3;
    print y;
} handle {
    SomeError => {
        print 0;
    }
}
```

The prototype supports:

- Variable declarations using `let`.
- Integer arithmetic with `+` and `-`.
- `print` statements.
- `error_scope { ... } handle { ... }` blocks. Handlers are parsed but not yet
  executed at runtime.

## Future Directions

- Implement `raise` semantics for propagating errors.
- Add coroutines and typed channels for concurrency.
- Expand the standard library with data structures and algorithms.

This specification is intentionally small to enable rapid iteration and
experimentation. The implementation can be found in `src/cell`.
