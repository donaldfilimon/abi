# Modern Patterns Catalog (Before/After)

## Error Handling

**Legacy**
```ts
try {
  const r = risky();
  return r;
} catch(e) { return null; }
```

**Modern**
```ts
const result = risky();
if (result.isErr()) {
  return Err(new DomainError('...', result.error));
}
return Ok(result.value);
```

## Types

**Legacy**
```zig
const status = "pending" | "done"; // string
```

**Modern**
```zig
const Status = enum { pending, done };
```

## Modularity

Extract pure core, push effects to edges.

See main skill for principles.
