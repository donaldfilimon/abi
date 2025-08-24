# Cell Language Specification

Cell is a domain-specific language integrated into the Abi AI framework, designed for high-performance computation with error handling capabilities.

## Language Overview

Cell is a simple, expression-based language with:
- Variable declarations
- Arithmetic operations
- Print statements
- Error scope handling

## Syntax

### Variable Declaration

```cell
let x = 5;
let sum = x + 10;
```

### Print Statement

```cell
print x;
print sum + 2;
```

### Expressions

Cell supports basic arithmetic expressions:

```cell
let a = 10;
let b = 20;
let result = a + b - 5;
```

### Error Scopes

Cell provides structured error handling through error scopes:

```cell
error_scope {
    let x = risky_operation();
    print x;
} handle {
    error1 => {
        print 0;
    }
    error2 => {
        print -1;
    }
}
```

## Grammar

```
program     = statement*
statement   = varDecl | printStmt | errorScope
varDecl     = "let" IDENTIFIER "=" expression ";"
printStmt   = "print" expression ";"
errorScope  = "error_scope" "{" statement* "}" "handle" "{" handler* "}"
handler     = IDENTIFIER "=>" "{" statement* "}"
expression  = term (("+"|"-") term)*
term        = primary
primary     = NUMBER | IDENTIFIER | "(" expression ")"
```

## Token Types

- `identifier` - Variable names
- `number` - Integer literals
- `plus`, `minus`, `star`, `slash` - Arithmetic operators
- `assign` - Assignment operator `=`
- `arrow` - Error handler arrow `=>`
- `semicolon` - Statement terminator
- `l_paren`, `r_paren` - Parentheses
- `l_brace`, `r_brace` - Braces
- `error_scope` - Error scope keyword
- `handle_kw` - Handle keyword

## Example Programs

### Basic Arithmetic

```cell
let a = 10;
let b = 20;
let sum = a + b;
print sum;
```

### With Error Handling

```cell
error_scope {
    let value = get_value();
    let result = value + 100;
    print result;
} handle {
    null_error => {
        print 0;
    }
    overflow_error => {
        print 999;
    }
}
```

## Implementation

The Cell language is implemented in Zig with:

- **Lexer** (`lexer.zig`): Tokenizes input text
- **Parser** (`parser.zig`): Builds AST from tokens
- **AST** (`ast.zig`): Abstract syntax tree representation
- **Interpreter** (`interpreter.zig`): Executes the AST
- **Token** (`token.zig`): Token type definitions

## Usage

### As a Library

```zig
const cell = @import("cell");

pub fn main() !void {
    const source = "let x = 1 + 2; print x;";
    var parser = cell.Parser.init(allocator, source);
    const program = try parser.parseProgram();
    
    var interpreter = cell.Interpreter.init(allocator);
    interpreter.evalProgram(program);
}
```

### As a REPL

```bash
abi cell
> let x = 10;
> let y = 20;
> print x + y;
30
```

## Future Extensions

Planned features include:

1. **Functions**: User-defined functions with parameters
2. **Conditionals**: If/else statements
3. **Loops**: While and for loops
4. **Types**: Static type system
5. **Modules**: Import/export system
6. **Async**: Asynchronous operations
7. **FFI**: Foreign function interface to Zig

## Performance

Cell is designed for performance:

- Zero-allocation parsing where possible
- Arena allocation for AST nodes
- Direct execution without intermediate bytecode
- Integration with Abi's SIMD and GPU capabilities
