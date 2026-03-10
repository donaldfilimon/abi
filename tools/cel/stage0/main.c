#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} StrBuf;

typedef enum {
    TOK_EOF = 0,
    TOK_IDENT,
    TOK_STRING,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_SEMI,
    TOK_EQ,
    TOK_DOT,
    TOK_COMMA,
    TOK_KW_MODULE,
    TOK_KW_IMPORT,
    TOK_KW_FN,
    TOK_KW_TEST,
    TOK_KW_DEFER,
    TOK_KW_LET,
    TOK_KW_VAR,
    TOK_KW_RETURN,
    TOK_KW_PRINT,
} TokenKind;

typedef struct {
    TokenKind kind;
    char *text;
    int line;
    int column;
} Token;

typedef struct {
    const char *src;
    size_t len;
    size_t pos;
    int line;
    int column;
} Lexer;

typedef struct {
    bool is_ident;
    char *value;
} Expr;

typedef enum {
    STMT_PRINT = 0,
    STMT_DEFER_PRINT,
    STMT_BIND,
    STMT_RETURN,
} StmtKind;

typedef struct {
    StmtKind kind;
    bool is_mutable;
    char *name;
    Expr expr;
} Stmt;

typedef struct {
    char *name;
    bool is_test;
    Stmt *items;
    size_t len;
    size_t cap;
} Decl;

typedef struct {
    char *module_name;
    char **imports;
    size_t import_len;
    size_t import_cap;
    Decl *decls;
    size_t decl_len;
    size_t decl_cap;
} Program;

typedef struct {
    Lexer lexer;
    Token current;
    const char *path;
} Parser;

typedef struct {
    const char *path;
    const char *message;
    int line;
    int column;
} Error;

static void die(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

static void report_error(const Error *err) {
    fprintf(stderr, "%s:%d:%d: %s\n", err->path, err->line, err->column, err->message);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) die("out of memory");
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) die("out of memory");
    return next;
}

static char *xstrdup(const char *src) {
    size_t len = strlen(src);
    char *out = xmalloc(len + 1);
    memcpy(out, src, len + 1);
    return out;
}

static char *xstrndup(const char *src, size_t len) {
    char *out = xmalloc(len + 1);
    memcpy(out, src, len);
    out[len] = '\0';
    return out;
}

static void sb_init(StrBuf *sb) {
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
}

static void sb_reserve(StrBuf *sb, size_t extra) {
    size_t need = sb->len + extra + 1;
    if (need <= sb->cap) return;
    size_t next = sb->cap ? sb->cap * 2 : 256;
    while (next < need) next *= 2;
    sb->data = xrealloc(sb->data, next);
    sb->cap = next;
}

static void sb_append_n(StrBuf *sb, const char *src, size_t len) {
    sb_reserve(sb, len);
    memcpy(sb->data + sb->len, src, len);
    sb->len += len;
    sb->data[sb->len] = '\0';
}

static void sb_append(StrBuf *sb, const char *src) {
    sb_append_n(sb, src, strlen(src));
}

static void sb_append_char(StrBuf *sb, char c) {
    sb_reserve(sb, 1);
    sb->data[sb->len++] = c;
    sb->data[sb->len] = '\0';
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    int needed = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (needed < 0) die("formatting failed");
    sb_reserve(sb, (size_t)needed);
    vsnprintf(sb->data + sb->len, sb->cap - sb->len, fmt, args);
    va_end(args);
    sb->len += (size_t)needed;
}

static char *sb_take(StrBuf *sb) {
    if (!sb->data) return xstrdup("");
    char *out = sb->data;
    sb->data = NULL;
    sb->len = 0;
    sb->cap = 0;
    return out;
}

static bool is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool is_ident_continue(char c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

static void lexer_init(Lexer *lexer, const char *src) {
    lexer->src = src;
    lexer->len = strlen(src);
    lexer->pos = 0;
    lexer->line = 1;
    lexer->column = 1;
}

static char lexer_peek(const Lexer *lexer) {
    if (lexer->pos >= lexer->len) return '\0';
    return lexer->src[lexer->pos];
}

static char lexer_peek_next(const Lexer *lexer) {
    if (lexer->pos + 1 >= lexer->len) return '\0';
    return lexer->src[lexer->pos + 1];
}

static char lexer_advance(Lexer *lexer) {
    char c = lexer_peek(lexer);
    if (c == '\0') return c;
    lexer->pos++;
    if (c == '\n') {
        lexer->line++;
        lexer->column = 1;
    } else {
        lexer->column++;
    }
    return c;
}

static void lexer_skip_ws(Lexer *lexer) {
    for (;;) {
        char c = lexer_peek(lexer);
        if (c == '/' && lexer_peek_next(lexer) == '/') {
            while (c != '\0' && c != '\n') c = lexer_advance(lexer);
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            lexer_advance(lexer);
            continue;
        }
        return;
    }
}

static Token make_token(TokenKind kind, char *text, int line, int column) {
    Token tok;
    tok.kind = kind;
    tok.text = text;
    tok.line = line;
    tok.column = column;
    return tok;
}

static TokenKind keyword_kind(const char *text) {
    if (strcmp(text, "module") == 0) return TOK_KW_MODULE;
    if (strcmp(text, "import") == 0) return TOK_KW_IMPORT;
    if (strcmp(text, "fn") == 0) return TOK_KW_FN;
    if (strcmp(text, "test") == 0) return TOK_KW_TEST;
    if (strcmp(text, "defer") == 0) return TOK_KW_DEFER;
    if (strcmp(text, "let") == 0) return TOK_KW_LET;
    if (strcmp(text, "var") == 0) return TOK_KW_VAR;
    if (strcmp(text, "return") == 0) return TOK_KW_RETURN;
    if (strcmp(text, "print") == 0) return TOK_KW_PRINT;
    return TOK_IDENT;
}

static Token lexer_next(Lexer *lexer) {
    lexer_skip_ws(lexer);
    int line = lexer->line;
    int column = lexer->column;
    char c = lexer_peek(lexer);
    if (c == '\0') return make_token(TOK_EOF, xstrdup(""), line, column);

    if (is_ident_start(c)) {
        size_t start = lexer->pos;
        while (is_ident_continue(lexer_peek(lexer))) lexer_advance(lexer);
        size_t len = lexer->pos - start;
        char *text = xstrndup(lexer->src + start, len);
        return make_token(keyword_kind(text), text, line, column);
    }

    if (c == '"') {
        lexer_advance(lexer);
        StrBuf sb;
        sb_init(&sb);
        for (;;) {
            char ch = lexer_peek(lexer);
            if (ch == '\0') die("%d:%d: unterminated string", line, column);
            if (ch == '"') {
                lexer_advance(lexer);
                break;
            }
            if (ch == '\\') {
                lexer_advance(lexer);
                char esc = lexer_peek(lexer);
                if (esc == '\0') die("%d:%d: bad escape", line, column);
                lexer_advance(lexer);
                switch (esc) {
                    case 'n': sb_append_char(&sb, '\n'); break;
                    case 't': sb_append_char(&sb, '\t'); break;
                    case '"': sb_append_char(&sb, '"'); break;
                    case '\\': sb_append_char(&sb, '\\'); break;
                    default: die("%d:%d: unsupported escape \\%c", line, column, esc);
                }
            } else {
                sb_append_char(&sb, lexer_advance(lexer));
            }
        }
        return make_token(TOK_STRING, sb_take(&sb), line, column);
    }

    lexer_advance(lexer);
    switch (c) {
        case '(': return make_token(TOK_LPAREN, xstrdup("("), line, column);
        case ')': return make_token(TOK_RPAREN, xstrdup(")"), line, column);
        case '{': return make_token(TOK_LBRACE, xstrdup("{"), line, column);
        case '}': return make_token(TOK_RBRACE, xstrdup("}"), line, column);
        case ';': return make_token(TOK_SEMI, xstrdup(";"), line, column);
        case '=': return make_token(TOK_EQ, xstrdup("="), line, column);
        case '.': return make_token(TOK_DOT, xstrdup("."), line, column);
        case ',': return make_token(TOK_COMMA, xstrdup(","), line, column);
        default: die("%d:%d: unexpected character '%c'", line, column, c);
    }
    return make_token(TOK_EOF, xstrdup(""), line, column);
}

static void token_free(Token *tok) {
    free(tok->text);
    tok->text = NULL;
}

static void parser_init(Parser *parser, const char *path, const char *src) {
    lexer_init(&parser->lexer, src);
    parser->path = path;
    parser->current = lexer_next(&parser->lexer);
}

static void parser_advance(Parser *parser) {
    token_free(&parser->current);
    parser->current = lexer_next(&parser->lexer);
}

static bool parser_accept(Parser *parser, TokenKind kind) {
    if (parser->current.kind != kind) return false;
    parser_advance(parser);
    return true;
}

static void parser_fail(Parser *parser, const char *message) {
    Error err = { parser->path, message, parser->current.line, parser->current.column };
    report_error(&err);
    exit(1);
}

static void parser_expect(Parser *parser, TokenKind kind, const char *message) {
    if (parser->current.kind != kind) parser_fail(parser, message);
    parser_advance(parser);
}

static char *parse_path_like(Parser *parser) {
    if (parser->current.kind != TOK_IDENT) parser_fail(parser, "expected identifier path");
    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, parser->current.text);
    parser_advance(parser);
    while (parser_accept(parser, TOK_DOT)) {
        if (parser->current.kind != TOK_IDENT) parser_fail(parser, "expected identifier after '.'");
        sb_append_char(&sb, '.');
        sb_append(&sb, parser->current.text);
        parser_advance(parser);
    }
    return sb_take(&sb);
}

static void push_import(Program *program, char *import_name) {
    if (program->import_len == program->import_cap) {
        size_t next = program->import_cap ? program->import_cap * 2 : 8;
        program->imports = xrealloc(program->imports, next * sizeof(char *));
        program->import_cap = next;
    }
    program->imports[program->import_len++] = import_name;
}

static void push_stmt(Decl *decl, Stmt stmt) {
    if (decl->len == decl->cap) {
        size_t next = decl->cap ? decl->cap * 2 : 8;
        decl->items = xrealloc(decl->items, next * sizeof(Stmt));
        decl->cap = next;
    }
    decl->items[decl->len++] = stmt;
}

static void push_decl(Program *program, Decl decl) {
    if (program->decl_len == program->decl_cap) {
        size_t next = program->decl_cap ? program->decl_cap * 2 : 8;
        program->decls = xrealloc(program->decls, next * sizeof(Decl));
        program->decl_cap = next;
    }
    program->decls[program->decl_len++] = decl;
}

static Expr parse_expr(Parser *parser) {
    Expr expr;
    if (parser->current.kind == TOK_STRING) {
        expr.is_ident = false;
        expr.value = xstrdup(parser->current.text);
        parser_advance(parser);
        return expr;
    }
    if (parser->current.kind == TOK_IDENT) {
        expr.is_ident = true;
        expr.value = xstrdup(parser->current.text);
        parser_advance(parser);
        return expr;
    }
    parser_fail(parser, "expected string literal or identifier");
    expr.is_ident = false;
    expr.value = NULL;
    return expr;
}

static Stmt parse_print_stmt(Parser *parser, bool is_defer) {
    parser_expect(parser, TOK_KW_PRINT, "expected 'print'");
    parser_expect(parser, TOK_LPAREN, "expected '(' after print");
    Expr expr = parse_expr(parser);
    parser_expect(parser, TOK_RPAREN, "expected ')' after print argument");
    parser_expect(parser, TOK_SEMI, "expected ';' after print statement");
    Stmt stmt;
    stmt.kind = is_defer ? STMT_DEFER_PRINT : STMT_PRINT;
    stmt.is_mutable = false;
    stmt.name = NULL;
    stmt.expr = expr;
    return stmt;
}

static Stmt parse_stmt(Parser *parser) {
    if (parser_accept(parser, TOK_KW_DEFER)) {
        return parse_print_stmt(parser, true);
    }
    if (parser->current.kind == TOK_KW_PRINT) {
        return parse_print_stmt(parser, false);
    }
    if (parser->current.kind == TOK_KW_LET || parser->current.kind == TOK_KW_VAR) {
        bool is_mutable = parser->current.kind == TOK_KW_VAR;
        parser_advance(parser);
        if (parser->current.kind != TOK_IDENT) parser_fail(parser, "expected binding name");
        char *name = xstrdup(parser->current.text);
        parser_advance(parser);
        parser_expect(parser, TOK_EQ, "expected '=' after binding name");
        Expr expr = parse_expr(parser);
        if (expr.is_ident) parser_fail(parser, "stage0 bindings only support string literals");
        parser_expect(parser, TOK_SEMI, "expected ';' after binding");
        Stmt stmt;
        stmt.kind = STMT_BIND;
        stmt.is_mutable = is_mutable;
        stmt.name = name;
        stmt.expr = expr;
        return stmt;
    }
    if (parser_accept(parser, TOK_KW_RETURN)) {
        parser_expect(parser, TOK_SEMI, "expected ';' after return");
        Stmt stmt;
        stmt.kind = STMT_RETURN;
        stmt.is_mutable = false;
        stmt.name = NULL;
        stmt.expr.is_ident = false;
        stmt.expr.value = NULL;
        return stmt;
    }
    parser_fail(parser, "unsupported statement in CEL stage0");
    Stmt impossible;
    memset(&impossible, 0, sizeof(impossible));
    return impossible;
}

static Decl parse_decl(Parser *parser) {
    Decl decl;
    memset(&decl, 0, sizeof(decl));
    if (parser_accept(parser, TOK_KW_FN)) {
        decl.is_test = false;
        if (parser->current.kind != TOK_IDENT) parser_fail(parser, "expected function name");
        decl.name = xstrdup(parser->current.text);
        parser_advance(parser);
        parser_expect(parser, TOK_LPAREN, "expected '(' after function name");
        parser_expect(parser, TOK_RPAREN, "expected ')' after function parameters");
    } else if (parser_accept(parser, TOK_KW_TEST)) {
        decl.is_test = true;
        if (parser->current.kind != TOK_STRING) parser_fail(parser, "expected string test name");
        decl.name = xstrdup(parser->current.text);
        parser_advance(parser);
    } else {
        parser_fail(parser, "expected 'fn' or 'test'");
    }
    parser_expect(parser, TOK_LBRACE, "expected '{' to start block");
    while (parser->current.kind != TOK_RBRACE && parser->current.kind != TOK_EOF) {
        push_stmt(&decl, parse_stmt(parser));
    }
    parser_expect(parser, TOK_RBRACE, "expected '}' to close block");
    return decl;
}

static Program parse_program(const char *path, const char *src) {
    Parser parser;
    Program program;
    memset(&program, 0, sizeof(program));
    parser_init(&parser, path, src);

    if (parser_accept(&parser, TOK_KW_MODULE)) {
        program.module_name = parse_path_like(&parser);
        parser_expect(&parser, TOK_SEMI, "expected ';' after module declaration");
    }

    while (parser_accept(&parser, TOK_KW_IMPORT)) {
        char *import_name = parse_path_like(&parser);
        parser_expect(&parser, TOK_SEMI, "expected ';' after import");
        push_import(&program, import_name);
    }

    while (parser.current.kind != TOK_EOF) {
        push_decl(&program, parse_decl(&parser));
    }
    token_free(&parser.current);
    return program;
}

static void free_program(Program *program) {
    free(program->module_name);
    for (size_t i = 0; i < program->import_len; i++) free(program->imports[i]);
    free(program->imports);
    for (size_t i = 0; i < program->decl_len; i++) {
        Decl *decl = &program->decls[i];
        free(decl->name);
        for (size_t j = 0; j < decl->len; j++) {
            free(decl->items[j].name);
            free(decl->items[j].expr.value);
        }
        free(decl->items);
    }
    free(program->decls);
    memset(program, 0, sizeof(*program));
}

typedef struct {
    char **names;
    size_t len;
    size_t cap;
} NameSet;

static bool nameset_contains(const NameSet *set, const char *name) {
    for (size_t i = 0; i < set->len; i++) {
        if (strcmp(set->names[i], name) == 0) return true;
    }
    return false;
}

static void nameset_add(NameSet *set, const char *name) {
    if (nameset_contains(set, name)) die("duplicate binding '%s'", name);
    if (set->len == set->cap) {
        size_t next = set->cap ? set->cap * 2 : 8;
        set->names = xrealloc(set->names, next * sizeof(char *));
        set->cap = next;
    }
    set->names[set->len++] = xstrdup(name);
}

static void nameset_free(NameSet *set) {
    for (size_t i = 0; i < set->len; i++) free(set->names[i]);
    free(set->names);
}

static void validate_program(const Program *program) {
    for (size_t i = 0; i < program->decl_len; i++) {
        const Decl *decl = &program->decls[i];
        NameSet names = {0};
        for (size_t j = 0; j < decl->len; j++) {
            const Stmt *stmt = &decl->items[j];
            if (stmt->kind == STMT_BIND) {
                nameset_add(&names, stmt->name);
                continue;
            }
            if ((stmt->kind == STMT_PRINT || stmt->kind == STMT_DEFER_PRINT) &&
                stmt->expr.is_ident &&
                !nameset_contains(&names, stmt->expr.value)) {
                die("undefined identifier '%s' in '%s'", stmt->expr.value, decl->name);
            }
        }
        nameset_free(&names);
    }
}

static void append_indent(StrBuf *sb, int depth) {
    for (int i = 0; i < depth; i++) sb_append(sb, "    ");
}

static void append_escaped(StrBuf *sb, const char *text) {
    sb_append_char(sb, '"');
    for (const char *p = text; *p; ++p) {
        switch (*p) {
            case '\n': sb_append(sb, "\\n"); break;
            case '\t': sb_append(sb, "\\t"); break;
            case '"': sb_append(sb, "\\\""); break;
            case '\\': sb_append(sb, "\\\\"); break;
            default: sb_append_char(sb, *p); break;
        }
    }
    sb_append_char(sb, '"');
}

static char *format_program(const Program *program) {
    StrBuf sb;
    sb_init(&sb);
    if (program->module_name) {
        sb_printf(&sb, "module %s;\n", program->module_name);
    }
    for (size_t i = 0; i < program->import_len; i++) {
        sb_printf(&sb, "import %s;\n", program->imports[i]);
    }
    if (program->module_name || program->import_len) sb_append_char(&sb, '\n');

    for (size_t i = 0; i < program->decl_len; i++) {
        const Decl *decl = &program->decls[i];
        if (decl->is_test) {
            sb_append(&sb, "test ");
            append_escaped(&sb, decl->name);
            sb_append(&sb, " {\n");
        } else {
            sb_printf(&sb, "fn %s() {\n", decl->name);
        }
        for (size_t j = 0; j < decl->len; j++) {
            const Stmt *stmt = &decl->items[j];
            append_indent(&sb, 1);
            if (stmt->kind == STMT_PRINT || stmt->kind == STMT_DEFER_PRINT) {
                if (stmt->kind == STMT_DEFER_PRINT) sb_append(&sb, "defer ");
                sb_append(&sb, "print(");
                if (stmt->expr.is_ident) {
                    sb_append(&sb, stmt->expr.value);
                } else {
                    append_escaped(&sb, stmt->expr.value);
                }
                sb_append(&sb, ");\n");
            } else if (stmt->kind == STMT_BIND) {
                sb_append(&sb, stmt->is_mutable ? "var " : "let ");
                sb_append(&sb, stmt->name);
                sb_append(&sb, " = ");
                append_escaped(&sb, stmt->expr.value);
                sb_append(&sb, ";\n");
            } else if (stmt->kind == STMT_RETURN) {
                sb_append(&sb, "return;\n");
            }
        }
        sb_append(&sb, "}\n");
        if (i + 1 < program->decl_len) sb_append_char(&sb, '\n');
    }
    return sb_take(&sb);
}

static void emit_c_expr(StrBuf *sb, const Expr *expr) {
    if (expr->is_ident) {
        sb_append(sb, expr->value);
    } else {
        append_escaped(sb, expr->value);
    }
}

static void emit_c_block(StrBuf *sb, const Decl *decl) {
    for (size_t i = 0; i < decl->len; i++) {
        const Stmt *stmt = &decl->items[i];
        if (stmt->kind == STMT_BIND) {
            append_indent(sb, 1);
            sb_printf(sb, "const char *%s = ", stmt->name);
            emit_c_expr(sb, &stmt->expr);
            sb_append(sb, ";\n");
        } else if (stmt->kind == STMT_PRINT) {
            append_indent(sb, 1);
            sb_append(sb, "cel_print(");
            emit_c_expr(sb, &stmt->expr);
            sb_append(sb, ");\n");
        } else if (stmt->kind == STMT_RETURN) {
            append_indent(sb, 1);
            sb_append(sb, "return;\n");
        }
    }

    for (size_t idx = decl->len; idx > 0; idx--) {
        const Stmt *stmt = &decl->items[idx - 1];
        if (stmt->kind != STMT_DEFER_PRINT) continue;
        append_indent(sb, 1);
        sb_append(sb, "cel_print(");
        emit_c_expr(sb, &stmt->expr);
        sb_append(sb, ");\n");
    }
}

static char *emit_c_main(const Program *program) {
    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, "#include <stdio.h>\n\n");
    sb_append(&sb, "static void cel_print(const char *value) {\n");
    sb_append(&sb, "    fputs(value, stdout);\n");
    sb_append(&sb, "    fputc('\\n', stdout);\n");
    sb_append(&sb, "}\n\n");

    const Decl *main_decl = NULL;
    for (size_t i = 0; i < program->decl_len; i++) {
        const Decl *decl = &program->decls[i];
        if (!decl->is_test && strcmp(decl->name, "main") == 0) {
            main_decl = decl;
            break;
        }
    }
    if (!main_decl) die("CEL stage0 run requires fn main()");

    sb_append(&sb, "static void cel_fn_main(void) {\n");
    emit_c_block(&sb, main_decl);
    sb_append(&sb, "}\n\n");
    sb_append(&sb, "int main(void) {\n");
    sb_append(&sb, "    cel_fn_main();\n");
    sb_append(&sb, "    return 0;\n");
    sb_append(&sb, "}\n");
    return sb_take(&sb);
}

static char *sanitize_test_name(const char *name, size_t index) {
    StrBuf sb;
    sb_init(&sb);
    sb_printf(&sb, "cel_test_%zu_", index);
    for (const char *p = name; *p; ++p) {
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9')) {
            sb_append_char(&sb, *p);
        } else {
            sb_append_char(&sb, '_');
        }
    }
    return sb_take(&sb);
}

static char *emit_c_tests(const Program *program) {
    StrBuf sb;
    sb_init(&sb);
    sb_append(&sb, "#include <stdio.h>\n\n");
    sb_append(&sb, "static void cel_print(const char *value) {\n");
    sb_append(&sb, "    fputs(value, stdout);\n");
    sb_append(&sb, "    fputc('\\n', stdout);\n");
    sb_append(&sb, "}\n\n");

    size_t test_count = 0;
    for (size_t i = 0; i < program->decl_len; i++) {
        const Decl *decl = &program->decls[i];
        if (!decl->is_test) continue;
        test_count++;
        char *fn_name = sanitize_test_name(decl->name, i);
        sb_printf(&sb, "static void %s(void) {\n", fn_name);
        emit_c_block(&sb, decl);
        sb_append(&sb, "}\n\n");
        free(fn_name);
    }
    if (test_count == 0) die("CEL stage0 test requires at least one test block");

    sb_append(&sb, "int main(void) {\n");
    sb_printf(&sb, "    size_t total = %zu;\n", test_count);
    sb_append(&sb, "    size_t passed = 0;\n");
    sb_append(&sb, "    printf(\"running %zu CEL test(s)\\n\", total);\n");
    for (size_t i = 0; i < program->decl_len; i++) {
        const Decl *decl = &program->decls[i];
        if (!decl->is_test) continue;
        char *fn_name = sanitize_test_name(decl->name, i);
        sb_printf(&sb, "    %s();\n", fn_name);
        sb_printf(&sb, "    printf(\"[pass] %s\\n\");\n", decl->name);
        sb_append(&sb, "    passed++;\n");
        free(fn_name);
    }
    sb_append(&sb, "    printf(\"ok: %zu/%zu passed\\n\", passed, total);\n");
    sb_append(&sb, "    return passed == total ? 0 : 1;\n");
    sb_append(&sb, "}\n");
    return sb_take(&sb);
}

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) die("failed to open %s: %s", path, strerror(errno));
    if (fseek(file, 0, SEEK_END) != 0) die("failed to seek %s", path);
    long size = ftell(file);
    if (size < 0) die("failed to tell %s", path);
    rewind(file);
    char *data = xmalloc((size_t)size + 1);
    size_t read = fread(data, 1, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) die("failed to read %s", path);
    data[read] = '\0';
    return data;
}

static void write_file(const char *path, const char *content) {
    FILE *file = fopen(path, "wb");
    if (!file) die("failed to write %s: %s", path, strerror(errno));
    size_t len = strlen(content);
    if (fwrite(content, 1, len, file) != len) die("failed to write %s", path);
    fclose(file);
}

static int run_command(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) die("fork failed: %s", strerror(errno));
    if (pid == 0) {
        execvp(argv[0], argv);
        fprintf(stderr, "failed to exec %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) die("waitpid failed: %s", strerror(errno));
    if (!WIFEXITED(status)) return 1;
    return WEXITSTATUS(status);
}

static void build_and_run(const char *c_source, bool run_binary) {
    char c_template[] = "/tmp/cel-stage0-XXXXXX";
    int fd = mkstemp(c_template);
    if (fd < 0) die("mkstemp failed: %s", strerror(errno));
    close(fd);

    StrBuf c_path;
    sb_init(&c_path);
    sb_append(&c_path, c_template);
    StrBuf exe_path;
    sb_init(&exe_path);
    sb_printf(&exe_path, "%s.bin", c_template);
    write_file(c_path.data, c_source);

    char *cc_argv[] = {
        (char *)(getenv("CC") ? getenv("CC") : "cc"),
        "-std=c11",
        "-D_POSIX_C_SOURCE=200809L",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-x",
        "c",
        c_path.data,
        "-o",
        exe_path.data,
        NULL,
    };
    if (run_command(cc_argv) != 0) die("C backend compilation failed");
    if (run_binary) {
        char *run_argv[] = { exe_path.data, NULL };
        if (run_command(run_argv) != 0) die("generated program failed");
    }

    unlink(c_path.data);
    unlink(exe_path.data);
    free(c_path.data);
    free(exe_path.data);
}

static void usage(FILE *stream) {
    fprintf(stream,
            "CEL stage0 compiler\n"
            "\n"
            "Usage:\n"
            "  cel check <file.cel>\n"
            "  cel fmt [-w] <file.cel>\n"
            "  cel run <file.cel>\n"
            "  cel test <file.cel>\n"
            "  cel emit-c <file.cel>\n");
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "help") == 0) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2 ? 1 : 0;
    }

    const char *command = argv[1];
    bool write_in_place = false;
    const char *path = NULL;

    if (strcmp(command, "fmt") == 0) {
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--write") == 0) {
                write_in_place = true;
            } else {
                path = argv[i];
            }
        }
    } else if (argc >= 3) {
        path = argv[2];
    }

    if (!path) {
        usage(stderr);
        return 1;
    }

    char *source = read_file(path);
    Program program = parse_program(path, source);
    validate_program(&program);

    if (strcmp(command, "check") == 0) {
        printf("ok: %s\n", path);
    } else if (strcmp(command, "fmt") == 0) {
        char *formatted = format_program(&program);
        if (write_in_place) {
            write_file(path, formatted);
        } else {
            fputs(formatted, stdout);
        }
        free(formatted);
    } else if (strcmp(command, "emit-c") == 0) {
        char *c_source = emit_c_main(&program);
        fputs(c_source, stdout);
        free(c_source);
    } else if (strcmp(command, "run") == 0) {
        char *c_source = emit_c_main(&program);
        build_and_run(c_source, true);
        free(c_source);
    } else if (strcmp(command, "test") == 0) {
        char *c_source = emit_c_tests(&program);
        build_and_run(c_source, true);
        free(c_source);
    } else {
        usage(stderr);
        free_program(&program);
        free(source);
        return 1;
    }

    free_program(&program);
    free(source);
    return 0;
}
