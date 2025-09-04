// ouro_lang.cc - OuroLang implementation
// Provided by user; integrated into repository.

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

// Token definitions and structures
enum class TokenType {
    LET, FN, IF, ELSE, RETURN, FOR, IN, ASYNC, AWAIT, GPU,
    INT, FLOAT, STRING, IDENTIFIER, NUMBER, STRING_LITERAL,
    COLON, EQUALS, LPAREN, RPAREN, LBRACE, RBRACE, SEMICOLON, COMMA,
    PLUS, MINUS, MUL, DIV, GT, DOTDOT, ARROW, EOF_TOKEN
};

struct Token {
    TokenType type;
    std::string value;
    int line;
};

class Lexer {
    std::string source;
    size_t pos = 0;
    int line = 1;

public:
    explicit Lexer(const std::string &src) : source(src) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos < source.size()) {
            char c = source[pos];
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (c == '\n') line++;
                pos++;
                continue;
            }
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                tokens.push_back(parse_identifier());
            } else if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
                tokens.push_back(parse_number());
            } else if (c == '"') {
                tokens.push_back(parse_string());
            } else {
                tokens.push_back(parse_symbol());
            }
        }
        tokens.push_back({TokenType::EOF_TOKEN, "", line});
        return tokens;
    }

private:
    Token parse_identifier() {
        std::string value;
        while (pos < source.size() &&
               (std::isalnum(static_cast<unsigned char>(source[pos])) ||
                source[pos] == '_')) {
            value += source[pos++];
        }
        if (value == "let") return {TokenType::LET, value, line};
        if (value == "fn") return {TokenType::FN, value, line};
        if (value == "if") return {TokenType::IF, value, line};
        if (value == "else") return {TokenType::ELSE, value, line};
        if (value == "return") return {TokenType::RETURN, value, line};
        if (value == "for") return {TokenType::FOR, value, line};
        if (value == "in") return {TokenType::IN, value, line};
        if (value == "async") return {TokenType::ASYNC, value, line};
        if (value == "await") return {TokenType::AWAIT, value, line};
        if (value == "gpu") return {TokenType::GPU, value, line};
        if (value == "int") return {TokenType::INT, value, line};
        if (value == "float") return {TokenType::FLOAT, value, line};
        if (value == "string") return {TokenType::STRING, value, line};
        return {TokenType::IDENTIFIER, value, line};
    }

    Token parse_number() {
        std::string value;
        bool has_dot = false;
        while (pos < source.size() && (std::isdigit(static_cast<unsigned char>(source[pos])) || source[pos] == '.')) {
            if (source[pos] == '.') has_dot = true;
            value += source[pos++];
        }
        return {TokenType::NUMBER, value, line};
    }

    Token parse_string() {
        std::string value;
        pos++; // Skip opening quote
        while (pos < source.size() && source[pos] != '"') {
            value += source[pos++];
        }
        pos++; // Skip closing quote
        return {TokenType::STRING_LITERAL, value, line};
    }

    Token parse_symbol() {
        char c = source[pos++];
        switch (c) {
            case ':': return {TokenType::COLON, ":", line};
            case '=': return {TokenType::EQUALS, "=", line};
            case '(': return {TokenType::LPAREN, "(", line};
            case ')': return {TokenType::RPAREN, ")", line};
            case '{': return {TokenType::LBRACE, "{", line};
            case '}': return {TokenType::RBRACE, "}", line};
            case ';': return {TokenType::SEMICOLON, ";", line};
            case ',': return {TokenType::COMMA, ",", line};
            case '+': return {TokenType::PLUS, "+", line};
            case '-':
                if (pos < source.size() && source[pos] == '>') {
                    pos++;
                    return {TokenType::ARROW, "->", line};
                }
                return {TokenType::MINUS, "-", line};
            case '*': return {TokenType::MUL, "*", line};
            case '/': return {TokenType::DIV, "/", line};
            case '>': return {TokenType::GT, ">", line};
            case '.':
                if (pos < source.size() && source[pos] == '.') {
                    pos++;
                    return {TokenType::DOTDOT, "..", line};
                }
            default:
                throw std::runtime_error("Unknown symbol at line " + std::to_string(line));
        }
    }
};

// Forward declarations for AST structures
struct Expr;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

// Expression node types
struct NumberExpr { double value; };
struct StringExpr { std::string value; };
struct IdentExpr { std::string name; };
struct BinaryExpr { TokenType op; ExprPtr left; ExprPtr right; };
struct CallExpr { std::string name; std::vector<ExprPtr> args; };
struct AwaitExpr { ExprPtr expr; };

using ExprVariant = std::variant<NumberExpr, StringExpr, IdentExpr, BinaryExpr, CallExpr, AwaitExpr>;
struct Expr { ExprVariant value; };

// Statement node types
struct VarDeclStmt { std::string name; std::string type; ExprPtr value; };
struct FnDeclStmt {
    std::string name;
    std::vector<std::pair<std::string, std::string>> params;
    std::string return_type;
    std::vector<StmtPtr> body;
    bool is_async;
    bool is_gpu;
    bool is_generic;
    std::vector<std::string> generic_params;
};
struct IfStmt { ExprPtr condition; std::vector<StmtPtr> then_branch; std::vector<StmtPtr> else_branch; };
struct ForStmt { std::string var; ExprPtr start; ExprPtr end; std::vector<StmtPtr> body; };
struct ReturnStmt { ExprPtr value; };

using StmtVariant = std::variant<VarDeclStmt, FnDeclStmt, IfStmt, ForStmt, ReturnStmt>;
struct Stmt { StmtVariant value; };

class Parser {
    std::vector<Token> tokens;
    size_t pos = 0;

public:
    explicit Parser(const std::vector<Token> &t) : tokens(t) {}

    std::vector<StmtPtr> parse() {
        std::vector<StmtPtr> stmts;
        while (tokens[pos].type != TokenType::EOF_TOKEN) {
            stmts.push_back(parse_stmt());
        }
        return stmts;
    }

private:
    Token peek() const { return tokens[pos]; }
    Token advance() { return tokens[pos++]; }
    Token consume(TokenType type, const std::string &msg) {
        if (peek().type == type) return advance();
        throw std::runtime_error(msg + " at line " + std::to_string(peek().line));
    }

    StmtPtr parse_stmt() {
        if (peek().type == TokenType::LET) return parse_var_decl();
        if (peek().type == TokenType::FN || peek().type == TokenType::ASYNC || peek().type == TokenType::GPU) {
            return parse_fn_decl();
        }
        if (peek().type == TokenType::IF) return parse_if_stmt();
        if (peek().type == TokenType::FOR) return parse_for_stmt();
        if (peek().type == TokenType::RETURN) return parse_return_stmt();
        throw std::runtime_error("Unexpected token at line " + std::to_string(peek().line));
    }

    StmtPtr parse_var_decl() {
        consume(TokenType::LET, "Expected 'let'");
        auto name = consume(TokenType::IDENTIFIER, "Expected identifier").value;
        std::string type;
        if (peek().type == TokenType::COLON) {
            consume(TokenType::COLON, "Expected ':'");
            type = consume(TokenType::IDENTIFIER, "Expected type").value;
        }
        consume(TokenType::EQUALS, "Expected '='");
        auto value = parse_expr();
        consume(TokenType::SEMICOLON, "Expected ';'");
        return std::make_unique<Stmt>(VarDeclStmt{name, type, std::move(value)});
    }

    StmtPtr parse_fn_decl() {
        bool is_async = false, is_gpu = false, is_generic = false;
        std::vector<std::string> generic_params;
        if (peek().type == TokenType::ASYNC) {
            consume(TokenType::ASYNC, "");
            is_async = true;
        } else if (peek().type == TokenType::GPU) {
            consume(TokenType::GPU, "");
            is_gpu = true;
        }
        consume(TokenType::FN, "Expected 'fn'");
        auto name = consume(TokenType::IDENTIFIER, "Expected identifier").value;
        if (peek().type == TokenType::GT) {
            consume(TokenType::GT, "Expected '<'");
            while (peek().type != TokenType::GT) {
                generic_params.push_back(consume(TokenType::IDENTIFIER, "Expected generic param").value);
                if (peek().type == TokenType::COMMA) consume(TokenType::COMMA, "");
            }
            consume(TokenType::GT, "Expected '>'");
            is_generic = true;
        }
        consume(TokenType::LPAREN, "Expected '('");
        std::vector<std::pair<std::string, std::string>> params;
        if (peek().type != TokenType::RPAREN) {
            do {
                auto param_name = consume(TokenType::IDENTIFIER, "Expected param name").value;
                consume(TokenType::COLON, "Expected ':'");
                auto param_type = consume(TokenType::IDENTIFIER, "Expected param type").value;
                params.push_back({param_name, param_type});
                if (peek().type == TokenType::COMMA) consume(TokenType::COMMA, "");
            } while (peek().type != TokenType::RPAREN);
        }
        consume(TokenType::RPAREN, "Expected ')'");
        std::string return_type;
        if (peek().type == TokenType::ARROW) {
            consume(TokenType::ARROW, "Expected '->'");
            return_type = consume(TokenType::IDENTIFIER, "Expected return type").value;
        }
        consume(TokenType::LBRACE, "Expected '{'");
        std::vector<StmtPtr> body;
        while (peek().type != TokenType::RBRACE) {
            body.push_back(parse_stmt());
        }
        consume(TokenType::RBRACE, "Expected '}'");
        return std::make_unique<Stmt>(FnDeclStmt{name, params, return_type, std::move(body),
                                                is_async, is_gpu, is_generic, generic_params});
    }

    StmtPtr parse_if_stmt() {
        consume(TokenType::IF, "Expected 'if'");
        auto condition = parse_expr();
        consume(TokenType::LBRACE, "Expected '{'");
        std::vector<StmtPtr> then_branch;
        while (peek().type != TokenType::RBRACE && peek().type != TokenType::ELSE) {
            then_branch.push_back(parse_stmt());
        }
        consume(TokenType::RBRACE, "Expected '}'");
        std::vector<StmtPtr> else_branch;
        if (peek().type == TokenType::ELSE) {
            consume(TokenType::ELSE, "");
            consume(TokenType::LBRACE, "Expected '{'");
            while (peek().type != TokenType::RBRACE) {
                else_branch.push_back(parse_stmt());
            }
            consume(TokenType::RBRACE, "Expected '}'");
        }
        return std::make_unique<Stmt>(IfStmt{std::move(condition), std::move(then_branch), std::move(else_branch)});
    }

    StmtPtr parse_for_stmt() {
        consume(TokenType::FOR, "Expected 'for'");
        auto var = consume(TokenType::IDENTIFIER, "Expected loop variable").value;
        consume(TokenType::IN, "Expected 'in'");
        auto start = parse_expr();
        consume(TokenType::DOTDOT, "Expected '..'");
        auto end = parse_expr();
        consume(TokenType::LBRACE, "Expected '{'");
        std::vector<StmtPtr> body;
        while (peek().type != TokenType::RBRACE) {
            body.push_back(parse_stmt());
        }
        consume(TokenType::RBRACE, "Expected '}'");
        return std::make_unique<Stmt>(ForStmt{var, std::move(start), std::move(end), std::move(body)});
    }

    StmtPtr parse_return_stmt() {
        consume(TokenType::RETURN, "Expected 'return'");
        ExprPtr value;
        if (peek().type != TokenType::SEMICOLON) {
            value = parse_expr();
        }
        consume(TokenType::SEMICOLON, "Expected ';'");
        return std::make_unique<Stmt>(ReturnStmt{std::move(value)});
    }

    ExprPtr parse_expr() { return parse_binary_expr(0); }

    ExprPtr parse_binary_expr(int precedence) {
        auto left = parse_primary_expr();
        while (true) {
            TokenType op = peek().type;
            int op_precedence = get_precedence(op);
            if (op_precedence <= precedence) break;
            advance();
            auto right = parse_binary_expr(op_precedence);
            left = std::make_unique<Expr>(BinaryExpr{op, std::move(left), std::move(right)});
        }
        return left;
    }

    int get_precedence(TokenType op) {
        switch (op) {
            case TokenType::MUL:
            case TokenType::DIV: return 2;
            case TokenType::PLUS:
            case TokenType::MINUS: return 1;
            case TokenType::GT: return 0;
            default: return -1;
        }
    }

    ExprPtr parse_primary_expr() {
        if (peek().type == TokenType::NUMBER) {
            double val = std::stod(consume(TokenType::NUMBER, "Expected number").value);
            return std::make_unique<Expr>(NumberExpr{val});
        }
        if (peek().type == TokenType::STRING_LITERAL) {
            auto val = consume(TokenType::STRING_LITERAL, "Expected string").value;
            return std::make_unique<Expr>(StringExpr{val});
        }
        if (peek().type == TokenType::IDENTIFIER) {
            auto name = consume(TokenType::IDENTIFIER, "Expected identifier").value;
            if (peek().type == TokenType::LPAREN) {
                consume(TokenType::LPAREN, "Expected '('");
                std::vector<ExprPtr> args;
                if (peek().type != TokenType::RPAREN) {
                    do {
                        args.push_back(parse_expr());
                        if (peek().type == TokenType::COMMA) consume(TokenType::COMMA, "");
                    } while (peek().type != TokenType::RPAREN);
                }
                consume(TokenType::RPAREN, "Expected ')'");
                return std::make_unique<Expr>(CallExpr{name, std::move(args)});
            }
            return std::make_unique<Expr>(IdentExpr{name});
        }
        if (peek().type == TokenType::AWAIT) {
            consume(TokenType::AWAIT, "Expected 'await'");
            auto expr = parse_expr();
            return std::make_unique<Expr>(AwaitExpr{std::move(expr)});
        }
        throw std::runtime_error("Expected expression at line " + std::to_string(peek().line));
    }
};

class TypeChecker {
    std::map<std::string, std::string> env;
    std::map<std::string, FnDeclStmt *> functions;

public:
    void check(const std::vector<StmtPtr> &stmts) {
        for (const auto &stmt : stmts) {
            check_stmt(*stmt);
        }
    }

private:
    void check_stmt(const Stmt &stmt) {
        if (auto *var = std::get_if<VarDeclStmt>(&stmt.value)) {
            auto inferred_type = infer_type(var->value.get());
            if (!var->type.empty() && var->type != inferred_type) {
                throw std::runtime_error("Type mismatch for " + var->name);
            }
            env[var->name] = var->type.empty() ? inferred_type : var->type;
        } else if (auto *fn = std::get_if<FnDeclStmt>(&stmt.value)) {
            functions[fn->name] = fn;
            auto saved = env;
            for (const auto &param : fn->params) {
                env[param.first] = param.second;
            }
            for (const auto &body_stmt : fn->body) {
                check_stmt(*body_stmt);
            }
            env = saved;
        } else if (auto *if_stmt = std::get_if<IfStmt>(&stmt.value)) {
            if (infer_type(if_stmt->condition.get()) != "int") {
                throw std::runtime_error("If condition must be int");
            }
            for (const auto &s : if_stmt->then_branch) check_stmt(*s);
            for (const auto &s : if_stmt->else_branch) check_stmt(*s);
        } else if (auto *for_stmt = std::get_if<ForStmt>(&stmt.value)) {
            if (infer_type(for_stmt->start.get()) != "int" || infer_type(for_stmt->end.get()) != "int") {
                throw std::runtime_error("For loop bounds must be int");
            }
            env[for_stmt->var] = "int";
            for (const auto &s : for_stmt->body) check_stmt(*s);
        } else if (auto *ret = std::get_if<ReturnStmt>(&stmt.value)) {
            if (ret->value) {
                /* additional checks could go here */
            }
        }
    }

    std::string infer_type(const Expr *expr) {
        if (std::holds_alternative<NumberExpr>(expr->value)) {
            return "float";
        } else if (std::holds_alternative<StringExpr>(expr->value)) {
            return "string";
        } else if (auto *ident = std::get_if<IdentExpr>(&expr->value)) {
            auto it = env.find(ident->name);
            if (it != env.end()) return it->second;
            throw std::runtime_error("Undefined variable: " + ident->name);
        } else if (auto *bin = std::get_if<BinaryExpr>(&expr->value)) {
            auto left_type = infer_type(bin->left.get());
            auto right_type = infer_type(bin->right.get());
            if (left_type != right_type) throw std::runtime_error("Type mismatch in binary op");
            if (bin->op == TokenType::GT) return "int";
            return left_type;
        } else if (auto *call = std::get_if<CallExpr>(&expr->value)) {
            auto it = functions.find(call->name);
            if (it != functions.end()) {
                return it->second->return_type;
            }
            throw std::runtime_error("Undefined function: " + call->name);
        }
        return "unknown";
    }
};

class Interpreter {
    using Value = std::variant<std::monostate, double, std::string>;
    using NativeFn = std::function<Value(const std::vector<Value> &, Interpreter &)>;

public:
    std::map<std::string, std::variant<Value, NativeFn>> env;
    Value return_value;

    Interpreter() {
        env["print"] = NativeFn{[](const std::vector<Value> &args, Interpreter &) -> Value {
            for (const auto &arg : args) {
                if (std::holds_alternative<double>(arg)) {
                    std::cout << std::get<double>(arg) << ' ';
                } else if (std::holds_alternative<std::string>(arg)) {
                    std::cout << std::get<std::string>(arg) << ' ';
                }
            }
            std::cout << std::endl;
            return {};
        }};

        env["sleep"] = NativeFn{[](const std::vector<Value> &args, Interpreter &) -> Value {
            if (!args.empty() && std::holds_alternative<double>(args[0])) {
                auto ms = static_cast<int>(std::get<double>(args[0]));
                std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            }
            return {};
        }};
    }

    void run(const std::string &source) {
        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        TypeChecker checker;
        checker.check(ast);
        for (const auto &stmt : ast) {
            execute_stmt(*stmt);
        }
    }

private:
    void execute_stmt(const Stmt &stmt) {
        if (auto *var = std::get_if<VarDeclStmt>(&stmt.value)) {
            env[var->name] = evaluate_expr(*var->value);
        } else if (auto *fn = std::get_if<FnDeclStmt>(&stmt.value)) {
            auto func = [this, fn](const std::vector<Value> &args, Interpreter &interpreter) -> Value {
                std::map<std::string, std::variant<Value, NativeFn>> fn_env = env;
                if (args.size() != fn->params.size()) {
                    throw std::runtime_error("Argument count mismatch");
                }
                for (size_t i = 0; i < args.size(); ++i) {
                    fn_env[fn->params[i].first] = args[i];
                }
                if (fn->is_async) {
                    auto future = std::async(std::launch::async, [fn, fn_env, &interpreter]() mutable {
                        Interpreter local_interp;
                        local_interp.env = fn_env;
                        for (const auto &body_stmt : fn->body) {
                            local_interp.execute_stmt(*body_stmt);
                        }
                        return local_interp.return_value;
                    });
                    return future.get();
                } else if (fn->is_gpu) {
                    std::cout << "GPU function " << fn->name << " called (placeholder)\n";
                    return {};
                } else {
                    Interpreter local_interp;
                    local_interp.env = fn_env;
                    for (const auto &body_stmt : fn->body) {
                        local_interp.execute_stmt(*body_stmt);
                    }
                    return local_interp.return_value;
                }
            };
            env[fn->name] = func;
        } else if (auto *if_stmt = std::get_if<IfStmt>(&stmt.value)) {
            auto cond = evaluate_expr(*if_stmt->condition);
            bool is_true = false;
            if (std::holds_alternative<double>(cond)) {
                is_true = std::get<double>(cond) != 0;
            }
            if (is_true) {
                for (const auto &s : if_stmt->then_branch) execute_stmt(*s);
            } else {
                for (const auto &s : if_stmt->else_branch) execute_stmt(*s);
            }
        } else if (auto *for_stmt = std::get_if<ForStmt>(&stmt.value)) {
            auto start_val = evaluate_expr(*for_stmt->start);
            auto end_val = evaluate_expr(*for_stmt->end);
            if (std::holds_alternative<double>(start_val) && std::holds_alternative<double>(end_val)) {
                int s = static_cast<int>(std::get<double>(start_val));
                int e = static_cast<int>(std::get<double>(end_val));
                for (int i = s; i < e; ++i) {
                    env[for_stmt->var] = static_cast<double>(i);
                    for (const auto &s : for_stmt->body) {
                        execute_stmt(*s);
                    }
                }
            }
        } else if (auto *ret = std::get_if<ReturnStmt>(&stmt.value)) {
            if (ret->value) {
                return_value = evaluate_expr(*ret->value);
            } else {
                return_value = {};
            }
        }
    }

    Value evaluate_expr(const Expr &expr) {
        if (auto *num = std::get_if<NumberExpr>(&expr.value)) {
            return num->value;
        } else if (auto *str = std::get_if<StringExpr>(&expr.value)) {
            return str->value;
        } else if (auto *ident = std::get_if<IdentExpr>(&expr.value)) {
            auto it = env.find(ident->name);
            if (it != env.end()) return std::get<Value>(it->second);
            throw std::runtime_error("Undefined variable: " + ident->name);
        } else if (auto *bin = std::get_if<BinaryExpr>(&expr.value)) {
            auto left = evaluate_expr(*bin->left);
            auto right = evaluate_expr(*bin->right);
            if (std::holds_alternative<double>(left) && std::holds_alternative<double>(right)) {
                double l = std::get<double>(left);
                double r = std::get<double>(right);
                switch (bin->op) {
                    case TokenType::PLUS: return l + r;
                    case TokenType::MINUS: return l - r;
                    case TokenType::MUL: return l * r;
                    case TokenType::DIV: return l / r;
                    case TokenType::GT: return static_cast<double>(l > r);
                    default: throw std::runtime_error("Invalid operator");
                }
            }
        } else if (auto *call = std::get_if<CallExpr>(&expr.value)) {
            auto it = env.find(call->name);
            if (it == env.end()) throw std::runtime_error("Undefined function: " + call->name);
            auto func = std::get<NativeFn>(it->second);
            std::vector<Value> args;
            for (const auto &arg : call->args) {
                args.push_back(evaluate_expr(*arg));
            }
            return func(args, *this);
        } else if (auto *await = std::get_if<AwaitExpr>(&expr.value)) {
            return evaluate_expr(*await->expr); // sync for now
        }
        throw std::runtime_error("Invalid expression");
    }
};

void repl() {
    Interpreter interp;
    std::string line;
    std::cout << "OuroLang REPL (type 'exit' to quit)\n";
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, line);
        if (line == "exit") break;
        try {
            interp.run(line);
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
}

int main() {
    repl();
    return 0;
}
