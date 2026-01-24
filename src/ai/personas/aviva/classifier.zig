//! Aviva Query Classifier
//!
//! Classifies incoming queries to determine the appropriate response strategy.
//! Enables Aviva to optimize her response format based on query type.
//!
//! Features:
//! - Query type classification (code, factual, explanation, etc.)
//! - Domain detection for specialized handling
//! - Complexity estimation
//! - Response format recommendation

const std = @import("std");

/// Types of queries Aviva can handle.
pub const QueryType = enum {
    /// Request to write or generate code.
    code_request,
    /// Request to fix or debug existing code.
    debugging,
    /// Request for factual information.
    factual_query,
    /// Request for an explanation of a concept.
    explanation,
    /// Request for documentation or API reference.
    documentation,
    /// Request to compare or evaluate options.
    comparison,
    /// Request to optimize or improve something.
    optimization,
    /// General query not fitting other categories.
    general,

    pub fn getDescription(self: QueryType) []const u8 {
        return switch (self) {
            .code_request => "Code generation request",
            .debugging => "Debugging or error fixing",
            .factual_query => "Factual information request",
            .explanation => "Concept explanation",
            .documentation => "Documentation or API reference",
            .comparison => "Comparison or evaluation",
            .optimization => "Performance or code optimization",
            .general => "General query",
        };
    }

    pub fn recommendsCodeBlock(self: QueryType) bool {
        return switch (self) {
            .code_request, .debugging, .optimization => true,
            else => false,
        };
    }

    pub fn recommendsBrevity(self: QueryType) bool {
        return switch (self) {
            .factual_query, .documentation => true,
            else => false,
        };
    }
};

/// Programming language detected in query.
pub const Language = enum {
    unknown,
    zig,
    rust,
    python,
    javascript,
    typescript,
    go,
    c,
    cpp,
    java,
    csharp,
    ruby,
    sql,
    bash,
    html,
    css,

    pub fn getFileExtension(self: Language) []const u8 {
        return switch (self) {
            .zig => ".zig",
            .rust => ".rs",
            .python => ".py",
            .javascript => ".js",
            .typescript => ".ts",
            .go => ".go",
            .c => ".c",
            .cpp => ".cpp",
            .java => ".java",
            .csharp => ".cs",
            .ruby => ".rb",
            .sql => ".sql",
            .bash => ".sh",
            .html => ".html",
            .css => ".css",
            .unknown => "",
        };
    }
};

/// Domain of expertise for the query.
pub const Domain = enum {
    general,
    web_development,
    systems_programming,
    data_science,
    devops,
    databases,
    networking,
    security,
    mobile,
    ai_ml,
    algorithms,
    testing,

    pub fn getDescription(self: Domain) []const u8 {
        return switch (self) {
            .general => "General programming",
            .web_development => "Web development",
            .systems_programming => "Systems programming",
            .data_science => "Data science and analytics",
            .devops => "DevOps and infrastructure",
            .databases => "Databases and storage",
            .networking => "Networking and protocols",
            .security => "Security and cryptography",
            .mobile => "Mobile development",
            .ai_ml => "AI and machine learning",
            .algorithms => "Algorithms and data structures",
            .testing => "Testing and quality assurance",
        };
    }
};

/// Complexity level of the query.
pub const Complexity = enum {
    trivial,
    simple,
    moderate,
    complex,
    expert,

    pub fn getEstimatedTokens(self: Complexity) usize {
        return switch (self) {
            .trivial => 100,
            .simple => 250,
            .moderate => 500,
            .complex => 1000,
            .expert => 2000,
        };
    }
};

/// Result of query classification.
pub const ClassificationResult = struct {
    /// Primary query type.
    query_type: QueryType,
    /// Confidence in the classification (0.0 - 1.0).
    confidence: f32,
    /// Detected programming language (if any).
    language: Language,
    /// Domain of expertise.
    domain: Domain,
    /// Estimated complexity.
    complexity: Complexity,
    /// Secondary query types (if mixed).
    secondary_types: [2]?QueryType = [_]?QueryType{null} ** 2,
    /// Keywords that influenced classification.
    key_indicators: [5]?[]const u8 = [_]?[]const u8{null} ** 5,
};

/// Pattern for classification detection.
const ClassificationPattern = struct {
    keywords: []const []const u8,
    query_type: QueryType,
    weight: f32,
};

/// Patterns for query type detection.
const QUERY_PATTERNS = [_]ClassificationPattern{
    // Code request patterns
    .{
        .keywords = &[_][]const u8{ "write", "create", "implement", "build", "generate", "code", "function", "class", "method" },
        .query_type = .code_request,
        .weight = 0.9,
    },
    // Debugging patterns
    .{
        .keywords = &[_][]const u8{ "fix", "debug", "error", "bug", "crash", "failing", "broken", "doesn't work", "not working" },
        .query_type = .debugging,
        .weight = 0.9,
    },
    // Factual patterns
    .{
        .keywords = &[_][]const u8{ "what is", "what are", "define", "list", "tell me", "when was", "who" },
        .query_type = .factual_query,
        .weight = 0.8,
    },
    // Explanation patterns
    .{
        .keywords = &[_][]const u8{ "explain", "how does", "why does", "how do", "understand", "learn", "tutorial" },
        .query_type = .explanation,
        .weight = 0.85,
    },
    // Documentation patterns
    .{
        .keywords = &[_][]const u8{ "documentation", "docs", "reference", "api", "signature", "parameters", "usage" },
        .query_type = .documentation,
        .weight = 0.85,
    },
    // Comparison patterns
    .{
        .keywords = &[_][]const u8{ "compare", "vs", "versus", "difference", "better", "which", "pros and cons" },
        .query_type = .comparison,
        .weight = 0.85,
    },
    // Optimization patterns
    .{
        .keywords = &[_][]const u8{ "optimize", "improve", "faster", "efficient", "performance", "refactor", "clean up" },
        .query_type = .optimization,
        .weight = 0.85,
    },
};

/// Language detection patterns.
const LANGUAGE_PATTERNS = [_]struct { keywords: []const []const u8, language: Language }{
    .{ .keywords = &[_][]const u8{ "zig", ".zig", "comptime", "allocator" }, .language = .zig },
    .{ .keywords = &[_][]const u8{ "rust", ".rs", "cargo", "impl", "trait" }, .language = .rust },
    .{ .keywords = &[_][]const u8{ "python", ".py", "pip", "def ", "import " }, .language = .python },
    .{ .keywords = &[_][]const u8{ "javascript", ".js", "npm", "const ", "function" }, .language = .javascript },
    .{ .keywords = &[_][]const u8{ "typescript", ".ts", "interface", ": string", ": number" }, .language = .typescript },
    .{ .keywords = &[_][]const u8{ "golang", "go", ".go", "func ", "package " }, .language = .go },
    .{ .keywords = &[_][]const u8{ " c ", ".c", "malloc", "printf", "#include" }, .language = .c },
    .{ .keywords = &[_][]const u8{ "c++", "cpp", ".cpp", "std::", "cout" }, .language = .cpp },
    .{ .keywords = &[_][]const u8{ "java", ".java", "public class", "System.out" }, .language = .java },
    .{ .keywords = &[_][]const u8{ "c#", "csharp", ".cs", "Console.", "namespace" }, .language = .csharp },
    .{ .keywords = &[_][]const u8{ "ruby", ".rb", "def ", "end", "puts" }, .language = .ruby },
    .{ .keywords = &[_][]const u8{ "sql", "select", "insert", "update", "delete", "from" }, .language = .sql },
    .{ .keywords = &[_][]const u8{ "bash", "shell", ".sh", "#!/bin" }, .language = .bash },
    .{ .keywords = &[_][]const u8{ "html", ".html", "<div", "<body", "<html" }, .language = .html },
    .{ .keywords = &[_][]const u8{ "css", ".css", "margin:", "padding:", "display:" }, .language = .css },
};

/// Domain detection patterns.
const DOMAIN_PATTERNS = [_]struct { keywords: []const []const u8, domain: Domain }{
    .{ .keywords = &[_][]const u8{ "react", "vue", "angular", "html", "css", "dom", "frontend", "backend", "api" }, .domain = .web_development },
    .{ .keywords = &[_][]const u8{ "kernel", "memory", "syscall", "thread", "process", "os", "embedded" }, .domain = .systems_programming },
    .{ .keywords = &[_][]const u8{ "pandas", "numpy", "data", "analytics", "statistics", "dataset", "visualization" }, .domain = .data_science },
    .{ .keywords = &[_][]const u8{ "docker", "kubernetes", "ci/cd", "deploy", "pipeline", "terraform", "aws", "cloud" }, .domain = .devops },
    .{ .keywords = &[_][]const u8{ "database", "sql", "query", "index", "postgresql", "mysql", "mongodb" }, .domain = .databases },
    .{ .keywords = &[_][]const u8{ "network", "socket", "tcp", "http", "protocol", "packet" }, .domain = .networking },
    .{ .keywords = &[_][]const u8{ "security", "encryption", "auth", "vulnerability", "hash", "crypto" }, .domain = .security },
    .{ .keywords = &[_][]const u8{ "ios", "android", "mobile", "swift", "kotlin", "flutter" }, .domain = .mobile },
    .{ .keywords = &[_][]const u8{ "machine learning", "neural", "model", "training", "ai", "llm", "tensor" }, .domain = .ai_ml },
    .{ .keywords = &[_][]const u8{ "algorithm", "sort", "search", "tree", "graph", "complexity", "big-o" }, .domain = .algorithms },
    .{ .keywords = &[_][]const u8{ "test", "unit test", "integration", "mock", "assert", "coverage" }, .domain = .testing },
};

/// Configuration for the classifier.
pub const ClassifierConfig = struct {
    /// Minimum confidence to report a classification.
    min_confidence: f32 = 0.5,
    /// Whether to detect language.
    detect_language: bool = true,
    /// Whether to detect domain.
    detect_domain: bool = true,
    /// Whether to estimate complexity.
    estimate_complexity: bool = true,
};

/// Query classifier for Aviva.
pub const QueryClassifier = struct {
    config: ClassifierConfig,

    const Self = @This();

    /// Initialize the classifier.
    pub fn init() Self {
        return initWithConfig(.{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(config: ClassifierConfig) Self {
        return .{ .config = config };
    }

    /// Classify a query.
    pub fn classify(self: *const Self, query: []const u8) ClassificationResult {
        // Convert to lowercase for matching
        var lower_buf: [4096]u8 = undefined;
        const query_lower = self.toLowerBounded(query, &lower_buf);

        // Detect query type
        var type_scores = [_]f32{0.0} ** 8;
        var key_indicators: [5]?[]const u8 = [_]?[]const u8{null} ** 5;
        var indicator_idx: usize = 0;

        for (QUERY_PATTERNS) |pattern| {
            for (pattern.keywords) |keyword| {
                if (std.mem.indexOf(u8, query_lower, keyword) != null) {
                    const type_idx = @intFromEnum(pattern.query_type);
                    type_scores[type_idx] = @max(type_scores[type_idx], pattern.weight);
                    if (indicator_idx < 5) {
                        key_indicators[indicator_idx] = keyword;
                        indicator_idx += 1;
                    }
                }
            }
        }

        // Find primary and secondary types
        var primary_type: QueryType = .general;
        var primary_score: f32 = 0.0;
        var secondary: [2]?QueryType = [_]?QueryType{null} ** 2;

        // Find top types
        var top_indices: [3]usize = [_]usize{7} ** 3; // Default to general
        var top_scores: [3]f32 = [_]f32{0.0} ** 3;

        for (type_scores, 0..) |score, idx| {
            if (score > top_scores[2]) {
                // Insert in sorted position
                var insert_pos: usize = 2;
                while (insert_pos > 0 and score > top_scores[insert_pos - 1]) : (insert_pos -= 1) {}

                // Shift down
                var j: usize = 2;
                while (j > insert_pos) : (j -= 1) {
                    top_scores[j] = top_scores[j - 1];
                    top_indices[j] = top_indices[j - 1];
                }
                top_scores[insert_pos] = score;
                top_indices[insert_pos] = idx;
            }
        }

        if (top_scores[0] >= self.config.min_confidence) {
            primary_type = @enumFromInt(top_indices[0]);
            primary_score = top_scores[0];
        }

        // Set secondary types
        for (0..2) |i| {
            if (top_scores[i + 1] >= self.config.min_confidence) {
                secondary[i] = @enumFromInt(top_indices[i + 1]);
            }
        }

        // Detect language
        var language: Language = .unknown;
        if (self.config.detect_language) {
            language = self.detectLanguage(query_lower);
        }

        // Detect domain
        var domain: Domain = .general;
        if (self.config.detect_domain) {
            domain = self.detectDomain(query_lower);
        }

        // Estimate complexity
        var complexity: Complexity = .moderate;
        if (self.config.estimate_complexity) {
            complexity = self.estimateComplexity(query, primary_type);
        }

        return .{
            .query_type = primary_type,
            .confidence = primary_score,
            .language = language,
            .domain = domain,
            .complexity = complexity,
            .secondary_types = secondary,
            .key_indicators = key_indicators,
        };
    }

    /// Detect programming language in query.
    fn detectLanguage(self: *const Self, query: []const u8) Language {
        _ = self;
        for (LANGUAGE_PATTERNS) |pattern| {
            for (pattern.keywords) |keyword| {
                if (std.mem.indexOf(u8, query, keyword) != null) {
                    return pattern.language;
                }
            }
        }
        return .unknown;
    }

    /// Detect domain of expertise.
    fn detectDomain(self: *const Self, query: []const u8) Domain {
        _ = self;
        for (DOMAIN_PATTERNS) |pattern| {
            for (pattern.keywords) |keyword| {
                if (std.mem.indexOf(u8, query, keyword) != null) {
                    return pattern.domain;
                }
            }
        }
        return .general;
    }

    /// Estimate query complexity.
    fn estimateComplexity(self: *const Self, query: []const u8, query_type: QueryType) Complexity {
        _ = self;
        const len = query.len;

        // Base complexity on query length
        var base: Complexity = if (len < 50) .trivial else if (len < 150) .simple else if (len < 400) .moderate else if (len < 800) .complex else .expert;

        // Adjust based on query type
        switch (query_type) {
            .debugging, .optimization => {
                // These tend to be more complex
                if (@intFromEnum(base) < @intFromEnum(Complexity.moderate)) {
                    base = .moderate;
                }
            },
            .factual_query, .documentation => {
                // These tend to be simpler
                if (@intFromEnum(base) > @intFromEnum(Complexity.simple)) {
                    base = @enumFromInt(@max(1, @intFromEnum(base) - 1));
                }
            },
            else => {},
        }

        return base;
    }

    /// Convert to lowercase with bounded buffer.
    fn toLowerBounded(self: *const Self, text: []const u8, buf: []u8) []const u8 {
        _ = self;
        const len = @min(text.len, buf.len);
        for (text[0..len], 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        return buf[0..len];
    }
};

// Tests

test "classifier initialization" {
    const classifier = QueryClassifier.init();
    try std.testing.expectEqual(@as(f32, 0.5), classifier.config.min_confidence);
}

test "classify code request" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("Write a function to sort an array in Zig");

    try std.testing.expectEqual(QueryType.code_request, result.query_type);
    try std.testing.expectEqual(Language.zig, result.language);
}

test "classify debugging" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("Fix this error: segmentation fault in my code");

    try std.testing.expectEqual(QueryType.debugging, result.query_type);
}

test "classify factual query" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("What is the difference between a mutex and a semaphore?");

    try std.testing.expect(result.query_type == .factual_query or result.query_type == .comparison);
}

test "classify explanation" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("Explain how garbage collection works");

    try std.testing.expectEqual(QueryType.explanation, result.query_type);
}

test "detect python language" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("Write a Python script to parse JSON");

    try std.testing.expectEqual(Language.python, result.language);
}

test "detect web domain" {
    const classifier = QueryClassifier.init();
    const result = classifier.classify("Create a React component for user login");

    try std.testing.expectEqual(Domain.web_development, result.domain);
}

test "complexity estimation" {
    const classifier = QueryClassifier.init();

    const simple = classifier.classify("What is a pointer?");
    const complex_query = classifier.classify("Explain the complete lifecycle of a Kubernetes pod including initialization, container startup, readiness probes, liveness probes, and termination with graceful shutdown handling and preStop hooks");

    try std.testing.expect(@intFromEnum(simple.complexity) < @intFromEnum(complex_query.complexity));
}

test "recommends code block" {
    try std.testing.expect(QueryType.code_request.recommendsCodeBlock());
    try std.testing.expect(QueryType.debugging.recommendsCodeBlock());
    try std.testing.expect(!QueryType.factual_query.recommendsCodeBlock());
}
