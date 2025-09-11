# Code API Index (Scanned)

Scanned 76 Zig files under `src/`. This index lists public declarations discovered along with leading doc comments.

## src\logging.zig

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum(u8) {
```

- fn `toString`

```zig
pub fn toString(self: LogLevel) []const u8 {
```

- fn `color`

```zig
pub fn color(self: LogLevel) []const u8 {
```

- type `OutputFormat`

Output format enumeration


```zig
pub const OutputFormat = enum {
```

- type `LoggerConfig`

Logger configuration


```zig
pub const LoggerConfig = struct {
```

- type `Logger`

Structured Logger


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) !*Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `log`

Log a message with structured fields


```zig
pub fn log(
```

- fn `trace`

Convenience methods for different log levels


```zig
pub fn trace(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `initGlobalLogger`

Initialize global logger


```zig
pub fn initGlobalLogger(allocator: std.mem.Allocator, config: LoggerConfig) !void {
```

- fn `deinitGlobalLogger`

Deinitialize global logger


```zig
pub fn deinitGlobalLogger() void {
```

- fn `getGlobalLogger`

Get global logger instance


```zig
pub fn getGlobalLogger() ?*Logger {
```

- fn `log`

Global logging functions


```zig
pub fn log(
```

- fn `trace`

```zig
pub fn trace(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `debug`

```zig
pub fn debug(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `info`

```zig
pub fn info(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `warn`

```zig
pub fn warn(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `err`

```zig
pub fn err(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `fatal`

```zig
pub fn fatal(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

## src\platform.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- type `PlatformInfo`

Platform capabilities and configuration


```zig
pub const PlatformInfo = struct {
```

- fn `detect`

```zig
pub fn detect() PlatformInfo {
```

- fn `initializePlatform`

Platform-specific initialization


```zig
pub fn initializePlatform() !void {
```

- type `FileOps`

Cross-platform file operations


```zig
pub const FileOps = struct {
```

- fn `openFile`

```zig
pub fn openFile(path: []const u8) !std.fs.File {
```

- fn `createFile`

```zig
pub fn createFile(path: []const u8) !std.fs.File {
```

- fn `deleteFile`

```zig
pub fn deleteFile(path: []const u8) !void {
```

- fn `fileExists`

```zig
pub fn fileExists(path: []const u8) bool {
```

- type `MemoryOps`

Cross-platform memory operations


```zig
pub const MemoryOps = struct {
```

- fn `getPageSize`

```zig
pub fn getPageSize() usize {
```

- fn `alignToPageSize`

```zig
pub fn alignToPageSize(size: usize) usize {
```

- fn `getVirtualMemoryLimit`

```zig
pub fn getVirtualMemoryLimit() usize {
```

- type `ThreadOps`

Cross-platform threading utilities


```zig
pub const ThreadOps = struct {
```

- fn `getOptimalThreadCount`

```zig
pub fn getOptimalThreadCount() u32 {
```

- fn `setThreadPriority`

```zig
pub fn setThreadPriority(thread: std.Thread, priority: ThreadPriority) !void {
```

- type `ThreadPriority`

```zig
pub const ThreadPriority = enum {
```

- type `PerfOps`

Cross-platform performance utilities


```zig
pub const PerfOps = struct {
```

- fn `getCpuFrequency`

```zig
pub fn getCpuFrequency() u64 {
```

- fn `getCacheInfo`

```zig
pub fn getCacheInfo() CacheInfo {
```

- type `CacheInfo`

```zig
pub const CacheInfo = struct {
```

- type `Colors`

ANSI color support


```zig
pub const Colors = struct {
```

- const `reset`

```zig
pub const reset = "\x1b[0m";
```

- const `bold`

```zig
pub const bold = "\x1b[1m";
```

- const `red`

```zig
pub const red = "\x1b[31m";
```

- const `green`

```zig
pub const green = "\x1b[32m";
```

- const `yellow`

```zig
pub const yellow = "\x1b[33m";
```

- const `blue`

```zig
pub const blue = "\x1b[34m";
```

- const `magenta`

```zig
pub const magenta = "\x1b[35m";
```

- const `cyan`

```zig
pub const cyan = "\x1b[36m";
```

- const `white`

```zig
pub const white = "\x1b[37m";
```

- fn `print`

```zig
pub fn print(comptime color: []const u8, comptime fmt: []const u8, args: anytype) void {
```

- const `PlatformError`

Platform-specific error handling


```zig
pub const PlatformError = error{
```

- fn `getTempDir`

Get platform-specific temporary directory


```zig
pub fn getTempDir(allocator: std.mem.Allocator) ![]const u8 {
```

- fn `sleep`

Platform-specific sleep function


```zig
pub fn sleep(milliseconds: u64) void {
```

- fn `getSystemInfo`

Get system information as a formatted string


```zig
pub fn getSystemInfo(allocator: std.mem.Allocator) ![]const u8 {
```

## src\root.zig

- const `database`

```zig
pub const database = @import("wdbx/database.zig");
```

- const `ai`

```zig
pub const ai = @import("ai/mod.zig");
```

- const `wdbx`

```zig
pub const wdbx = @import("wdbx/mod.zig");
```

- const `plugins`

```zig
pub const plugins = @import("plugins/mod.zig");
```

- const `monitoring`

```zig
pub const monitoring = @import("monitoring/mod.zig");
```

- const `logging`

```zig
pub const logging = @import("logging.zig");
```

- const `core`

```zig
pub const core = @import("core/mod.zig");
```

- const `gpu`

```zig
pub const gpu = @import("gpu/mod.zig");
```

- const `connectors`

```zig
pub const connectors = @import("connectors/mod.zig");
```

- const `services`

```zig
pub const services = @import("services/mod.zig");
```

- const `Db`

```zig
pub const Db = database.Db;
```

- const `DbError`

```zig
pub const DbError = database.DbError;
```

- const `Result`

```zig
pub const Result = database.Result;
```

- const `WdbxHeader`

```zig
pub const WdbxHeader = database.WdbxHeader;
```

- const `Vector`

```zig
pub const Vector = core.Vector;
```

- const `VectorOps`

```zig
pub const VectorOps = core.VectorOps;
```

- const `MatrixOps`

```zig
pub const MatrixOps = core.MatrixOps;
```

- const `NeuralNetwork`

```zig
pub const NeuralNetwork = ai.NeuralNetwork;
```

- const `EmbeddingGenerator`

```zig
pub const EmbeddingGenerator = ai.EmbeddingGenerator;
```

- const `ModelTrainer`

```zig
pub const ModelTrainer = ai.ModelTrainer;
```

- const `Layer`

```zig
pub const Layer = ai.Layer;
```

- const `Activation`

```zig
pub const Activation = ai.Activation;
```

- const `DynamicPersonaRouter`

```zig
pub const DynamicPersonaRouter = ai.DynamicPersonaRouter;
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- const `ArrayList`

```zig
pub const ArrayList = std.ArrayList;
```

- const `HashMap`

```zig
pub const HashMap = std.HashMap;
```

- const `Command`

```zig
pub const Command = wdbx.Command;
```

- const `WdbxOutputFormat`

```zig
pub const WdbxOutputFormat = wdbx.OutputFormat;
```

- const `WdbxLogLevel`

```zig
pub const WdbxLogLevel = wdbx.LogLevel;
```

- const `Logger`

```zig
pub const Logger = logging.Logger;
```

- const `LogLevel`

```zig
pub const LogLevel = logging.LogLevel;
```

- const `LogOutputFormat`

```zig
pub const LogOutputFormat = logging.OutputFormat;
```

- const `LoggerConfig`

```zig
pub const LoggerConfig = logging.LoggerConfig;
```

- const `GPURenderer`

```zig
pub const GPURenderer = gpu.GPURenderer;
```

- const `GpuBackend`

```zig
pub const GpuBackend = gpu.GpuBackend;
```

- const `GPUConfig`

```zig
pub const GPUConfig = gpu.GPUConfig;
```

- const `GpuError`

```zig
pub const GpuError = gpu.GpuError;
```

- const `Tracer`

```zig
pub const Tracer = monitoring.Tracer;
```

- const `TraceId`

```zig
pub const TraceId = monitoring.TraceId;
```

- const `Span`

```zig
pub const Span = monitoring.Span;
```

- const `SpanId`

```zig
pub const SpanId = monitoring.SpanId;
```

- const `TraceContext`

```zig
pub const TraceContext = monitoring.TraceContext;
```

- const `TracingError`

```zig
pub const TracingError = monitoring.TracingError;
```

- const `MemoryTracker`

```zig
pub const MemoryTracker = monitoring.MemoryTracker;
```

- const `PerformanceProfiler`

```zig
pub const PerformanceProfiler = monitoring.PerformanceProfiler;
```

- const `WeatherService`

```zig
pub const WeatherService = services.WeatherService;
```

- const `WeatherData`

```zig
pub const WeatherData = services.WeatherData;
```

- const `WeatherConfig`

```zig
pub const WeatherConfig = services.WeatherConfig;
```

- fn `main`

Main application entry point


```zig
pub fn main() !void {
```

- fn `init`

Initialize the ABI system


```zig
pub fn init(allocator: std.mem.Allocator) !void {
```

- fn `deinit`

Cleanup the ABI system


```zig
pub fn deinit() void {
```

- fn `getSystemInfo`

Get system information


```zig
pub fn getSystemInfo() struct {
```

- fn `runSystemTest`

Run a quick system test


```zig
pub fn runSystemTest() !void {
```

## src\utils.zig

- const `VERSION`

Project version information


```zig
pub const VERSION = .{
```

- type `Config`

Common configuration struct


```zig
pub const Config = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8) Config {
```

- type `DefinitionType`

Definition types used throughout the project


```zig
pub const DefinitionType = enum {
```

- fn `toString`

```zig
pub fn toString(self: DefinitionType) []const u8 {
```

- type `HttpStatus`

HTTP status codes


```zig
pub const HttpStatus = enum(u16) {
```

- fn `phrase`

```zig
pub fn phrase(self: HttpStatus) []const u8 {
```

- type `HttpMethod`

HTTP method types


```zig
pub const HttpMethod = enum {
```

- fn `fromString`

```zig
pub fn fromString(method: []const u8) ?HttpMethod {
```

- fn `toString`

```zig
pub fn toString(self: HttpMethod) []const u8 {
```

- type `Headers`

HTTP header management


```zig
pub const Headers = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Headers {
```

- fn `deinit`

```zig
pub fn deinit(self: *Headers) void {
```

- fn `set`

```zig
pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
```

- fn `get`

```zig
pub fn get(self: *Headers, name: []const u8) ?[]const u8 {
```

- fn `remove`

```zig
pub fn remove(self: *Headers, name: []const u8) bool {
```

- type `HttpRequest`

HTTP request structure


```zig
pub const HttpRequest = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, method: HttpMethod, path: []const u8) HttpRequest {
```

- fn `deinit`

```zig
pub fn deinit(self: *HttpRequest) void {
```

- type `HttpResponse`

HTTP response structure


```zig
pub const HttpResponse = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, status: HttpStatus) HttpResponse {
```

- fn `deinit`

```zig
pub fn deinit(self: *HttpResponse) void {
```

- fn `setContentType`

```zig
pub fn setContentType(self: *HttpResponse, content_type: []const u8) !void {
```

- fn `setJson`

```zig
pub fn setJson(self: *HttpResponse) !void {
```

- fn `setText`

```zig
pub fn setText(self: *HttpResponse) !void {
```

- fn `setHtml`

```zig
pub fn setHtml(self: *HttpResponse) !void {
```

- type `StringUtils`

String utilities


```zig
pub const StringUtils = struct {
```

- fn `isEmptyOrWhitespace`

Check if string is empty or whitespace only


```zig
pub fn isEmptyOrWhitespace(str: []const u8) bool {
```

- fn `toLower`

Convert string to lowercase (allocates)


```zig
pub fn toLower(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
```

- fn `toUpper`

Convert string to uppercase (allocates)


```zig
pub fn toUpper(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
```

- type `ArrayUtils`

Array utilities


```zig
pub const ArrayUtils = struct {
```

- fn `contains`

Check if array contains element


```zig
pub fn contains(comptime T: type, haystack: []const T, needle: T) bool {
```

- fn `indexOf`

Find index of element in array


```zig
pub fn indexOf(comptime T: type, haystack: []const T, needle: T) ?usize {
```

- type `TimeUtils`

Time utilities


```zig
pub const TimeUtils = struct {
```

- fn `nowMs`

Get current timestamp in milliseconds


```zig
pub fn nowMs() i64 {
```

- fn `nowUs`

Get current timestamp in microseconds


```zig
pub fn nowUs() i64 {
```

- fn `nowNs`

Get current timestamp in nanoseconds


```zig
pub fn nowNs() i64 {
```

- fn `formatDuration`

Format duration in human readable format


```zig
pub fn formatDuration(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
```

## src\wdbx\cli.zig

- const `Db`

Re-export database types for convenience


```zig
pub const Db = database.Db;
```

- const `DbError`

```zig
pub const DbError = database.Db.DbError;
```

- const `Result`

```zig
pub const Result = database.Db.Result;
```

- const `WdbxHeader`

```zig
pub const WdbxHeader = database.WdbxHeader;
```

- type `Command`

WDBX CLI command types


```zig
pub const Command = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?Command {
```

- fn `getDescription`

```zig
pub fn getDescription(self: Command) []const u8 {
```

- type `OutputFormat`

Output format options


```zig
pub const OutputFormat = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?OutputFormat {
```

- fn `getExtension`

```zig
pub fn getExtension(self: OutputFormat) []const u8 {
```

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?LogLevel {
```

- fn `toInt`

```zig
pub fn toInt(self: LogLevel) u8 {
```

- type `Options`

CLI options structure


```zig
pub const Options = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *Options, allocator: std.mem.Allocator) void {
```

- type `WdbxCLI`

WDBX CLI implementation


```zig
pub const WdbxCLI = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: Options) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `run`

```zig
pub fn run(self: *Self) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: LogLevel) Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `trace`

```zig
pub fn trace(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `main`

Main entry point


```zig
pub fn main() !void {
```

## src\wdbx\config.zig

- const `ConfigValidationError`

Configuration validation error codes


```zig
pub const ConfigValidationError = error{
```

- type `ConfigValidator`

Configuration schema validator


```zig
pub const ConfigValidator = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `validateConfig`

Validate entire configuration with comprehensive checks


```zig
pub fn validateConfig(self: *Self, config: *const WdbxConfig) ConfigValidationError!void {
```

- fn `generateValidationReport`

Generate validation report


```zig
pub fn generateValidationReport(self: *Self, config: *const WdbxConfig) ![]const u8 {
```

- type `WdbxConfig`

Configuration file format (JSON-based)


```zig
pub const WdbxConfig = struct {
```

- type `DatabaseConfig`

```zig
pub const DatabaseConfig = struct {
```

- type `ServerConfig`

```zig
pub const ServerConfig = struct {
```

- type `PerformanceConfig`

```zig
pub const PerformanceConfig = struct {
```

- type `MonitoringConfig`

```zig
pub const MonitoringConfig = struct {
```

- type `SecurityConfig`

```zig
pub const SecurityConfig = struct {
```

- type `LoggingConfig`

```zig
pub const LoggingConfig = struct {
```

- type `ConfigManager`

Configuration manager


```zig
pub const ConfigManager = struct {
```

- const `DEFAULT_CONFIG_FILE`

Default configuration file name


```zig
pub const DEFAULT_CONFIG_FILE = ".wdbx-config";
```

- fn `init`

Initialize configuration manager


```zig
pub fn init(allocator: std.mem.Allocator, config_path: ?[]const u8) !*Self {
```

- fn `deinit`

Deinitialize configuration manager


```zig
pub fn deinit(self: *Self) void {
```

- fn `getConfig`

Get current configuration


```zig
pub fn getConfig(self: *Self) *const WdbxConfig {
```

- fn `loadFromFile`

Load configuration from file


```zig
pub fn loadFromFile(self: *Self) !void {
```

- fn `createDefaultConfigFile`

Create default configuration file (Zig 0.16 compatible)


```zig
pub fn createDefaultConfigFile(self: *Self) !void {
```

- fn `applyEnvironmentOverrides`

Apply environment variable overrides


```zig
pub fn applyEnvironmentOverrides(self: *Self) !void {
```

- fn `checkAndReload`

Check if configuration file has been modified and reload if necessary


```zig
pub fn checkAndReload(self: *Self) !bool {
```

- fn `validate`

Validate configuration values using comprehensive schema validation


```zig
pub fn validate(self: *Self) !void {
```

- fn `save`

Save current configuration to file


```zig
pub fn save(self: *Self) !void {
```

- type `ConfigUtils`

Configuration utilities


```zig
pub const ConfigUtils = struct {
```

- fn `getValue`

Get configuration value by path (e.g., "database.hnsw_m")


```zig
pub fn getValue(config: *const WdbxConfig, path: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
```

- fn `printSummary`

Print configuration summary


```zig
pub fn printSummary(config: *const WdbxConfig) void {
```

## src\wdbx\core.zig

- const `WdbxError`

WDBX standardized error codes with numeric IDs for consistent error handling


```zig
pub const WdbxError = error{
```

- type `ErrorCodes`

Error code mapping for consistent error handling across modules


```zig
pub const ErrorCodes = struct {
```

- fn `getErrorCode`

```zig
pub fn getErrorCode(err: WdbxError) u32 {
```

- fn `getErrorDescription`

```zig
pub fn getErrorDescription(err: WdbxError) []const u8 {
```

- fn `getErrorCategory`

```zig
pub fn getErrorCategory(err: WdbxError) []const u8 {
```

- fn `formatError`

Format error for logging and display


```zig
pub fn formatError(allocator: std.mem.Allocator, err: WdbxError, context: ?[]const u8) ![]const u8 {
```

- type `VERSION`

WDBX version information


```zig
pub const VERSION = struct {
```

- const `MAJOR`

```zig
pub const MAJOR = 1;
```

- const `MINOR`

```zig
pub const MINOR = 0;
```

- const `PATCH`

```zig
pub const PATCH = 0;
```

- const `PRE_RELEASE`

```zig
pub const PRE_RELEASE = "alpha";
```

- fn `string`

```zig
pub fn string() []const u8 {
```

- fn `isCompatible`

```zig
pub fn isCompatible(major: u32, minor: u32) bool {
```

- type `OutputFormat`

Output format options


```zig
pub const OutputFormat = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?OutputFormat {
```

- fn `toString`

```zig
pub fn toString(self: OutputFormat) []const u8 {
```

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum(u8) {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?LogLevel {
```

- fn `toString`

```zig
pub fn toString(self: LogLevel) []const u8 {
```

- fn `toInt`

```zig
pub fn toInt(self: LogLevel) u8 {
```

- type `Config`

Common WDBX configuration


```zig
pub const Config = struct {
```

- fn `init`

```zig
pub fn init() Config {
```

- fn `validate`

```zig
pub fn validate(self: *const Config) WdbxError!void {
```

- type `Timer`

Performance timer for benchmarking


```zig
pub const Timer = struct {
```

- fn `init`

```zig
pub fn init() Timer {
```

- fn `elapsed`

```zig
pub fn elapsed(self: *const Timer) u64 {
```

- fn `elapsedMs`

```zig
pub fn elapsedMs(self: *const Timer) f64 {
```

- fn `elapsedUs`

```zig
pub fn elapsedUs(self: *const Timer) f64 {
```

- fn `restart`

```zig
pub fn restart(self: *Timer) void {
```

- type `Logger`

Simple logging utility


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: LogLevel) Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `log`

```zig
pub fn log(self: *Logger, level: LogLevel, comptime fmt: []const u8, args: anytype) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- type `MemoryStats`

Memory usage statistics


```zig
pub const MemoryStats = struct {
```

- fn `init`

```zig
pub fn init() MemoryStats {
```

- fn `allocate`

```zig
pub fn allocate(self: *MemoryStats, size: usize) void {
```

- fn `deallocate`

```zig
pub fn deallocate(self: *MemoryStats, size: usize) void {
```

- fn `reset`

```zig
pub fn reset(self: *MemoryStats) void {
```

## src\wdbx\database.zig

- const `DatabaseError`

Database-specific error types


```zig
pub const DatabaseError = error{
```

- const `MAGIC`

Magic identifier for ABI files (7 bytes + NUL)


```zig
pub const MAGIC: [8]u8 = "WDBXAI\x00\x00".*;
```

- const `FORMAT_VERSION`

Current file format version


```zig
pub const FORMAT_VERSION: u16 = 1;
```

- const `DEFAULT_PAGE_SIZE`

Default page size for file operations (4 KiB)


```zig
pub const DEFAULT_PAGE_SIZE: u32 = 4096;
```

- type `WdbxHeader`

File-header fixed at 4 KiB (4096 bytes)


```zig
pub const WdbxHeader = struct {
```

- fn `validateMagic`

```zig
pub fn validateMagic(self: *const WdbxHeader) bool {
```

- fn `createDefault`

```zig
pub fn createDefault() WdbxHeader {
```

- type `Db`

```zig
pub const Db = struct {
```

- const `DbError`

```zig
pub const DbError = error{
```

- fn `isInitialized`

```zig
pub fn isInitialized(self: *const Db) bool {
```

- fn `init`

```zig
pub fn init(self: *Db, dim: u16) DbError!void {
```

- fn `addEmbedding`

```zig
pub fn addEmbedding(self: *Db, embedding: []const f32) DbError!u64 {
```

- fn `addEmbeddingsBatch`

```zig
pub fn addEmbeddingsBatch(self: *Db, embeddings: []const []const f32) DbError![]u64 {
```

- fn `search`

```zig
pub fn search(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) DbError![]Result {
```

- type `Result`

```zig
pub const Result = struct {
```

- fn `lessThanAsc`

```zig
pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
```

- type `DbStats`

```zig
pub const DbStats = struct {
```

- fn `getAverageSearchTime`

```zig
pub fn getAverageSearchTime(self: *const DbStats) u64 {
```

- fn `open`

```zig
pub fn open(path: []const u8, create_if_missing: bool) DbError!*Db {
```

- fn `close`

```zig
pub fn close(self: *Db) void {
```

- fn `getStats`

```zig
pub fn getStats(self: *const Db) DbStats {
```

- fn `getRowCount`

```zig
pub fn getRowCount(self: *const Db) u64 {
```

- fn `getDimension`

```zig
pub fn getDimension(self: *const Db) u16 {
```

- type `HNSWIndex`

HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search


```zig
pub const HNSWIndex = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, id: u64, vector: []const f32, layer: u32) !*Node {
```

- fn `deinit`

```zig
pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
```

- fn `addConnection`

```zig
pub fn addConnection(self: *Node, allocator: std.mem.Allocator, node_id: u64) !void {
```

- fn `compare`

```zig
pub fn compare(_: void, a: SearchResult, b: SearchResult) std.math.Order {
```

- fn `compare`

```zig
pub fn compare(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, dimension: u16) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addVector`

Add a vector to the HNSW index


```zig
pub fn addVector(self: *Self, id: u64, vector: []const f32) !void {
```

- fn `search`

Search for approximate nearest neighbors


```zig
pub fn search(self: *Self, query: []const f32, top_k: usize) ![]SearchResult {
```

- fn `initHNSW`

Initialize HNSW index for faster search


```zig
pub fn initHNSW(self: *Db) !void {
```

- fn `addToHNSW`

Add vector to HNSW index


```zig
pub fn addToHNSW(self: *Db, id: u64, vector: []const f32) !void {
```

- fn `setHNSWParams`

Adjust HNSW parameters (useful for tests/benchmarks)


```zig
pub fn setHNSWParams(self: *Db, params: struct { max_connections: u32 = 16, ef_construction: u32 = 200, ef_search: u32 = 100 }) void {
```

- fn `searchHNSW`

Search using HNSW index (fallback to brute force if not available)


```zig
pub fn searchHNSW(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) ![]Result {
```

- fn `searchParallel`

Parallel search using multiple threads for brute force search


```zig
pub fn searchParallel(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator, num_threads: u32) ![]Result {
```

## src\wdbx\mod.zig

- const `cli`

```zig
pub const cli = @import("cli.zig");
```

- const `http`

```zig
pub const http = @import("../server/wdbx_http.zig");
```

- const `core`

```zig
pub const core = @import("core.zig");
```

- const `config`

```zig
pub const config = @import("config.zig");
```

- const `WdbxCLI`

```zig
pub const WdbxCLI = cli.WdbxCLI;
```

- const `WdbxHttpServer`

```zig
pub const WdbxHttpServer = http.WdbxHttpServer;
```

- const `Command`

```zig
pub const Command = cli.Command;
```

- const `Options`

```zig
pub const Options = cli.Options;
```

- const `ServerConfig`

```zig
pub const ServerConfig = http.ServerConfig;
```

- const `OutputFormat`

```zig
pub const OutputFormat = cli.OutputFormat;
```

- const `LogLevel`

```zig
pub const LogLevel = cli.LogLevel;
```

- const `WdbxConfig`

```zig
pub const WdbxConfig = config.WdbxConfig;
```

- const `ConfigManager`

```zig
pub const ConfigManager = config.ConfigManager;
```

- const `ConfigUtils`

```zig
pub const ConfigUtils = config.ConfigUtils;
```

- const `database`

```zig
pub const database = @import("database.zig");
```

- const `main`

```zig
pub const main = cli.main;
```

- const `createServer`

```zig
pub const createServer = http.createServer;
```

## src\services\mod.zig

- const `WeatherService`

Represents the weather service interface for interacting with OpenWeatherMap API.
Provides methods for retrieving current weather data and forecasts.


```zig
pub const WeatherService = weather.WeatherService;
```

- const `WeatherData`

Contains structured weather data including temperature, humidity, wind, and location information.
All string fields are allocated and must be freed by the caller.


```zig
pub const WeatherData = weather.WeatherData;
```

- const `WeatherConfig`

Configuration settings for the weather service including API credentials,
timeout settings, and response size limits.


```zig
pub const WeatherConfig = weather.WeatherConfig;
```

- const `WeatherError`

Error types related to weather service operations including network errors,
API errors, and data parsing failures.


```zig
pub const WeatherError = weather.WeatherError;
```

- const `Allocator`

Allocator for managing memory allocations within service modules.
Used for dynamic string allocation and HTTP response buffering.


```zig
pub const Allocator = std.mem.Allocator;
```

- fn `createWeatherService`

Initializes a weather service with the provided API key and default configuration.

This function creates a WeatherService instance with sensible defaults for production use.
The configuration can be further customized using WeatherConfig.fromEnv() for environment
variable overrides.

Parameters:
- `allocator`: Allocator for memory management during service operations
- `api_key`: API key for authenticating with the OpenWeatherMap service

Returns:
- A WeatherService instance ready for use

Errors:
- Returns an error if the service initialization fails due to invalid configuration


```zig
pub fn createWeatherService(allocator: Allocator, api_key: []const u8) !WeatherService {
```

## src\services\weather.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `WeatherError`

Weather-specific error types


```zig
pub const WeatherError = error{
```

- type `WeatherData`

```zig
pub const WeatherData = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *WeatherData, allocator: std.mem.Allocator) void {
```

- type `WeatherConfig`

```zig
pub const WeatherConfig = struct {
```

- fn `fromEnv`

```zig
pub fn fromEnv(allocator: std.mem.Allocator, base: WeatherConfig) WeatherConfig {
```

- type `WeatherService`

```zig
pub const WeatherService = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) !WeatherService {
```

- fn `deinit`

```zig
pub fn deinit(self: *WeatherService) void {
```

- fn `getCurrentWeather`

```zig
pub fn getCurrentWeather(self: *WeatherService, city: []const u8) !WeatherData {
```

- type `WeatherUtils`

```zig
pub const WeatherUtils = struct {
```

- fn `kelvinToCelsius`

```zig
pub fn kelvinToCelsius(kelvin: f32) f32 {
```

- fn `testParseWeatherResponse`

```zig
pub fn testParseWeatherResponse(self: *WeatherService, json_str: []const u8) !WeatherData {
```

- fn `testParseForecastResponse`

```zig
pub fn testParseForecastResponse(self: *WeatherService, json_str: []const u8) ![]WeatherData {
```

- fn `celsiusToFahrenheit`

```zig
pub fn celsiusToFahrenheit(celsius: f32) f32 {
```

- fn `fahrenheitToCelsius`

```zig
pub fn fahrenheitToCelsius(fahrenheit: f32) f32 {
```

- fn `getWindDirection`

```zig
pub fn getWindDirection(degrees: u16) []const u8 {
```

- fn `formatWeatherJson`

```zig
pub fn formatWeatherJson(weather: WeatherData, allocator: std.mem.Allocator) ![]u8 {
```

- fn `getWeatherEmoji`

```zig
pub fn getWeatherEmoji(icon: []const u8) []const u8 {
```

## src\server\enhanced_web_server.zig

- type `EnhancedWebServer`

Enhanced web server with production-ready features


```zig
pub const EnhancedWebServer = struct {
```

- fn `init`

Initialize the enhanced web server


```zig
pub fn init(allocator: std.mem.Allocator, server_config: WebServerConfig) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the enhanced web server


```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

Start the web server


```zig
pub fn start(self: *Self) FrameworkError!void {
```

- fn `stop`

Stop the web server


```zig
pub fn stop(self: *Self) void {
```

- fn `addRoute`

Add a route to the server


```zig
pub fn addRoute(self: *Self, route: Route) FrameworkError!void {
```

- fn `addMiddleware`

Add middleware to the server


```zig
pub fn addMiddleware(self: *Self, middleware: Middleware) FrameworkError!void {
```

- fn `getStats`

Get server statistics


```zig
pub fn getStats(self: *const Self) ServerStats {
```

- fn `healthCheck`

Health check for the server


```zig
pub fn healthCheck(self: *const Self) ServerHealthStatus {
```

- type `ServerState`

Server state management


```zig
pub const ServerState = enum {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: ServerState, new_state: ServerState) bool {
```

- type `HttpMethod`

HTTP methods


```zig
pub const HttpMethod = enum {
```

- type `Route`

Route definition


```zig
pub const Route = struct {
```

- const `RouteHandler`

Route handler function type


```zig
pub const RouteHandler = *const fn (request: *Request, response: *Response) anyerror!void;
```

- type `Middleware`

Middleware definition


```zig
pub const Middleware = struct {
```

- const `MiddlewareHandler`

Middleware handler function type


```zig
pub const MiddlewareHandler = *const fn (request: *Request, response: *Response, next: *const fn (*Request, *Response) anyerror!void) anyerror!void;
```

- type `RateLimit`

Rate limiting configuration


```zig
pub const RateLimit = struct {
```

- type `Request`

HTTP request structure


```zig
pub const Request = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Request {
```

- fn `deinit`

```zig
pub fn deinit(self: *Request) void {
```

- fn `getHeader`

```zig
pub fn getHeader(self: *const Request, name: []const u8) ?[]const u8 {
```

- fn `getQueryParam`

```zig
pub fn getQueryParam(self: *const Request, name: []const u8) ?[]const u8 {
```

- fn `hasHeader`

```zig
pub fn hasHeader(self: *const Request, name: []const u8) bool {
```

- fn `hasQueryParam`

```zig
pub fn hasQueryParam(self: *const Request, name: []const u8) bool {
```

- type `Response`

HTTP response structure


```zig
pub const Response = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Response {
```

- fn `deinit`

```zig
pub fn deinit(self: *Response) void {
```

- fn `setHeader`

```zig
pub fn setHeader(self: *Response, name: []const u8, value: []const u8) !void {
```

- fn `getHeader`

```zig
pub fn getHeader(self: *const Response, name: []const u8) ?[]const u8 {
```

- fn `setContentType`

```zig
pub fn setContentType(self: *Response, content_type: []const u8) void {
```

- fn `setBody`

```zig
pub fn setBody(self: *Response, body: []const u8) void {
```

- fn `send`

```zig
pub fn send(self: *Response, status: u16, body: []const u8) !void {
```

- fn `sendJson`

```zig
pub fn sendJson(self: *Response, status: u16, data: anytype) !void {
```

- fn `sendError`

```zig
pub fn sendError(self: *Response, status: u16, message: []const u8) !void {
```

- type `ErrorResponse`

Error response structure


```zig
pub const ErrorResponse = struct {
```

- type `ServerStats`

Server statistics


```zig
pub const ServerStats = struct {
```

- type `ServerHealthStatus`

Server health status


```zig
pub const ServerHealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *ServerHealthStatus) void {
```

- type `ComponentHealth`

Component health status


```zig
pub const ComponentHealth = struct {
```

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WebServerConfig) !*HttpServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *HttpServer) void {
```

- fn `start`

```zig
pub fn start(self: *HttpServer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *HttpServer) void {
```

- fn `getActiveConnections`

```zig
pub fn getActiveConnections(self: *const HttpServer) u32 {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const HttpServer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WebServerConfig) !*WebSocketServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *WebSocketServer) void {
```

- fn `start`

```zig
pub fn start(self: *WebSocketServer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *WebSocketServer) void {
```

- fn `getActiveConnections`

```zig
pub fn getActiveConnections(self: *const WebSocketServer) u32 {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const WebSocketServer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RouteRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *RouteRegistry) void {
```

- fn `registerRoute`

```zig
pub fn registerRoute(self: *RouteRegistry, route: Route) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RequestPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *RequestPool) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ResponsePool {
```

- fn `deinit`

```zig
pub fn deinit(self: *ResponsePool) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*AgentRouter {
```

- fn `deinit`

```zig
pub fn deinit(self: *AgentRouter) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const AgentRouter) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*AuthManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *AuthManager) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *AuthManager) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const AuthManager) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RateLimiter {
```

- fn `deinit`

```zig
pub fn deinit(self: *RateLimiter) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *RateLimiter) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const RateLimiter) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityManager) void {
```

- fn `initializePolicies`

```zig
pub fn initializePolicies(self: *SecurityManager) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const SecurityManager) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *PerformanceMonitor) !void {
```

- fn `start`

```zig
pub fn start(self: *PerformanceMonitor) !void {
```

- fn `stop`

```zig
pub fn stop(self: *PerformanceMonitor) void {
```

- fn `getTotalRequests`

```zig
pub fn getTotalRequests(self: *const PerformanceMonitor) u64 {
```

- fn `getAverageResponseTime`

```zig
pub fn getAverageResponseTime(self: *const PerformanceMonitor) f64 {
```

- fn `getErrorRate`

```zig
pub fn getErrorRate(self: *const PerformanceMonitor) f32 {
```

- fn `getMemoryUsage`

```zig
pub fn getMemoryUsage(self: *const PerformanceMonitor) usize {
```

- fn `getCpuUsage`

```zig
pub fn getCpuUsage(self: *const PerformanceMonitor) f32 {
```

- fn `registerCallback`

```zig
pub fn registerCallback(self: *PerformanceMonitor, callback: anytype) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const PerformanceMonitor) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*LoadBalancer {
```

- fn `deinit`

```zig
pub fn deinit(self: *LoadBalancer) void {
```

- fn `start`

```zig
pub fn start(self: *LoadBalancer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *LoadBalancer) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const LoadBalancer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ClusterManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *ClusterManager) void {
```

- fn `start`

```zig
pub fn start(self: *ClusterManager) !void {
```

- fn `stop`

```zig
pub fn stop(self: *ClusterManager) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const ClusterManager) ComponentHealth {
```

## src\server\wdbx_http.zig

- type `ServerConfig`

HTTP server configuration


```zig
pub const ServerConfig = struct {
```

- type `WdbxHttpServer`

HTTP server for WDBX vector database


```zig
pub const WdbxHttpServer = struct {
```

- fn `init`

Initialize HTTP server


```zig
pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*Self {
```

- fn `deinit`

Deinitialize HTTP server


```zig
pub fn deinit(self: *Self) void {
```

- fn `run`

Run the HTTP server (alias for start)


```zig
pub fn run(self: *Self) !void {
```

- fn `openDatabase`

Open database connection


```zig
pub fn openDatabase(self: *Self, path: []const u8) !void {
```

- fn `start`

Start HTTP server


```zig
pub fn start(self: *Self) !void {
```

- fn `handleRequests`

Non-blocking request handler used by tests; returns error.Timeout on no activity


```zig
pub fn handleRequests(self: *Self) !void {
```

- fn `configureSocket`

Configure socket for Windows compatibility and optimal performance


```zig
pub fn configureSocket(self: *Self, socket_handle: std.posix.socket_t) !void {
```

- fn `testConnectivity`

Test basic connectivity - useful for diagnosing Windows networking issues


```zig
pub fn testConnectivity(self: *Self) !bool {
```

## src\server\web_server.zig

- const `Allocator`

Re-export commonly used types for convenience


```zig
pub const Allocator = std.mem.Allocator;
```

- type `WebConfig`

Configuration settings for the web server.

This structure contains all the configurable parameters that control
the server's behavior, including network settings, security options,
and feature toggles.


```zig
pub const WebConfig = struct {
```

- type `WebServer`

Main web server instance that handles HTTP/WebSocket connections.

The WebServer manages network connections, routes requests to appropriate handlers,
and integrates with the AI agent system for intelligent request processing.
It supports both traditional HTTP endpoints and real-time WebSocket communication.


```zig
pub const WebServer = struct {
```

- fn `init`

Initializes a new WebServer instance with the specified configuration.

This function sets up the server with the provided configuration and
initializes the integrated AI agent for intelligent request processing.

Parameters:
- `allocator`: Memory allocator for the server and its components
- `config`: Configuration settings controlling server behavior

Returns:
- A pointer to the initialized WebServer instance

Errors:
- Returns an error if memory allocation fails
- Returns an error if AI agent initialization fails


```zig
pub fn init(allocator: std.mem.Allocator, config: WebConfig) !*WebServer {
```

- fn `deinit`

Properly releases all resources held by the WebServer.

This function should be called when the server is no longer needed
to prevent memory leaks and properly close network connections.


```zig
pub fn deinit(self: *WebServer) void {
```

- fn `start`

Starts the web server and begins accepting connections.

This function binds to the configured address and port, then enters
an infinite loop accepting and handling incoming connections.
Each connection is processed synchronously in the current implementation.

Errors:
- Returns an error if the address cannot be parsed
- Returns an error if the server cannot bind to the specified port


```zig
pub fn start(self: *WebServer) !void {
```

- fn `parseWebSocketFrame`

Parses a WebSocket frame from raw bytes.

This function implements the WebSocket frame parsing algorithm
according to RFC 6455, handling variable-length payload lengths
and masking (though masking is typically only used by clients).

Parameters:
- `data`: Raw frame data received from the WebSocket connection

Returns:
- A parsed WebSocketFrame structure

Errors:
- Returns InvalidFrame if the frame data is malformed or incomplete


```zig
pub fn parseWebSocketFrame(_: *WebServer, data: []const u8) !WebSocketFrame {
```

- fn `startOnce`

Starts the server for a single connection cycle (testing utility).

This function is intended for testing scenarios where you need to
accept exactly one connection, handle it, and then stop the server.
It's useful for unit tests and development scenarios.

Errors:
- Returns an error if the server cannot start
- Returns an error if connection handling fails


```zig
pub fn startOnce(self: *WebServer) !void {
```

- fn `handlePathForTest`

Test helper function for routing requests by path.

This function provides a simple way to test routing logic without
requiring actual network connections. It returns the response body
that would be sent for a given path.

Parameters:
- `path`: The request path to route
- `allocator`: Memory allocator for the response

Returns:
- The response body as a string

Errors:
- Returns an error if memory allocation fails


```zig
pub fn handlePathForTest(self: *WebServer, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
```

## src\plugins\enhanced_plugin_system.zig

- type `EnhancedPluginSystem`

Enhanced plugin system with production-ready features


```zig
pub const EnhancedPluginSystem = struct {
```

- fn `init`

Initialize the enhanced plugin system


```zig
pub fn init(allocator: std.mem.Allocator) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the enhanced plugin system


```zig
pub fn deinit(self: *Self) void {
```

- fn `loadPlugin`

Load a plugin from file


```zig
pub fn loadPlugin(self: *Self, path: []const u8) FrameworkError!*Plugin {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
```

- fn `reloadPlugin`

Reload a plugin


```zig
pub fn reloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
```

- fn `getPlugin`

Get a plugin by name


```zig
pub fn getPlugin(self: *Self, name: []const u8) ?*Plugin {
```

- fn `listPlugins`

List all loaded plugins


```zig
pub fn listPlugins(self: *const Self) []const []const u8 {
```

- fn `getPluginStats`

Get plugin statistics


```zig
pub fn getPluginStats(self: *const Self) PluginStats {
```

- fn `healthCheck`

Health check for all plugins


```zig
pub fn healthCheck(self: *const Self) PluginHealthStatus {
```

- type `PluginSystemConfig`

Plugin system configuration


```zig
pub const PluginSystemConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: PluginSystemConfig) FrameworkError!void {
```

- type `Plugin`

Enhanced plugin with production-ready features


```zig
pub const Plugin = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8, version: []const u8, description: []const u8, path: []const u8) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *Self, allocator: std.mem.Allocator) FrameworkError!void {
```

- fn `start`

```zig
pub fn start(self: *Self) FrameworkError!void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) FrameworkError!void {
```

- fn `deinitialize`

```zig
pub fn deinitialize(self: *Self) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const Self) PluginHealth {
```

- fn `addService`

```zig
pub fn addService(self: *Self, service: PluginService) FrameworkError!void {
```

- fn `removeService`

```zig
pub fn removeService(self: *Self, service_name: []const u8) bool {
```

- fn `getService`

```zig
pub fn getService(self: *const Self, service_name: []const u8) ?PluginService {
```

- type `PluginState`

Plugin state management


```zig
pub const PluginState = enum {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
```

- type `PluginService`

Plugin service definition


```zig
pub const PluginService = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8, version: []const u8, description: []const u8, capabilities: PluginCapabilities, handler: *const fn ([]const u8) anyerror![]const u8) PluginService {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginService) void {
```

- type `PluginCapabilities`

Plugin capabilities


```zig
pub const PluginCapabilities = struct {
```

- type `PluginPerformanceMetrics`

Plugin performance metrics


```zig
pub const PluginPerformanceMetrics = struct {
```

- fn `updateResponseTime`

```zig
pub fn updateResponseTime(self: *PluginPerformanceMetrics, response_time_ms: f64) void {
```

- fn `recordSuccess`

```zig
pub fn recordSuccess(self: *PluginPerformanceMetrics) void {
```

- fn `recordFailure`

```zig
pub fn recordFailure(self: *PluginPerformanceMetrics) void {
```

- fn `getSuccessRate`

```zig
pub fn getSuccessRate(self: *const PluginPerformanceMetrics) f32 {
```

- type `PluginHealth`

Plugin health status


```zig
pub const PluginHealth = struct {
```

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- type `PluginStats`

Plugin statistics


```zig
pub const PluginStats = struct {
```

- type `PluginHealthStatus`

Plugin health status for all plugins


```zig
pub const PluginHealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginHealthStatus) void {
```

- type `PluginEventType`

Plugin event types


```zig
pub const PluginEventType = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *PluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*NativePluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *NativePluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *NativePluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ScriptPluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *ScriptPluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *ScriptPluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*WebPluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *WebPluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *WebPluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginRegistry) void {
```

- fn `registerPlugin`

```zig
pub fn registerPlugin(self: *PluginRegistry, plugin: *Plugin) !void {
```

- fn `unregisterPlugin`

```zig
pub fn unregisterPlugin(self: *PluginRegistry, name: []const u8) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginWatcher {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginWatcher) void {
```

- fn `start`

```zig
pub fn start(self: *PluginWatcher, directory: []const u8) !void {
```

- fn `stop`

```zig
pub fn stop(self: *PluginWatcher) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginManager) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ServiceDiscovery {
```

- fn `deinit`

```zig
pub fn deinit(self: *ServiceDiscovery) void {
```

- fn `registerService`

```zig
pub fn registerService(self: *ServiceDiscovery, service: PluginService) !void {
```

- fn `unregisterService`

```zig
pub fn unregisterService(self: *ServiceDiscovery, name: []const u8) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityManager) void {
```

- fn `validatePluginPath`

```zig
pub fn validatePluginPath(self: *SecurityManager, path: []const u8) !void {
```

- fn `validatePlugin`

```zig
pub fn validatePlugin(self: *SecurityManager, plugin: *Plugin) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityContext {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityContext) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `validatePlugin`

```zig
pub fn validatePlugin(self: *PerformanceMonitor, plugin: *Plugin) !void {
```

## src\plugins\interface.zig

- type `PluginInterface`

Standard plugin interface using C-compatible function pointers
This vtable approach ensures compatibility across different compilation units


```zig
pub const PluginInterface = extern struct {
```

- fn `isValid`

```zig
pub fn isValid(self: *const PluginInterface) bool {
```

- type `Plugin`

Plugin wrapper that provides a safer Zig API around the C interface


```zig
pub const Plugin = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, interface: *const PluginInterface) !Plugin {
```

- fn `deinit`

```zig
pub fn deinit(self: *Plugin) void {
```

- fn `getInfo`

```zig
pub fn getInfo(self: *Plugin) *const PluginInfo {
```

- fn `initialize`

```zig
pub fn initialize(self: *Plugin, config: *PluginConfig) !void {
```

- fn `start`

```zig
pub fn start(self: *Plugin) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Plugin) !void {
```

- fn `pause`

```zig
pub fn pause(self: *Plugin) !void {
```

- fn `resumePlugin`

```zig
pub fn resumePlugin(self: *Plugin) !void {
```

- fn `process`

```zig
pub fn process(self: *Plugin, input: ?*anyopaque, output: ?*anyopaque) !void {
```

- fn `configure`

```zig
pub fn configure(self: *Plugin, config: *const PluginConfig) !void {
```

- fn `getStatus`

```zig
pub fn getStatus(self: *Plugin) i32 {
```

- fn `getMetrics`

```zig
pub fn getMetrics(self: *Plugin, buffer: []u8) !usize {
```

- fn `onEvent`

```zig
pub fn onEvent(self: *Plugin, event_type: u32, event_data: ?*anyopaque) !void {
```

- fn `getApi`

```zig
pub fn getApi(self: *Plugin, api_name: [:0]const u8) ?*anyopaque {
```

- fn `getState`

```zig
pub fn getState(self: *const Plugin) PluginState {
```

- fn `setState`

```zig
pub fn setState(self: *Plugin, new_state: PluginState) !void {
```

- const `PluginFactoryFn`

Plugin factory function type


```zig
pub const PluginFactoryFn = *const fn () callconv(.c) ?*const PluginInterface;
```

- const `PLUGIN_ENTRY_POINT`

Standard plugin entry point function name


```zig
pub const PLUGIN_ENTRY_POINT = "abi_plugin_create";
```

- const `PLUGIN_ABI_VERSION`

ABI version for plugin compatibility


```zig
pub const PLUGIN_ABI_VERSION = types.PluginVersion.init(1, 0, 0);
```

- fn `createPlugin`

Create a plugin from a loaded interface


```zig
pub fn createPlugin(allocator: std.mem.Allocator, interface: *const PluginInterface) !*Plugin {
```

- fn `destroyPlugin`

Destroy a plugin instance


```zig
pub fn destroyPlugin(allocator: std.mem.Allocator, plugin: *Plugin) void {
```

## src\plugins\loader.zig

- type `PluginLoader`

Cross-platform plugin loader


```zig
pub const PluginLoader = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) PluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginLoader) void {
```

- fn `addPluginPath`

Add a directory to search for plugins


```zig
pub fn addPluginPath(self: *PluginLoader, path: []const u8) !void {
```

- fn `removePluginPath`

Remove a plugin search path


```zig
pub fn removePluginPath(self: *PluginLoader, path: []const u8) void {
```

- fn `discoverPlugins`

Discover plugins in the search paths


```zig
pub fn discoverPlugins(self: *PluginLoader) !std.ArrayList([]const u8) {
```

- fn `loadPlugin`

Load a plugin from a file path


```zig
pub fn loadPlugin(self: *PluginLoader, plugin_path: []const u8) !*const PluginInterface {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *PluginLoader, plugin_path: []const u8) !void {
```

- fn `getLoadedPlugins`

Get the list of loaded plugins


```zig
pub fn getLoadedPlugins(self: *PluginLoader) []const LoadedLibrary {
```

- fn `makeLibraryName`

Construct a platform-specific library filename


```zig
pub fn makeLibraryName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
```

- fn `createLoader`

Create a plugin loader instance


```zig
pub fn createLoader(allocator: std.mem.Allocator) PluginLoader {
```

## src\plugins\mod.zig

- const `interface`

```zig
pub const interface = @import("interface.zig");
```

- const `loader`

```zig
pub const loader = @import("loader.zig");
```

- const `registry`

```zig
pub const registry = @import("registry.zig");
```

- const `types`

```zig
pub const types = @import("types.zig");
```

- const `Plugin`

```zig
pub const Plugin = interface.Plugin;
```

- const `PluginInterface`

```zig
pub const PluginInterface = interface.PluginInterface;
```

- const `PluginLoader`

```zig
pub const PluginLoader = loader.PluginLoader;
```

- const `PluginRegistry`

```zig
pub const PluginRegistry = registry.PluginRegistry;
```

- const `PluginError`

```zig
pub const PluginError = types.PluginError;
```

- const `PluginType`

```zig
pub const PluginType = types.PluginType;
```

- const `PluginInfo`

```zig
pub const PluginInfo = types.PluginInfo;
```

- const `PluginConfig`

```zig
pub const PluginConfig = types.PluginConfig;
```

- const `createLoader`

```zig
pub const createLoader = loader.createLoader;
```

- const `createRegistry`

```zig
pub const createRegistry = registry.createRegistry;
```

- const `registerBuiltinInterface`

```zig
pub const registerBuiltinInterface = registry.registerBuiltinInterface;
```

- fn `init`

Initialize the plugin system


```zig
pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
```

- type `VERSION`

Plugin system version


```zig
pub const VERSION = struct {
```

- const `MAJOR`

```zig
pub const MAJOR = 1;
```

- const `MINOR`

```zig
pub const MINOR = 0;
```

- const `PATCH`

```zig
pub const PATCH = 0;
```

- fn `string`

```zig
pub fn string() []const u8 {
```

- fn `isCompatible`

```zig
pub fn isCompatible(major: u32, minor: u32) bool {
```

## src\plugins\registry.zig

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, plugin: *Plugin, load_order: u32) PluginEntry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginEntry, allocator: std.mem.Allocator) void {
```

- type `PluginRegistry`

Centralized plugin registry


```zig
pub const PluginRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginRegistry) void {
```

- fn `addPluginPath`

Add a search path for plugins


```zig
pub fn addPluginPath(self: *PluginRegistry, path: []const u8) !void {
```

- fn `discoverPlugins`

Discover plugins in search paths


```zig
pub fn discoverPlugins(self: *PluginRegistry) !std.ArrayList([]const u8) {
```

- fn `loadPlugin`

Load a plugin from file


```zig
pub fn loadPlugin(self: *PluginRegistry, plugin_path: []const u8) !void {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `registerBuiltinInterface`

Register a plugin interface that is built into the process (no dynamic library)


```zig
pub fn registerBuiltinInterface(self: *PluginRegistry, plugin_interface: *const interface.PluginInterface) !void {
```

- fn `initializePlugin`

Initialize a plugin with configuration


```zig
pub fn initializePlugin(self: *PluginRegistry, plugin_name: []const u8, config: ?*PluginConfig) !void {
```

- fn `startPlugin`

Start a plugin


```zig
pub fn startPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `stopPlugin`

Stop a plugin


```zig
pub fn stopPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `startAllPlugins`

Start all plugins in dependency order


```zig
pub fn startAllPlugins(self: *PluginRegistry) !void {
```

- fn `stopAllPlugins`

Stop all plugins in reverse order


```zig
pub fn stopAllPlugins(self: *PluginRegistry) !void {
```

- fn `getPlugin`

Get plugin by name


```zig
pub fn getPlugin(self: *PluginRegistry, plugin_name: []const u8) ?*Plugin {
```

- fn `getPluginsByType`

Get plugins by type


```zig
pub fn getPluginsByType(self: *PluginRegistry, plugin_type: PluginType) !std.ArrayList(*Plugin) {
```

- fn `getPluginNames`

Get all plugin names


```zig
pub fn getPluginNames(self: *PluginRegistry) !std.ArrayList([]const u8) {
```

- fn `getPluginCount`

Get plugin count


```zig
pub fn getPluginCount(self: *PluginRegistry) usize {
```

- fn `getPluginInfo`

Get plugin information


```zig
pub fn getPluginInfo(self: *PluginRegistry, plugin_name: []const u8) ?*const PluginInfo {
```

- fn `configurePlugin`

Configure a plugin


```zig
pub fn configurePlugin(self: *PluginRegistry, plugin_name: []const u8, config: *const PluginConfig) !void {
```

- fn `broadcastEvent`

Send event to all interested plugins


```zig
pub fn broadcastEvent(self: *PluginRegistry, event_type: u32, event_data: ?*anyopaque) !void {
```

- fn `registerEventHandler`

Register an event handler


```zig
pub fn registerEventHandler(self: *PluginRegistry, event_type: u32, handler_fn: *const fn (event_data: ?*anyopaque) void) !void {
```

- fn `createRegistry`

Create a plugin registry instance


```zig
pub fn createRegistry(allocator: std.mem.Allocator) !PluginRegistry {
```

## src\plugins\types.zig

- const `PluginError`

Plugin system errors


```zig
pub const PluginError = error{
```

- type `PluginType`

Plugin types supported by the system


```zig
pub const PluginType = enum {
```

- fn `toString`

```zig
pub fn toString(self: PluginType) []const u8 {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?PluginType {
```

- type `PluginVersion`

Plugin version information


```zig
pub const PluginVersion = struct {
```

- fn `init`

```zig
pub fn init(major: u32, minor: u32, patch: u32) PluginVersion {
```

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginVersion, required: PluginVersion) bool {
```

- fn `format`

```zig
pub fn format(
```

- type `PluginInfo`

Plugin metadata and information


```zig
pub const PluginInfo = struct {
```

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginInfo, framework_abi: PluginVersion) bool {
```

- type `PluginConfig`

Plugin configuration


```zig
pub const PluginConfig = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) PluginConfig {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginConfig) void {
```

- fn `setParameter`

```zig
pub fn setParameter(self: *PluginConfig, key: []const u8, value: []const u8) !void {
```

- fn `getParameter`

```zig
pub fn getParameter(self: *PluginConfig, key: []const u8) ?[]const u8 {
```

- type `PluginState`

Plugin state tracking


```zig
pub const PluginState = enum {
```

- fn `toString`

```zig
pub fn toString(self: PluginState) []const u8 {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
```

- type `PluginContext`

Plugin execution context


```zig
pub const PluginContext = struct {
```

- fn `log`

```zig
pub fn log(self: *PluginContext, level: u8, message: []const u8) void {
```

- fn `getService`

```zig
pub fn getService(self: *PluginContext, service_name: []const u8) ?*anyopaque {
```

## src\monitoring\health.zig

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- fn `toString`

```zig
pub fn toString(self: HealthStatus) []const u8 {
```

- type `HealthCheck`

Individual health check result


```zig
pub const HealthCheck = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, status: HealthStatus, message: []const u8, response_time_ms: u64) !HealthCheck {
```

- fn `deinit`

```zig
pub fn deinit(self: *HealthCheck, allocator: std.mem.Allocator) void {
```

- fn `addMetadata`

```zig
pub fn addMetadata(self: *HealthCheck, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `HealthConfig`

Health checker configuration


```zig
pub const HealthConfig = struct {
```

- type `SystemHealth`

Overall system health status


```zig
pub const SystemHealth = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) SystemHealth {
```

- fn `deinit`

```zig
pub fn deinit(self: *SystemHealth) void {
```

- fn `updateOverallStatus`

```zig
pub fn updateOverallStatus(self: *SystemHealth) void {
```

- fn `addCheck`

```zig
pub fn addCheck(self: *SystemHealth, check: HealthCheck) !void {
```

- type `HealthChecker`

Comprehensive health checker


```zig
pub const HealthChecker = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: HealthConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) void {
```

- fn `setHealthChangeCallback`

```zig
pub fn setHealthChangeCallback(self: *Self, callback: *const fn (SystemHealth) void) void {
```

- fn `getCurrentHealth`

```zig
pub fn getCurrentHealth(self: *Self) *const SystemHealth {
```

- fn `exportHealthStatus`

Export health status as JSON


```zig
pub fn exportHealthStatus(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

## src\monitoring\memory_tracker.zig

- type `AllocationRecord`

Memory allocation record


```zig
pub const AllocationRecord = struct {
```

- fn `memoryUsage`

Calculate memory usage for this allocation


```zig
pub fn memoryUsage(self: AllocationRecord) usize {
```

- fn `age`

Get allocation age in nanoseconds


```zig
pub fn age(self: AllocationRecord, current_time: u64) u64 {
```

- fn `isPotentialLeak`

Check if allocation is a potential leak


```zig
pub fn isPotentialLeak(self: AllocationRecord, current_time: u64, leak_threshold_ns: u64) bool {
```

- type `MemoryStats`

Memory statistics snapshot


```zig
pub const MemoryStats = struct {
```

- fn `currentUsage`

Calculate current memory usage


```zig
pub fn currentUsage(self: MemoryStats) usize {
```

- fn `efficiency`

Calculate memory efficiency (1.0 = no waste, lower = more fragmentation)


```zig
pub fn efficiency(self: MemoryStats) f64 {
```

- fn `allocationSuccessRate`

Get allocation success rate


```zig
pub fn allocationSuccessRate(self: MemoryStats) f64 {
```

- type `MemoryProfilerConfig`

Memory profiler configuration


```zig
pub const MemoryProfilerConfig = struct {
```

- type `MemoryProfiler`

Memory profiler main structure


```zig
pub const MemoryProfiler = struct {
```

- fn `init`

Initialize memory profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !*MemoryProfiler {
```

- fn `deinit`

Deinitialize memory profiler


```zig
pub fn deinit(self: *MemoryProfiler) void {
```

- fn `recordAllocation`

Record a memory allocation


```zig
pub fn recordAllocation(
```

- fn `recordDeallocation`

Record a memory deallocation


```zig
pub fn recordDeallocation(self: *MemoryProfiler, id: u64) void {
```

- fn `getStats`

Get current memory statistics


```zig
pub fn getStats(self: *MemoryProfiler) MemoryStats {
```

- fn `getPotentialLeaks`

Get potential memory leaks


```zig
pub fn getPotentialLeaks(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]AllocationRecord {
```

- fn `generateReport`

Generate memory usage report


```zig
pub fn generateReport(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]u8 {
```

- fn `resetStats`

Reset statistics


```zig
pub fn resetStats(self: *MemoryProfiler) void {
```

- fn `collectPeriodicStats`

Collect periodic statistics


```zig
pub fn collectPeriodicStats(self: *MemoryProfiler) void {
```

- fn `initGlobalProfiler`

Initialize global memory profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global memory profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global memory profiler instance


```zig
pub fn getGlobalProfiler() ?*MemoryProfiler {
```

- type `TrackedAllocator`

Tracked allocator that integrates with memory profiler


```zig
pub const TrackedAllocator = struct {
```

- fn `init`

Initialize tracked allocator


```zig
pub fn init(parent_allocator: std.mem.Allocator, profiler: *MemoryProfiler) TrackedAllocator {
```

- fn `allocator`

Get allocator interface


```zig
pub fn allocator(self: *TrackedAllocator) std.mem.Allocator {
```

- type `MemoryMonitor`

Memory usage monitor


```zig
pub const MemoryMonitor = struct {
```

- fn `init`

Initialize memory monitor


```zig
pub fn init(profiler: *MemoryProfiler) !*MemoryMonitor {
```

- fn `start`

Start monitoring thread


```zig
pub fn start(self: *MemoryMonitor) !void {
```

- fn `stop`

Stop monitoring


```zig
pub fn stop(self: *MemoryMonitor) void {
```

- fn `deinit`

Deinitialize monitor


```zig
pub fn deinit(self: *MemoryMonitor) void {
```

- type `PerformanceMonitor`

Performance monitoring utilities


```zig
pub const PerformanceMonitor = struct {
```

- fn `start`

Start performance measurement


```zig
pub fn start(self: *PerformanceMonitor) void {
```

- fn `end`

End performance measurement


```zig
pub fn end(self: *PerformanceMonitor) void {
```

- fn `elapsedTime`

Get elapsed time in nanoseconds


```zig
pub fn elapsedTime(self: PerformanceMonitor) u64 {
```

- fn `memoryDelta`

Get memory usage delta


```zig
pub fn memoryDelta(self: PerformanceMonitor) i64 {
```

- fn `generateReport`

Generate performance report


```zig
pub fn generateReport(self: PerformanceMonitor, allocator: std.mem.Allocator, operation_name: []const u8) ![]u8 {
```

- type `utils`

Utility functions for memory profiling


```zig
pub const utils = struct {
```

- fn `simpleConfig`

Create a simple memory profiler configuration


```zig
pub fn simpleConfig() MemoryProfilerConfig {
```

- fn `developmentConfig`

Create a development configuration with more detailed tracking


```zig
pub fn developmentConfig() MemoryProfilerConfig {
```

- fn `productionConfig`

Create a production configuration with minimal overhead


```zig
pub fn productionConfig() MemoryProfilerConfig {
```

## src\monitoring\mod.zig

- const `prometheus`

```zig
pub const prometheus = @import("prometheus.zig");
```

- const `sampling`

```zig
pub const sampling = @import("sampling.zig");
```

- const `regression`

```zig
pub const regression = @import("regression.zig");
```

- const `health`

```zig
pub const health = @import("health.zig");
```

- const `tracing`

```zig
pub const tracing = @import("tracing.zig");
```

- const `memory_tracker`

```zig
pub const memory_tracker = @import("memory_tracker.zig");
```

- const `performance_profiler`

```zig
pub const performance_profiler = @import("performance_profiler.zig");
```

- const `performance`

```zig
pub const performance = @import("performance.zig");
```

- const `PrometheusServer`

```zig
pub const PrometheusServer = prometheus.PrometheusServer;
```

- const `MetricsCollector`

```zig
pub const MetricsCollector = prometheus.MetricsCollector;
```

- const `PerformanceSampler`

```zig
pub const PerformanceSampler = sampling.PerformanceSampler;
```

- const `RegressionDetector`

```zig
pub const RegressionDetector = regression.RegressionDetector;
```

- const `HealthChecker`

```zig
pub const HealthChecker = health.HealthChecker;
```

- const `Tracer`

```zig
pub const Tracer = tracing.Tracer;
```

- const `TraceId`

```zig
pub const TraceId = tracing.TraceId;
```

- const `Span`

```zig
pub const Span = tracing.Span;
```

- const `SpanId`

```zig
pub const SpanId = tracing.SpanId;
```

- const `TraceContext`

```zig
pub const TraceContext = tracing.TraceContext;
```

- const `TracingError`

```zig
pub const TracingError = tracing.TracingError;
```

- const `MemoryTracker`

```zig
pub const MemoryTracker = memory_tracker.MemoryTracker;
```

- const `PerformanceProfiler`

```zig
pub const PerformanceProfiler = performance_profiler.PerformanceProfiler;
```

- const `PerformanceMetrics`

```zig
pub const PerformanceMetrics = performance.PerformanceMetrics;
```

## src\monitoring\performance.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `PerformanceError`

Performance monitoring specific error types


```zig
pub const PerformanceError = error{
```

- type `MetricType`

Performance metric types


```zig
pub const MetricType = enum {
```

- const `MetricValue`

Performance metric value


```zig
pub const MetricValue = union(MetricType) {
```

- type `HistogramData`

Histogram data for latency measurements


```zig
pub const HistogramData = struct {
```

- fn `record`

```zig
pub fn record(self: *HistogramData, value: f64) void {
```

- fn `percentile`

```zig
pub fn percentile(self: *const HistogramData, p: f64) f64 {
```

- type `TimerData`

Timer data for duration measurements


```zig
pub const TimerData = struct {
```

- fn `start`

```zig
pub fn start(self: *TimerData) void {
```

- fn `stop`

```zig
pub fn stop(self: *TimerData) void {
```

- fn `averageDuration`

```zig
pub fn averageDuration(self: *const TimerData) f64 {
```

- type `Metric`

Performance metric entry


```zig
pub const Metric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, value: MetricValue) !Metric {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- fn `addLabel`

```zig
pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `CPUProfiler`

CPU profiler with sampling


```zig
pub const CPUProfiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, sampling_rate: u32) CPUProfiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *CPUProfiler) void {
```

- fn `start`

```zig
pub fn start(self: *CPUProfiler) !void {
```

- fn `stop`

```zig
pub fn stop(self: *CPUProfiler) void {
```

- type `MemoryTracker`

Memory allocation tracker


```zig
pub const MemoryTracker = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryTracker) void {
```

- fn `recordAllocation`

```zig
pub fn recordAllocation(self: *MemoryTracker, ptr: usize, size: usize) void {
```

- fn `recordDeallocation`

```zig
pub fn recordDeallocation(self: *MemoryTracker, ptr: usize) void {
```

- fn `getCurrentUsage`

```zig
pub fn getCurrentUsage(self: *const MemoryTracker) u64 {
```

- fn `getPeakUsage`

```zig
pub fn getPeakUsage(self: *const MemoryTracker) u64 {
```

- type `PerformanceMonitor`

Global performance monitoring system


```zig
pub const PerformanceMonitor = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(self: *PerformanceMonitor, name: []const u8, value: MetricValue) !void {
```

- fn `startProfiling`

```zig
pub fn startProfiling(self: *PerformanceMonitor) !void {
```

- fn `stopProfiling`

```zig
pub fn stopProfiling(self: *PerformanceMonitor) void {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *PerformanceMonitor, name: []const u8) ?Metric {
```

- type `TracyProfiler`

Tracy profiler integration (when enabled)


```zig
pub const TracyProfiler = struct {
```

- fn `zoneName`

```zig
pub fn zoneName(comptime name: []const u8) void {
```

- fn `zoneStart`

```zig
pub fn zoneStart() void {
```

- fn `zoneEnd`

```zig
pub fn zoneEnd() void {
```

- fn `plot`

```zig
pub fn plot(name: []const u8, value: f64) void {
```

- fn `init`

```zig
pub fn init() !void {
```

- fn `deinit`

```zig
pub fn deinit() void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(name: []const u8, value: f64) void {
```

- fn `recordCounter`

```zig
pub fn recordCounter(name: []const u8, value: u64) void {
```

- fn `recordLatency`

```zig
pub fn recordLatency(name: []const u8, duration_ns: u64) void {
```

- type `Timer`

Timer utility for measuring execution time


```zig
pub const Timer = struct {
```

- fn `start`

```zig
pub fn start(comptime name: []const u8) Timer {
```

- fn `stop`

```zig
pub fn stop(self: Timer) void {
```

- fn `timed`

Convenient macro for timing function execution


```zig
pub fn timed(comptime name: []const u8, func: anytype) @TypeOf(func()) {
```

## src\monitoring\performance_profiler.zig

- type `ProfilingConfig`

Performance profiling configuration


```zig
pub const ProfilingConfig = struct {
```

- type `CallRecord`

Function call record (for call tracing and call tree)


```zig
pub const CallRecord = struct {
```

- fn `duration`

Calculate call duration (nanoseconds)


```zig
pub fn duration(self: CallRecord) u64 {
```

- fn `isComplete`

Check if call is complete (has exit time)


```zig
pub fn isComplete(self: CallRecord) bool {
```

- type `PerformanceCounter`

Performance counter (for custom and built-in metrics)


```zig
pub const PerformanceCounter = struct {
```

- fn `increment`

```zig
pub fn increment(self: *PerformanceCounter) void {
```

- fn `add`

```zig
pub fn add(self: *PerformanceCounter, delta: u64) void {
```

- fn `set`

```zig
pub fn set(self: *PerformanceCounter, new_value: u64) void {
```

- fn `reset`

```zig
pub fn reset(self: *PerformanceCounter) void {
```

- type `PerformanceProfile`

Performance profile data (per session)


```zig
pub const PerformanceProfile = struct {
```

- fn `duration`

```zig
pub fn duration(self: PerformanceProfile) u64 {
```

- fn `durationSeconds`

```zig
pub fn durationSeconds(self: PerformanceProfile) f64 {
```

- fn `cpuUtilization`

```zig
pub fn cpuUtilization(self: PerformanceProfile) f64 {
```

- type `FunctionProfiler`

Function profiler for instrumenting and aggregating function stats


```zig
pub const FunctionProfiler = struct {
```

- fn `enter`

```zig
pub fn enter(self: *FunctionProfiler) u64 {
```

- fn `exit`

```zig
pub fn exit(self: *FunctionProfiler, entry_time: u64) void {
```

- fn `averageExecutionTime`

```zig
pub fn averageExecutionTime(self: FunctionProfiler) u64 {
```

- type `PerformanceProfiler`

Main performance profiler


```zig
pub const PerformanceProfiler = struct {
```

- fn `init`

Initialize performance profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: ProfilingConfig) !*PerformanceProfiler {
```

- fn `deinit`

Deinitialize performance profiler and free all resources


```zig
pub fn deinit(self: *PerformanceProfiler) void {
```

- fn `startSession`

Start profiling session


```zig
pub fn startSession(self: *PerformanceProfiler, session_name: []const u8) !void {
```

- fn `endSession`

End profiling session and return report


```zig
pub fn endSession(self: *PerformanceProfiler) ![]u8 {
```

- fn `startFunctionCall`

Start function call (for call tracing)


```zig
pub fn startFunctionCall(self: *PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) !u64 {
```

- fn `endFunctionCall`

End function call (for call tracing)


```zig
pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void {
```

- fn `updateCounter`

Update or create a performance counter


```zig
pub fn updateCounter(self: *PerformanceProfiler, name: []const u8, delta: u64) void {
```

- fn `getFunctionStats`

Get function profiler statistics (sorted by total_time descending)


```zig
pub fn getFunctionStats(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]FunctionProfiler {
```

- fn `stop`

Stop profiling thread


```zig
pub fn stop(self: *PerformanceProfiler) void {
```

- fn `setMemoryTracker`

Integrate with memory tracker


```zig
pub fn setMemoryTracker(self: *PerformanceProfiler, tracker: *memory_tracker.MemoryProfiler) void {
```

- fn `createScope`

Create performance scope for measuring code blocks


```zig
pub fn createScope(self: *PerformanceProfiler, name: []const u8) Scope {
```

- type `Scope`

Performance measurement scope (RAII-style)


```zig
pub const Scope = struct {
```

- fn `end`

End the scope and record measurements


```zig
pub fn end(self: Scope) void {
```

- fn `initGlobalProfiler`

Initialize global performance profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: ProfilingConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global performance profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global performance profiler instance


```zig
pub fn getGlobalProfiler() ?*PerformanceProfiler {
```

- fn `startScope`

Convenience function to start a performance scope


```zig
pub fn startScope(name: []const u8) ?Scope {
```

- fn `profileFunctionCall`

Convenience function for profiling function calls (to be used with defer)


```zig
pub fn profileFunctionCall(profiler: ?*PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) FunctionCall {
```

- type `FunctionCall`

Function call scope for automatic profiling (RAII-style)


```zig
pub const FunctionCall = struct {
```

- fn `end`

```zig
pub fn end(self: FunctionCall) void {
```

- type `utils`

Performance monitoring utilities and presets


```zig
pub const utils = struct {
```

- fn `developmentConfig`

Create a development profiling configuration


```zig
pub fn developmentConfig() ProfilingConfig {
```

- fn `productionConfig`

Create a production profiling configuration


```zig
pub fn productionConfig() ProfilingConfig {
```

- fn `minimalConfig`

Create a minimal profiling configuration


```zig
pub fn minimalConfig() ProfilingConfig {
```

## src\monitoring\prometheus.zig

- type `MetricType`

Prometheus metric types


```zig
pub const MetricType = enum {
```

- type `Metric`

Individual metric definition


```zig
pub const Metric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, help: []const u8, metric_type: MetricType) !Metric {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- fn `addLabel`

```zig
pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- fn `setValue`

```zig
pub fn setValue(self: *Metric, value: f64) void {
```

- fn `increment`

```zig
pub fn increment(self: *Metric) void {
```

- fn `add`

```zig
pub fn add(self: *Metric, value: f64) void {
```

- type `MetricsCollector`

Metrics collector and registry


```zig
pub const MetricsCollector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `registerMetric`

```zig
pub fn registerMetric(self: *Self, name: []const u8, help: []const u8, metric_type: MetricType) !*Metric {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *Self, name: []const u8) ?*Metric {
```

- fn `recordDatabaseOperation`

```zig
pub fn recordDatabaseOperation(self: *Self, operation: []const u8, duration_ns: u64) !void {
```

- fn `updateDatabaseStats`

```zig
pub fn updateDatabaseStats(self: *Self, vectors_stored: u64, compression_ratio: f64) void {
```

- fn `recordSearch`

```zig
pub fn recordSearch(self: *Self, duration_ns: u64, results_count: usize) !void {
```

- fn `recordHttpRequest`

```zig
pub fn recordHttpRequest(self: *Self, method: []const u8, path: []const u8, status_code: u16, duration_ns: u64, response_size: usize) !void {
```

- fn `updateSystemMetrics`

```zig
pub fn updateSystemMetrics(self: *Self, cpu_percent: f64, memory_used: u64, memory_available: u64, disk_used: u64) void {
```

- fn `updateProcessMetrics`

```zig
pub fn updateProcessMetrics(self: *Self, cpu_seconds: f64, memory_bytes: u64, thread_count: u32) void {
```

- fn `exportPrometheusFormat`

Export metrics in Prometheus format


```zig
pub fn exportPrometheusFormat(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

- type `PrometheusServer`

Prometheus HTTP server for metrics export


```zig
pub const PrometheusServer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metrics_collector: *MetricsCollector, host: []const u8, port: u16, path: []const u8) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

## src\monitoring\regression.zig

- type `RegressionSensitivity`

Regression sensitivity levels


```zig
pub const RegressionSensitivity = enum {
```

- type `PerformanceMetric`

Performance metric for regression analysis


```zig
pub const PerformanceMetric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, value: f64) !PerformanceMetric {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMetric, allocator: std.mem.Allocator) void {
```

- fn `addMetadata`

```zig
pub fn addMetadata(self: *PerformanceMetric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `PerformanceBaseline`

Statistical performance baseline


```zig
pub const PerformanceBaseline = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metric_name: []const u8) !PerformanceBaseline {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceBaseline, allocator: std.mem.Allocator) void {
```

- fn `updateWithValue`

```zig
pub fn updateWithValue(self: *PerformanceBaseline, value: f64) void {
```

- fn `isRegression`

```zig
pub fn isRegression(self: *const PerformanceBaseline, value: f64, sensitivity: RegressionSensitivity) bool {
```

- fn `getRegressionSeverity`

```zig
pub fn getRegressionSeverity(self: *const PerformanceBaseline, value: f64) RegressionSensitivity {
```

- type `RegressionAlert`

Regression alert information


```zig
pub const RegressionAlert = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metric_name: []const u8, current_value: f64, baseline_mean: f64, severity: RegressionSensitivity) !RegressionAlert {
```

- fn `deinit`

```zig
pub fn deinit(self: *RegressionAlert, allocator: std.mem.Allocator) void {
```

- type `RegressionConfig`

Configuration for regression detection


```zig
pub const RegressionConfig = struct {
```

- type `RegressionDetector`

Performance regression detector


```zig
pub const RegressionDetector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: RegressionConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `setRegressionCallback`

```zig
pub fn setRegressionCallback(self: *Self, callback: *const fn (RegressionAlert) void) void {
```

- fn `recordMetric`

Record a performance metric and check for regressions


```zig
pub fn recordMetric(self: *Self, metric: PerformanceMetric) !void {
```

- fn `getBaseline`

Get baseline for a metric


```zig
pub fn getBaseline(self: *Self, metric_name: []const u8) ?*const PerformanceBaseline {
```

- fn `getRecentAlerts`

Get recent alerts


```zig
pub fn getRecentAlerts(self: *Self, limit: ?usize) []const RegressionAlert {
```

- fn `exportStats`

Export regression detection statistics


```zig
pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `updateAllBaselines`

Force update all baselines (useful for testing)


```zig
pub fn updateAllBaselines(self: *Self) !void {
```

## src\monitoring\sampling.zig

- type `PerformanceSample`

Performance sample data point


```zig
pub const PerformanceSample = struct {
```

- fn `init`

```zig
pub fn init() PerformanceSample {
```

- fn `calculateMemoryUsagePercent`

```zig
pub fn calculateMemoryUsagePercent(self: *const PerformanceSample) f64 {
```

- fn `calculateProcessMemoryPercent`

```zig
pub fn calculateProcessMemoryPercent(self: *const PerformanceSample) f64 {
```

- type `SamplerConfig`

Performance sampler configuration


```zig
pub const SamplerConfig = struct {
```

- fn `getCpuUsage`

Get system CPU usage percentage


```zig
pub fn getCpuUsage() !f64 {
```

- fn `getMemoryInfo`

Get system memory information


```zig
pub fn getMemoryInfo() !struct { total: u64, used: u64, available: u64 } {
```

- fn `getProcessInfo`

Get process-specific information


```zig
pub fn getProcessInfo() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
```

- fn `getDiskInfo`

Get disk usage information


```zig
pub fn getDiskInfo() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
```

- type `PerformanceSampler`

Performance sampler with periodic monitoring


```zig
pub const PerformanceSampler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: SamplerConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) void {
```

- fn `setAlertCallbacks`

```zig
pub fn setAlertCallbacks(self: *Self, cpu_callback: ?*const fn (f64) void, memory_callback: ?*const fn (f64) void, disk_callback: ?*const fn (f64) void) void {
```

- fn `getCurrentSample`

```zig
pub fn getCurrentSample(self: *Self) ?PerformanceSample {
```

- fn `getAverageCpuUsage`

```zig
pub fn getAverageCpuUsage(self: *Self, duration_seconds: u32) f64 {
```

- fn `getAverageMemoryUsage`

```zig
pub fn getAverageMemoryUsage(self: *Self, duration_seconds: u32) f64 {
```

- fn `getSamplesInRange`

```zig
pub fn getSamplesInRange(self: *Self, start_time: i64, end_time: i64, allocator: std.mem.Allocator) ![]PerformanceSample {
```

- fn `exportStats`

Export performance statistics


```zig
pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

## src\monitoring\tracing.zig

- const `TracingError`

Tracing error types


```zig
pub const TracingError = error{
```

- type `TraceId`

Trace ID - unique identifier for a trace


```zig
pub const TraceId = struct {
```

- fn `init`

```zig
pub fn init() TraceId {
```

- fn `toString`

```zig
pub fn toString(self: TraceId, allocator: std.mem.Allocator) ![]u8 {
```

- fn `fromString`

```zig
pub fn fromString(str: []const u8) !TraceId {
```

- const `SpanId`

Span ID - unique identifier for a span within a trace


```zig
pub const SpanId = u64;
```

- type `SpanKind`

Span kind enumeration


```zig
pub const SpanKind = enum {
```

- type `SpanStatus`

Span status


```zig
pub const SpanStatus = enum {
```

- type `Span`

Trace span representing a single operation


```zig
pub const Span = struct {
```

- type `SpanEvent`

Span event for annotations


```zig
pub const SpanEvent = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *SpanEvent, allocator: std.mem.Allocator) void {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *Span, allocator: std.mem.Allocator) void {
```

- fn `end`

End the span


```zig
pub fn end(self: *Span) void {
```

- fn `setStatus`

Set span status


```zig
pub fn setStatus(self: *Span, status: SpanStatus) void {
```

- fn `setAttribute`

Add an attribute to the span


```zig
pub fn setAttribute(self: *Span, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- fn `addEvent`

Add an event to the span


```zig
pub fn addEvent(self: *Span, allocator: std.mem.Allocator, name: []const u8) !void {
```

- fn `duration`

Get span duration in nanoseconds


```zig
pub fn duration(self: Span) ?i128 {
```

- fn `isActive`

Check if span is still active


```zig
pub fn isActive(self: Span) bool {
```

- type `TraceContext`

Trace context for propagating tracing information


```zig
pub const TraceContext = struct {
```

- fn `init`

Create a new trace context


```zig
pub fn init() TraceContext {
```

- fn `child`

Create child context


```zig
pub fn child(self: TraceContext) TraceContext {
```

- fn `toString`

Serialize context to string


```zig
pub fn toString(self: TraceContext, allocator: std.mem.Allocator) ![]u8 {
```

- fn `fromString`

Deserialize context from string


```zig
pub fn fromString(str: []const u8) !TraceContext {
```

- type `Tracer`

Tracer - main tracing interface


```zig
pub const Tracer = struct {
```

- type `TracerConfig`

```zig
pub const TracerConfig = struct {
```

- const `Sampler`

Sampling strategy


```zig
pub const Sampler = union(enum) {
```

- fn `shouldSample`

```zig
pub fn shouldSample(self: *Sampler) bool {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: TracerConfig) !*Tracer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Tracer) void {
```

- fn `startSpan`

Start a new span


```zig
pub fn startSpan(self: *Tracer, name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
```

- fn `endSpan`

End a span


```zig
pub fn endSpan(self: *Tracer, span: *Span) void {
```

- fn `getSpan`

Get active span by ID


```zig
pub fn getSpan(self: *Tracer, span_id: SpanId) ?*Span {
```

- fn `exportToJson`

Export traces to JSON (simplified)


```zig
pub fn exportToJson(self: *Tracer, allocator: std.mem.Allocator) ![]u8 {
```

- fn `initGlobalTracer`

Initialize global tracer


```zig
pub fn initGlobalTracer(allocator: std.mem.Allocator, config: Tracer.TracerConfig) !void {
```

- fn `deinitGlobalTracer`

Deinitialize global tracer


```zig
pub fn deinitGlobalTracer() void {
```

- fn `getGlobalTracer`

Get global tracer instance


```zig
pub fn getGlobalTracer() ?*Tracer {
```

- fn `startSpan`

Helper function to start a span with global tracer


```zig
pub fn startSpan(name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
```

- fn `endSpan`

Helper function to end a span with global tracer


```zig
pub fn endSpan(span: *Span) void {
```

- fn `traceFunction`

Convenience macro-like function for tracing function calls


```zig
pub fn traceFunction(comptime func_name: []const u8, context: ?TraceContext) !*Span {
```

- fn `integrateWithPerformance`

Integration with performance monitoring


```zig
pub fn integrateWithPerformance(tracer: *Tracer, perf_monitor: *performance.PerformanceMonitor) void {
```

## src\gpu\mod.zig

- const `core`

```zig
pub const core = @import("core/mod.zig");
```

- const `unified_memory`

```zig
pub const unified_memory = @import("unified_memory.zig");
```

- const `GPURenderer`

```zig
pub const GPURenderer = core.GPURenderer;
```

- const `GPUConfig`

```zig
pub const GPUConfig = core.GPUConfig;
```

- const `GpuError`

```zig
pub const GpuError = core.GpuError;
```

- const `GpuBackend`

```zig
pub const GpuBackend = core.GpuBackend;
```

- const `GpuBackendConfig`

```zig
pub const GpuBackendConfig = core.GpuBackendConfig;
```

- const `GpuBackendError`

```zig
pub const GpuBackendError = core.GpuBackendError;
```

- const `BatchConfig`

```zig
pub const BatchConfig = core.BatchConfig;
```

- const `BatchProcessor`

```zig
pub const BatchProcessor = core.BatchProcessor;
```

- const `GpuStats`

```zig
pub const GpuStats = core.GpuStats;
```

- const `Db`

```zig
pub const Db = core.Db;
```

- const `KernelManager`

```zig
pub const KernelManager = core.KernelManager;
```

- const `GPUBackendManager`

```zig
pub const GPUBackendManager = core.GPUBackendManager;
```

- const `CUDADriver`

```zig
pub const CUDADriver = core.CUDADriver;
```

- const `SPIRVCompiler`

```zig
pub const SPIRVCompiler = core.SPIRVCompiler;
```

- const `BackendType`

```zig
pub const BackendType = core.BackendType;
```

- const `HardwareCapabilities`

```zig
pub const HardwareCapabilities = core.HardwareCapabilities;
```

- const `MemoryPool`

```zig
pub const MemoryPool = core.MemoryPool;
```

- const `BackendSupport`

```zig
pub const BackendSupport = core.BackendSupport;
```

- const `PerformanceProfiler`

```zig
pub const PerformanceProfiler = core.PerformanceProfiler;
```

- const `MemoryBandwidthBenchmark`

```zig
pub const MemoryBandwidthBenchmark = core.MemoryBandwidthBenchmark;
```

- const `ComputeThroughputBenchmark`

```zig
pub const ComputeThroughputBenchmark = core.ComputeThroughputBenchmark;
```

- const `Backend`

```zig
pub const Backend = core.Backend;
```

- const `PowerPreference`

```zig
pub const PowerPreference = core.PowerPreference;
```

- const `has_webgpu_support`

```zig
pub const has_webgpu_support = core.has_webgpu_support;
```

- const `Color`

```zig
pub const Color = core.Color;
```

- const `GPUHandle`

```zig
pub const GPUHandle = core.GPUHandle;
```

- const `UnifiedMemoryManager`

```zig
pub const UnifiedMemoryManager = unified_memory.UnifiedMemoryManager;
```

- const `UnifiedMemoryType`

```zig
pub const UnifiedMemoryType = unified_memory.UnifiedMemoryType;
```

- const `UnifiedMemoryConfig`

```zig
pub const UnifiedMemoryConfig = unified_memory.UnifiedMemoryConfig;
```

- const `UnifiedBuffer`

```zig
pub const UnifiedBuffer = unified_memory.UnifiedBuffer;
```

- const `initDefault`

```zig
pub const initDefault = core.initDefault;
```

- const `isGpuAvailable`

```zig
pub const isGpuAvailable = core.isGpuAvailable;
```

- type `utils`

```zig
pub const utils = struct {
```

- fn `isAccelerationAvailable`

Check if any GPU acceleration is available


```zig
pub fn isAccelerationAvailable() bool {
```

- fn `getRecommendedConfig`

Get recommended GPU configuration


```zig
pub fn getRecommendedConfig(_: std.mem.Allocator) !core.GPUConfig {
```

- fn `initRecommended`

Initialize GPU system with recommended settings


```zig
pub fn initRecommended(allocator: std.mem.Allocator) !*core.GPURenderer {
```

- type `version`

```zig
pub const version = struct {
```

- const `major`

```zig
pub const major = 1;
```

- const `minor`

```zig
pub const minor = 0;
```

- const `patch`

```zig
pub const patch = 0;
```

- const `string`

```zig
pub const string = std.fmt.comptimePrint("{}.{}.{}", .{ major, minor, patch });
```

## src\gpu\unified_memory.zig

- type `UnifiedMemoryType`

Unified Memory Architecture types


```zig
pub const UnifiedMemoryType = enum {
```

- type `UnifiedMemoryConfig`

Unified Memory Configuration


```zig
pub const UnifiedMemoryConfig = struct {
```

- type `UnifiedMemoryManager`

Unified Memory Manager


```zig
pub const UnifiedMemoryManager = struct {
```

- fn `init`

Initialize the unified memory manager


```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

Deinitialize the unified memory manager


```zig
pub fn deinit(self: *Self) void {
```

- fn `allocateUnified`

Allocate unified memory that can be accessed by both CPU and GPU


```zig
pub fn allocateUnified(self: *Self, size: usize, alignment: u29) ![]u8 {
```

- fn `freeUnified`

Free unified memory


```zig
pub fn freeUnified(self: *Self, memory: []u8) void {
```

- fn `getPerformanceInfo`

Get unified memory performance characteristics


```zig
pub fn getPerformanceInfo(self: *Self) struct {
```

- type `UnifiedBuffer`

Unified Memory Buffer for zero-copy operations


```zig
pub const UnifiedBuffer = struct {
```

- fn `create`

Create a new unified buffer


```zig
pub fn create(manager: *UnifiedMemoryManager, size: usize) !Self {
```

- fn `destroy`

Destroy the unified buffer


```zig
pub fn destroy(self: *const Self) void {
```

- fn `getData`

Get raw data pointer


```zig
pub fn getData(self: *Self) []u8 {
```

- fn `getSize`

Get buffer size


```zig
pub fn getSize(self: *Self) usize {
```

- fn `isGpuAccessible`

Check if buffer is GPU accessible


```zig
pub fn isGpuAccessible(self: *const Self) bool {
```

- fn `transferToGpu`

Zero-copy data transfer (if supported)


```zig
pub fn transferToGpu(self: *Self) !void {
```

- fn `transferFromGpu`

Zero-copy data transfer from GPU (if supported)


```zig
pub fn transferFromGpu(self: *Self) !void {
```

## src\gpu\memory\memory_pool.zig

- type `MemoryPoolConfig`

GPU Memory Pool Configuration


```zig
pub const MemoryPoolConfig = struct {
```

- type `MemoryStats`

Memory pool statistics


```zig
pub const MemoryStats = struct {
```

- type `BufferMetadata`

Buffer metadata for pool management


```zig
pub const BufferMetadata = struct {
```

- type `MemoryPool`

GPU Memory Pool Manager


```zig
pub const MemoryPool = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate a buffer from the pool or create a new one


```zig
pub fn allocBuffer(
```

- fn `freeBuffer`

Return a buffer to the pool for reuse


```zig
pub fn freeBuffer(self: *MemoryPool, handle: u32) !void {
```

- fn `cleanup`

Periodic cleanup of old buffers


```zig
pub fn cleanup(self: *MemoryPool) !void {
```

- fn `getStats`

Get comprehensive memory statistics


```zig
pub fn getStats(self: *MemoryPool) MemoryStats {
```

- fn `defragment`

Defragment memory pool (advanced feature)


```zig
pub fn defragment(self: *MemoryPool) !void {
```

- fn `prefetchBuffers`

Prefetch buffers for anticipated usage patterns


```zig
pub fn prefetchBuffers(self: *MemoryPool, sizes: []const usize, usage: gpu_renderer.BufferUsage) !void {
```

- fn `resizeBuffer`

Resize a buffer while maintaining pool efficiency


```zig
pub fn resizeBuffer(self: *MemoryPool, old_handle: u32, new_size: usize) !u32 {
```

- fn `getMemoryReport`

Get memory usage report


```zig
pub fn getMemoryReport(self: *MemoryPool, allocator: std.mem.Allocator) ![]const u8 {
```

## src\gpu\demo\gpu_demo.zig

- fn `format`

```zig
pub fn format(self: PerformanceMetrics, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
```

- fn `fromTemperature`

```zig
pub fn fromTemperature(temp_c: f32) ThermalState {
```

- fn `detect`

```zig
pub fn detect() ArchitectureFeatures {
```

- fn `logFeatures`

```zig
pub fn logFeatures(self: ArchitectureFeatures) void {
```

- fn `main`

```zig
pub fn main() !void {
```

- fn `update`

```zig
pub fn update(self: *HardwareMonitor) void {
```

- fn `checkThrottling`

```zig
pub fn checkThrottling(self: *ThermalMonitor) bool {
```

- fn `checkPowerLimit`

```zig
pub fn checkPowerLimit(self: *PowerMonitor) bool {
```

- fn `deinit`

```zig
pub fn deinit(self: ComprehensiveReportData) void {
```

## src\gpu\core\backend.zig

- type `Db`

```zig
pub const Db = struct {
```

- type `Result`

```zig
pub const Result = struct {
```

- fn `lessThanAsc`

```zig
pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
```

- type `Header`

```zig
pub const Header = struct {
```

- fn `open`

```zig
pub fn open(_: []const u8, _: bool) !*Db {
```

- fn `close`

```zig
pub fn close(_: *Db) void {
```

- fn `init`

```zig
pub fn init(_: *Db, _: u32) !void {
```

- fn `addEmbedding`

```zig
pub fn addEmbedding(_: *Db, _: []const f32) !u32 {
```

- type `GpuBackendConfig`

Configuration for the GPU backend.


```zig
pub const GpuBackendConfig = struct {
```

- type `GpuBackend`

Main context for GPU-accelerated operations.


```zig
pub const GpuBackend = struct {
```

- const `Error`

Error set for all GPU backend operations.


```zig
pub const Error = error{
```

- fn `init`

Initialize the GPU backend with the given configuration.


```zig
pub fn init(allocator: std.mem.Allocator, config: GpuBackendConfig) Error!*GpuBackend {
```

- fn `deinit`

Clean up and release all resources held by the backend.


```zig
pub fn deinit(self: *GpuBackend) void {
```

- fn `isGpuAvailable`

Check if a suitable GPU is available for compute.


```zig
pub fn isGpuAvailable(self: *const GpuBackend) bool {
```

- fn `searchSimilar`

Perform a vector similarity search for the given query vector against the database.
Returns the top_k closest results. Falls back to CPU if GPU is unavailable.


```zig
pub fn searchSimilar(self: *GpuBackend, db: *const Db, query: []const f32, top_k: usize) Error![]Db.Result {
```

- fn `hasMemoryFor`

Returns true if there is enough GPU memory for an operation.


```zig
pub fn hasMemoryFor(self: *const GpuBackend, bytes: u64) bool {
```

- fn `batchSearch`

Batch version of searchSimilar, processes multiple queries and returns results for each.


```zig
pub fn batchSearch(self: *GpuBackend, db: *const Db, queries: []const []const f32, top_k: usize) Error![][]Db.Result {
```

- type `BatchConfig`

Configuration for batch processing.


```zig
pub const BatchConfig = struct {
```

- type `BatchProcessor`

Utility for batch processing of vector search queries.


```zig
pub const BatchProcessor = struct {
```

- fn `init`

```zig
pub fn init(backend: *GpuBackend, config: BatchConfig) BatchProcessor {
```

- fn `processBatch`

Process a batch of queries with optional progress reporting.


```zig
pub fn processBatch(self: *BatchProcessor, db: *const Db, queries: []const []const f32, top_k: usize) ![][]Db.Result {
```

- fn `processBatchWithCallback`

Process queries with a callback for each result.


```zig
pub fn processBatchWithCallback(
```

- type `GpuStats`

Tracks performance statistics for GPU operations.


```zig
pub const GpuStats = struct {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *GpuStats, duration_us: u64, memory_used: u64, used_cpu: bool) void {
```

- fn `getAverageOperationTime`

```zig
pub fn getAverageOperationTime(self: *const GpuStats) u64 {
```

- fn `print`

```zig
pub fn print(self: *const GpuStats) void {
```

## src\gpu\core\gpu_renderer.zig

- const `has_webgpu_support`

```zig
pub const has_webgpu_support = @hasDecl(std, "gpu") and @hasDecl(std.gpu, "Instance");
```

- const `GpuError`

GPU renderer errors


```zig
pub const GpuError = error{
```

- type `SPIRVOptimizationLevel`

SPIR-V compiler optimization levels


```zig
pub const SPIRVOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toInt(self: SPIRVOptimizationLevel) u32 {
```

- type `SPIRVCompilerOptions`

SPIR-V compiler configuration structure


```zig
pub const SPIRVCompilerOptions = struct {
```

- fn `validate`

```zig
pub fn validate(self: SPIRVCompilerOptions) !void {
```

- type `MSLOptimizationLevel`

Metal Shading Language optimization levels


```zig
pub const MSLOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toInt(self: MSLOptimizationLevel) u32 {
```

- type `MetalVersion`

Metal target versions


```zig
pub const MetalVersion = enum {
```

- pub `inline`

```zig
pub inline fn toVersionString(self: MetalVersion) []const u8 {
```

- type `MSLCompilerOptions`

Metal Shading Language compiler configuration structure


```zig
pub const MSLCompilerOptions = struct {
```

- type `Platform`

```zig
pub const Platform = enum {
```

- pub `inline`

```zig
pub inline fn isApplePlatform(self: Platform) bool {
```

- fn `validate`

```zig
pub fn validate(self: MSLCompilerOptions) !void {
```

- type `PTXOptimizationLevel`

PTX optimization levels


```zig
pub const PTXOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toString(self: PTXOptimizationLevel) []const u8 {
```

- type `CudaComputeCapability`

CUDA compute capabilities


```zig
pub const CudaComputeCapability = enum {
```

- pub `inline`

```zig
pub inline fn toString(self: CudaComputeCapability) []const u8 {
```

- pub `inline`

```zig
pub inline fn getMajorVersion(self: CudaComputeCapability) u32 {
```

- type `PTXCompilerOptions`

PTX (Parallel Thread Execution) compiler configuration structure


```zig
pub const PTXCompilerOptions = struct {
```

- fn `validate`

```zig
pub fn validate(self: PTXCompilerOptions) !void {
```

- pub `inline`

```zig
pub inline fn getComputeCapabilityString(self: PTXCompilerOptions) []const u8 {
```

- type `SPIRVCompiler`

SPIR-V compiler for Vulkan, OpenGL, and OpenCL backends


```zig
pub const SPIRVCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: SPIRVCompilerOptions) !*SPIRVCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *SPIRVCompiler) void {
```

- fn `compileShader`

```zig
pub fn compileShader(self: *SPIRVCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
```

- type `MSLCompiler`

Metal Shading Language compiler for Apple platforms


```zig
pub const MSLCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: MSLCompilerOptions) !*MSLCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *MSLCompiler) void {
```

- fn `compileShader`

```zig
pub fn compileShader(self: *MSLCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
```

- type `PTXCompiler`

PTX compiler for NVIDIA CUDA platforms


```zig
pub const PTXCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: PTXCompilerOptions) !*PTXCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *PTXCompiler) void {
```

- fn `compileKernel`

```zig
pub fn compileKernel(self: *PTXCompiler, source: []const u8) ![]u8 {
```

- type `GPUConfig`

GPU renderer configuration with compile-time optimization


```zig
pub const GPUConfig = struct {
```

- fn `validate`

Compile-time validation of configuration


```zig
pub fn validate(comptime config: GPUConfig) void {
```

- type `PowerPreference`

Power preference for GPU selection


```zig
pub const PowerPreference = enum {
```

- pub `inline`

Inline function for quick preference checks


```zig
pub inline fn isHighPerformance(self: PowerPreference) bool {
```

- type `Backend`

GPU backend types with platform detection and compile-time optimization


```zig
pub const Backend = enum {
```

- pub `inline`

```zig
pub inline fn isAvailable(self: Backend) bool {
```

- fn `getBest`

```zig
pub fn getBest() Backend {
```

- pub `inline`

Inline function for performance checks


```zig
pub inline fn requiresGPU(self: Backend) bool {
```

- fn `getPriority`

Get priority score for backend selection (higher is better)


```zig
pub fn getPriority(self: Backend) u8 {
```

- fn `toString`

Convert backend to human-readable string


```zig
pub fn toString(self: Backend) []const u8 {
```

- type `BufferUsage`

GPU buffer usage flags


```zig
pub const BufferUsage = packed struct {
```

- pub `inline`

Inline function for quick usage checks


```zig
pub inline fn isReadable(self: BufferUsage) bool {
```

- pub `inline`

Inline function for quick usage checks


```zig
pub inline fn isWritable(self: BufferUsage) bool {
```

- type `TextureFormat`

GPU texture format


```zig
pub const TextureFormat = enum {
```

- pub `inline`

Inline function for format properties


```zig
pub inline fn getBytesPerPixel(self: TextureFormat) u32 {
```

- pub `inline`

Inline function for format checks


```zig
pub inline fn isFloatFormat(self: TextureFormat) bool {
```

- type `ShaderStage`

Shader stage types


```zig
pub const ShaderStage = enum {
```

- pub `inline`

```zig
pub inline fn toWebGPU(self: ShaderStage) u32 {
```

- type `Color`

Color for clearing operations with inline utility functions


```zig
pub const Color = struct {
```

- const `BLACK`

Compile-time color constants


```zig
pub const BLACK = Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };
```

- const `WHITE`

```zig
pub const WHITE = Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };
```

- const `RED`

```zig
pub const RED = Color{ .r = 1.0, .g = 0.0, .b = 0.0, .a = 1.0 };
```

- const `GREEN`

```zig
pub const GREEN = Color{ .r = 0.0, .g = 1.0, .b = 0.0, .a = 1.0 };
```

- const `BLUE`

```zig
pub const BLUE = Color{ .r = 0.0, .g = 0.0, .b = 1.0, .a = 1.0 };
```

- pub `inline`

Inline utility functions


```zig
pub inline fn fromRGB(r: f32, g: f32, b: f32) Color {
```

- pub `inline`

```zig
pub inline fn lerp(a: Color, b: Color, t: f32) Color {
```

- pub `inline`

Inline function to convert to packed format


```zig
pub inline fn toPackedRGBA(self: Color) u32 {
```

- type `GPUHandle`

GPU resource handle with generation for safety and inline utilities


```zig
pub const GPUHandle = struct {
```

- pub `inline`

```zig
pub inline fn invalid() GPUHandle {
```

- pub `inline`

```zig
pub inline fn isValid(self: GPUHandle) bool {
```

- pub `inline`

Inline function for handle comparison


```zig
pub inline fn equals(self: GPUHandle, other: GPUHandle) bool {
```

- type `MathUtils`

High-performance math utilities with SIMD operations


```zig
pub const MathUtils = struct {
```

- pub `inline`

Inline vector operations


```zig
pub inline fn vectorAdd(comptime T: type, a: []const T, b: []const T, result: []T) void {
```

- pub `inline`

Inline matrix multiplication with cache-friendly access patterns


```zig
pub inline fn matrixMultiply(comptime T: type, a: []const T, b: []const T, result: []T, size: usize) void {
```

- pub `inline`

Inline approximation functions for faster math


```zig
pub inline fn fastSqrt(x: f32) f32 {
```

- pub `inline`

Inline function for fast approximate equality


```zig
pub inline fn approxEqual(a: f32, b: f32) bool {
```

- type `Instance`

```zig
pub const Instance = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Instance {
```

- fn `deinit`

```zig
pub fn deinit(self: *Instance) void {
```

- fn `requestAdapter`

```zig
pub fn requestAdapter(self: *Instance) !*Adapter {
```

- type `Adapter`

```zig
pub const Adapter = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Adapter {
```

- fn `deinit`

```zig
pub fn deinit(self: *Adapter) void {
```

- fn `getName`

```zig
pub fn getName(self: *Adapter) []const u8 {
```

- fn `requestDevice`

```zig
pub fn requestDevice(self: *Adapter) !*Device {
```

- type `Device`

```zig
pub const Device = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Device {
```

- fn `deinit`

```zig
pub fn deinit(self: *Device) void {
```

- fn `createBuffer`

```zig
pub fn createBuffer(self: *Device, size: usize, usage: BufferUsage) !*MockGPU.Buffer {
```

- type `Queue`

```zig
pub const Queue = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Queue {
```

- fn `deinit`

```zig
pub fn deinit(self: *Queue) void {
```

- fn `writeBuffer`

```zig
pub fn writeBuffer(self: *Queue, buffer: *MockGPU.Buffer, data: []const u8) void {
```

- fn `submit`

```zig
pub fn submit(self: *Queue) void {
```

- fn `onSubmittedWorkDone`

```zig
pub fn onSubmittedWorkDone(self: *Queue) void {
```

- type `Buffer`

```zig
pub const Buffer = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator, size: usize) !*MockGPU.Buffer {
```

- fn `deinit`

```zig
pub fn deinit(self: *MockGPU.Buffer) void {
```

- fn `getMappedRange`

```zig
pub fn getMappedRange(self: *MockGPU.Buffer, comptime T: type, offset: usize, length: usize) ?[]T {
```

- fn `unmap`

```zig
pub fn unmap(self: *MockGPU.Buffer) void {
```

- type `GPUContext`

Extended GPU context with compiler support


```zig
pub const GPUContext = struct {
```

- fn `init`

Initialize GPU context with error handling


```zig
pub fn init(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initVulkan`

Initialize Vulkan-specific context


```zig
pub fn initVulkan(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initMetal`

Initialize Metal-specific context


```zig
pub fn initMetal(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initDX12`

Initialize DirectX 12-specific context


```zig
pub fn initDX12(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initOpenGL`

Initialize OpenGL-specific context


```zig
pub fn initOpenGL(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initOpenCL`

Initialize OpenCL-specific context


```zig
pub fn initOpenCL(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initCUDA`

Initialize CUDA-specific context


```zig
pub fn initCUDA(allocator: std.mem.Allocator) !GPUContext {
```

- fn `deinit`

Clean up GPU resources


```zig
pub fn deinit(self: *GPUContext) void {
```

- fn `printDeviceInfo`

Get device info for debugging


```zig
pub fn printDeviceInfo(self: *GPUContext) void {
```

- type `BufferManager`

Buffer manager for simplified GPU buffer operations


```zig
pub const BufferManager = struct {
```

- fn `createBuffer`

Create a GPU buffer with specified type and usage


```zig
pub fn createBuffer(self: BufferManager, comptime T: type, size: u64, usage: BufferUsage) !*MockGPU.Buffer {
```

- fn `writeBuffer`

Write data to GPU buffer


```zig
pub fn writeBuffer(self: BufferManager, buffer: *MockGPU.Buffer, data: anytype) void {
```

- fn `readBuffer`

Read data from GPU buffer


```zig
pub fn readBuffer(self: BufferManager, comptime T: type, buffer: *MockGPU.Buffer, size: u64, allocator: std.mem.Allocator) ![]T {
```

- fn `createBufferWithData`

Create a buffer with initial data


```zig
pub fn createBufferWithData(self: BufferManager, comptime T: type, data: []const T, usage: BufferUsage) !*MockGPU.Buffer {
```

- type `Buffer`

GPU buffer resource with platform abstraction


```zig
pub const Buffer = struct {
```

- fn `init`

```zig
pub fn init(gpu_buffer: *MockGPU.Buffer, size: usize, usage: BufferUsage, id: u64) Buffer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Buffer) void {
```

- fn `map`

```zig
pub fn map(self: *Buffer, allocator: std.mem.Allocator) ![]u8 {
```

- fn `unmap`

```zig
pub fn unmap(self: *Buffer) void {
```

- type `Shader`

Shader resource


```zig
pub const Shader = struct {
```

- fn `compile`

```zig
pub fn compile(allocator: std.mem.Allocator, stage: ShaderStage, source: []const u8) !Shader {
```

- fn `deinit`

```zig
pub fn deinit(self: *Shader) void {
```

- type `BindGroup`

Bind group for resource binding


```zig
pub const BindGroup = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, id: u64) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addBuffer`

```zig
pub fn addBuffer(self: *Self, buffer: Buffer) !void {
```

- type `RendererStats`

Lightweight renderer statistics


```zig
pub const RendererStats = struct {
```

- type `GPURenderer`

Main GPU renderer with cross-platform support and CPU fallbacks


```zig
pub const GPURenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: GPUConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `createBuffer`

Create a GPU buffer with specified usage


```zig
pub fn createBuffer(self: *Self, size: usize, usage: BufferUsage) !u32 {
```

- fn `createBufferWithData`

Convenience: create a buffer initialized with data


```zig
pub fn createBufferWithData(self: *Self, comptime T: type, data: []const T, usage: BufferUsage) !u32 {
```

- fn `destroyBuffer`

Destroy a buffer by handle


```zig
pub fn destroyBuffer(self: *Self, handle: u32) !void {
```

- fn `writeBuffer`

Write raw bytes into a buffer


```zig
pub fn writeBuffer(self: *Self, handle: u32, data: []const u8) !void {
```

- fn `readBuffer`

Read raw bytes from a buffer (copies into a new slice)


```zig
pub fn readBuffer(self: *Self, handle: u32, allocator: std.mem.Allocator) ![]u8 {
```

- fn `copyBuffer`

Copy contents from src to dst (copies min(src.size, dst.size) bytes)


```zig
pub fn copyBuffer(self: *Self, src_handle: u32, dst_handle: u32) !usize {
```

- fn `computeVectorDotBuffers`

Compute vector dot product directly on buffers (length in f32 elements)


```zig
pub fn computeVectorDotBuffers(self: *Self, a_handle: u32, b_handle: u32, length: usize) !f32 {
```

- fn `beginFrame`

Begin frame rendering


```zig
pub fn beginFrame(self: *Self) !void {
```

- fn `endFrame`

End frame rendering


```zig
pub fn endFrame(self: *Self) !void {
```

- fn `clear`

Clear the render target with specified color


```zig
pub fn clear(self: *Self, color: Color) !void {
```

- fn `vectorAdd`

High-performance vector addition with optimized memory patterns


```zig
pub fn vectorAdd(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `matrixMultiply`

High-performance matrix multiplication with cache-optimized implementation


```zig
pub fn matrixMultiply(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `imageProcessing`

High-performance image processing with optimized blur algorithm


```zig
pub fn imageProcessing(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `computeMatrixMultiply`

High-performance matrix multiplication with optimized algorithms


```zig
pub fn computeMatrixMultiply(self: *Self, a: []const f32, b: []const f32, result: []f32, m: u32, n: u32, k: u32) !void {
```

- fn `computeNeuralInference`

Run neural network inference


```zig
pub fn computeNeuralInference(self: *Self, input: []const f32, weights: []const f32, output: []f32) !void {
```

- fn `renderNeuralNetwork`

Render neural network visualization


```zig
pub fn renderNeuralNetwork(self: *Self, neural_engine: anytype) !void {
```

- fn `runExamples`

High-performance example runner with benchmarking


```zig
pub fn runExamples(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `getFPS`

Get current FPS


```zig
pub fn getFPS(self: *Self) f32 {
```

- fn `getFrameCount`

Get frame count


```zig
pub fn getFrameCount(self: *Self) u64 {
```

- fn `getStats`

Get current renderer stats (copy)


```zig
pub fn getStats(self: *Self) RendererStats {
```

- fn `runExamples`

Standalone function for running optimized GPU examples


```zig
pub fn runExamples() !void {
```

- fn `main`

Main function for running the combined GPU examples


```zig
pub fn main() !void {
```

## src\gpu\core\mod.zig

- const `GPURenderer`

```zig
pub const GPURenderer = @import("gpu_renderer.zig").GPURenderer;
```

- const `GPUConfig`

```zig
pub const GPUConfig = @import("gpu_renderer.zig").GPUConfig;
```

- const `GpuError`

```zig
pub const GpuError = @import("gpu_renderer.zig").GpuError;
```

- const `Color`

```zig
pub const Color = @import("gpu_renderer.zig").Color;
```

- const `GPUHandle`

```zig
pub const GPUHandle = @import("gpu_renderer.zig").GPUHandle;
```

- const `Backend`

```zig
pub const Backend = @import("gpu_renderer.zig").Backend;
```

- const `PowerPreference`

```zig
pub const PowerPreference = @import("gpu_renderer.zig").PowerPreference;
```

- const `has_webgpu_support`

```zig
pub const has_webgpu_support = @import("gpu_renderer.zig").has_webgpu_support;
```

- const `GpuBackend`

```zig
pub const GpuBackend = @import("backend.zig").GpuBackend;
```

- const `GpuBackendConfig`

```zig
pub const GpuBackendConfig = @import("backend.zig").GpuBackendConfig;
```

- const `GpuBackendError`

```zig
pub const GpuBackendError = @import("backend.zig").GpuBackend.Error;
```

- const `BatchConfig`

```zig
pub const BatchConfig = @import("backend.zig").BatchConfig;
```

- const `BatchProcessor`

```zig
pub const BatchProcessor = @import("backend.zig").BatchProcessor;
```

- const `GpuStats`

```zig
pub const GpuStats = @import("backend.zig").GpuStats;
```

- const `Db`

```zig
pub const Db = @import("backend.zig").Db;
```

- const `KernelManager`

```zig
pub const KernelManager = @import("../compute/kernels.zig").KernelManager;
```

- const `GPUBackendManager`

```zig
pub const GPUBackendManager = @import("../compute/gpu_backend_manager.zig").GPUBackendManager;
```

- const `CUDADriver`

```zig
pub const CUDADriver = @import("../compute/gpu_backend_manager.zig").CUDADriver;
```

- const `SPIRVCompiler`

```zig
pub const SPIRVCompiler = @import("../compute/gpu_backend_manager.zig").SPIRVCompiler;
```

- const `BackendType`

```zig
pub const BackendType = @import("../compute/gpu_backend_manager.zig").BackendType;
```

- const `HardwareCapabilities`

```zig
pub const HardwareCapabilities = @import("../compute/gpu_backend_manager.zig").HardwareCapabilities;
```

- const `MemoryPool`

```zig
pub const MemoryPool = @import("../memory/memory_pool.zig").MemoryPool;
```

- const `BackendSupport`

```zig
pub const BackendSupport = @import("../compute/kernels.zig").BackendSupport;
```

- const `PerformanceProfiler`

```zig
pub const PerformanceProfiler = @import("../benchmark/benchmarks.zig").PerformanceProfiler;
```

- const `MemoryBandwidthBenchmark`

```zig
pub const MemoryBandwidthBenchmark = @import("../benchmark/benchmarks.zig").MemoryBandwidthBenchmark;
```

- const `ComputeThroughputBenchmark`

```zig
pub const ComputeThroughputBenchmark = @import("../benchmark/benchmarks.zig").ComputeThroughputBenchmark;
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- fn `initDefault`

Initialize the GPU system with default configuration


```zig
pub fn initDefault(allocator: std.mem.Allocator) !*GPURenderer {
```

- fn `isGpuAvailable`

Check if GPU acceleration is available


```zig
pub fn isGpuAvailable() bool {
```

## src\gpu\compute\gpu_backend_manager.zig

- type `BackendType`

GPU Backend Type


```zig
pub const BackendType = enum {
```

- fn `priority`

```zig
pub fn priority(self: BackendType) u8 {
```

- fn `displayName`

```zig
pub fn displayName(self: BackendType) []const u8 {
```

- type `HardwareCapabilities`

Hardware Capabilities


```zig
pub const HardwareCapabilities = struct {
```

- fn `format`

```zig
pub fn format(
```

- type `CUDADriver`

CUDA Driver Interface


```zig
pub const CUDADriver = struct {
```

- type `DeviceProps`

CUDA Device Properties


```zig
pub const DeviceProps = struct {
```

- fn `init`

Initialize CUDA driver


```zig
pub fn init(allocator: std.mem.Allocator) !*CUDADriver {
```

- fn `deinit`

```zig
pub fn deinit(self: *CUDADriver) void {
```

- fn `getDeviceCount`

Get CUDA device count


```zig
pub fn getDeviceCount(self: *CUDADriver) u32 {
```

- fn `getDeviceProperties`

Get device properties


```zig
pub fn getDeviceProperties(self: *CUDADriver, device: u32) !HardwareCapabilities {
```

- fn `setDevice`

Set current device


```zig
pub fn setDevice(self: *CUDADriver, device: u32) !void {
```

- fn `allocMemory`

Allocate CUDA memory


```zig
pub fn allocMemory(self: *CUDADriver, size: usize) !*anyopaque {
```

- fn `freeMemory`

Free CUDA memory


```zig
pub fn freeMemory(self: *CUDADriver, ptr: *anyopaque) void {
```

- fn `copyToDevice`

Copy data to CUDA device


```zig
pub fn copyToDevice(self: *CUDADriver, dst: *anyopaque, src: *const anyopaque, size: usize) !void {
```

- fn `copyFromDevice`

Copy data from CUDA device


```zig
pub fn copyFromDevice(self: *CUDADriver, dst: *anyopaque, src: *const anyopaque, size: usize) !void {
```

- fn `launchKernel`

Launch CUDA kernel


```zig
pub fn launchKernel(
```

- type `SPIRVCompiler`

SPIRV Compiler for Vulkan/OpenCL shaders


```zig
pub const SPIRVCompiler = struct {
```

- type `Target`

Shader Compilation Target


```zig
pub const Target = enum {
```

- type `CompileOptions`

Compilation Options


```zig
pub const CompileOptions = struct {
```

- type `CompileResult`

Compilation Result


```zig
pub const CompileResult = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *CompileResult, allocator: std.mem.Allocator) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SPIRVCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *SPIRVCompiler) void {
```

- fn `compileToSPIRV`

Compile GLSL/HLSL to SPIRV


```zig
pub fn compileToSPIRV(
```

- fn `compileHLSLToSPIRV`

Compile HLSL to SPIRV


```zig
pub fn compileHLSLToSPIRV(
```

- fn `validateSPIRV`

Validate SPIRV code


```zig
pub fn validateSPIRV(self: *SPIRVCompiler, spirv_code: []const u32) !bool {
```

- fn `optimizeSPIRV`

Optimize SPIRV code


```zig
pub fn optimizeSPIRV(
```

- fn `disassembleSPIRV`

Disassemble SPIRV to text format


```zig
pub fn disassembleSPIRV(self: *SPIRVCompiler, spirv_code: []const u32) ![]const u8 {
```

- type `GPUBackendManager`

GPU Backend Manager - Main interface for multi-backend GPU support


```zig
pub const GPUBackendManager = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*GPUBackendManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *GPUBackendManager) void {
```

- fn `selectBackend`

Force selection of a specific backend


```zig
pub fn selectBackend(self: *GPUBackendManager, backend: BackendType) !void {
```

- fn `hasBackend`

Check if a specific backend is available


```zig
pub fn hasBackend(self: *GPUBackendManager, backend: BackendType) bool {
```

- fn `getBackendCapabilities`

Get capabilities for a specific backend


```zig
pub fn getBackendCapabilities(self: *GPUBackendManager, backend: BackendType) !HardwareCapabilities {
```

- fn `compileShader`

Compile shader for current backend


```zig
pub fn compileShader(
```

- fn `printSystemInfo`

Print system information


```zig
pub fn printSystemInfo(self: *GPUBackendManager) void {
```

## src\gpu\compute\kernels.zig

- type `KernelConfig`

Configuration for GPU kernel operations


```zig
pub const KernelConfig = struct {
```

- type `LayerType`

Neural network layer types


```zig
pub const LayerType = enum {
```

- type `ActivationType`

Activation functions


```zig
pub const ActivationType = enum {
```

- type `OptimizerType`

Optimization algorithms


```zig
pub const OptimizerType = enum {
```

- type `KernelManager`

GPU Kernel Manager - Manages specialized compute kernels


```zig
pub const KernelManager = struct {
```

- type `Kernel`

```zig
pub const Kernel = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *GPURenderer) !*KernelManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *KernelManager) void {
```

- fn `createDenseLayer`

Create a dense neural network layer kernel


```zig
pub fn createDenseLayer(
```

- fn `forwardDense`

Forward pass through a dense layer


```zig
pub fn forwardDense(self: *KernelManager, kernel_idx: usize, input_handle: u32, output_handle: u32) !void {
```

- fn `backwardDense`

Backward pass through a dense layer


```zig
pub fn backwardDense(self: *KernelManager, kernel_idx: usize, input_handle: u32, grad_output_handle: u32, grad_input_handle: u32) !void {
```

- fn `createConvLayer`

Create a convolutional layer kernel


```zig
pub fn createConvLayer(
```

- fn `createAttentionLayer`

Create an attention layer kernel for transformer models


```zig
pub fn createAttentionLayer(
```

- fn `matrixMultiplyGPU`

Perform matrix multiplication optimized for GPU


```zig
pub fn matrixMultiplyGPU(
```

- fn `softmaxGPU`

Compute softmax activation function on GPU


```zig
pub fn softmaxGPU(
```

- fn `layerNormGPU`

Compute layer normalization on GPU


```zig
pub fn layerNormGPU(
```

- fn `adamUpdateGPU`

Update weights using Adam optimizer on GPU


```zig
pub fn adamUpdateGPU(
```

- type `MemoryPool`

GPU Memory Pool for efficient memory management


```zig
pub const MemoryPool = struct {
```

- type `BufferInfo`

```zig
pub const BufferInfo = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *GPURenderer) !*MemoryPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate or reuse a buffer from the pool


```zig
pub fn allocBuffer(self: *MemoryPool, size: usize, usage: gpu_renderer.BufferUsage) !u32 {
```

- fn `freeBuffer`

Return a buffer to the pool for reuse


```zig
pub fn freeBuffer(self: *MemoryPool, handle: u32) !void {
```

- fn `cleanup`

Clean up old unused buffers to free memory


```zig
pub fn cleanup(self: *MemoryPool, max_age_ms: i64) !void {
```

- fn `getStats`

Get memory pool statistics


```zig
pub fn getStats(self: *MemoryPool) MemoryStats {
```

- type `MemoryStats`

```zig
pub const MemoryStats = struct {
```

- type `BackendSupport`

GPU Backend Support for multiple APIs


```zig
pub const BackendSupport = struct {
```

- type `Backend`

Supported GPU backends


```zig
pub const Backend = enum {
```

- type `Capabilities`

Backend capabilities


```zig
pub const Capabilities = struct {
```

- fn `init`

Initialize backend support detection


```zig
pub fn init(allocator: std.mem.Allocator) !*BackendSupport {
```

- fn `deinit`

```zig
pub fn deinit(self: *BackendSupport) void {
```

- fn `selectBestBackend`

Select the best available backend


```zig
pub fn selectBestBackend(self: *BackendSupport) ?Backend {
```

- fn `selectBackend`

Force selection of a specific backend


```zig
pub fn selectBackend(self: *BackendSupport, backend: Backend) !void {
```

- fn `detectAvailableBackends`

Detect available GPU backends


```zig
pub fn detectAvailableBackends(self: *BackendSupport) ![]Backend {
```

- fn `getCapabilities`

Get capabilities for a specific backend


```zig
pub fn getCapabilities(self: *BackendSupport, backend: Backend) !Capabilities {
```

- const `GPURenderer`

```zig
pub const GPURenderer = gpu_renderer.GPURenderer;
```

- const `GpuError`

```zig
pub const GpuError = gpu_renderer.GpuError;
```

## src\gpu\benchmark\benchmarks.zig

- type `BenchmarkConfig`

Benchmark configuration


```zig
pub const BenchmarkConfig = struct {
```

- type `OutputFormat`

Output format for benchmark reports


```zig
pub const OutputFormat = enum {
```

- type `WorkloadType`

Types of compute workloads


```zig
pub const WorkloadType = enum {
```

- fn `displayName`

Get the human-readable name for this workload type


```zig
pub fn displayName(self: WorkloadType) []const u8 {
```

- fn `complexityClass`

Get the computational complexity class for this workload


```zig
pub fn complexityClass(self: WorkloadType) []const u8 {
```

- type `BenchmarkResult`

Benchmark result with comprehensive metrics


```zig
pub const BenchmarkResult = struct {
```

- type `ExecutionContext`

Execution context information


```zig
pub const ExecutionContext = struct {
```

- fn `efficiencyScore`

Calculate efficiency score (throughput per watt)


```zig
pub fn efficiencyScore(self: BenchmarkResult) f32 {
```

- fn `stabilityScore`

Calculate performance stability (inverse of coefficient of variation)


```zig
pub fn stabilityScore(self: BenchmarkResult) f32 {
```

- fn `performanceGrade`

Get performance grade based on multiple metrics


```zig
pub fn performanceGrade(self: BenchmarkResult) PerformanceGrade {
```

- type `PerformanceGrade`

```zig
pub const PerformanceGrade = enum {
```

- fn `toString`

```zig
pub fn toString(self: PerformanceGrade) []const u8 {
```

- fn `format`

```zig
pub fn format(
```

- type `PerformanceProfiler`

Performance profiler with advanced metrics and analysis


```zig
pub const PerformanceProfiler = struct {
```

- type `TimingMeasurement`

Detailed timing measurement with hierarchical support


```zig
pub const TimingMeasurement = struct {
```

- fn `duration_ns`

```zig
pub fn duration_ns(self: TimingMeasurement) u64 {
```

- fn `memory_delta`

```zig
pub fn memory_delta(self: TimingMeasurement) i64 {
```

- type `PowerMonitor`

Power consumption monitoring


```zig
pub const PowerMonitor = struct {
```

- type `PowerSample`

```zig
pub const PowerSample = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, sampling_interval_ms: u32) !PowerMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PowerMonitor) void {
```

- fn `recordSample`

```zig
pub fn recordSample(self: *PowerMonitor, sample: PowerSample) !void {
```

- fn `getAveragePower`

```zig
pub fn getAveragePower(self: PowerMonitor) f32 {
```

- fn `getPeakPower`

```zig
pub fn getPeakPower(self: PowerMonitor) f32 {
```

- type `MemoryTracker`

Memory usage tracking


```zig
pub const MemoryTracker = struct {
```

- type `MemoryAllocation`

```zig
pub const MemoryAllocation = struct {
```

- type `MemoryType`

```zig
pub const MemoryType = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryTracker) void {
```

- fn `recordAllocation`

```zig
pub fn recordAllocation(self: *MemoryTracker, size: usize, memory_type: MemoryType) !void {
```

- fn `recordDeallocation`

```zig
pub fn recordDeallocation(self: *MemoryTracker, size: usize) void {
```

- type `ThermalMonitor`

Thermal monitoring for throttling detection


```zig
pub const ThermalMonitor = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, threshold: f32) !ThermalMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *ThermalMonitor) void {
```

- fn `recordTemperature`

```zig
pub fn recordTemperature(self: *ThermalMonitor, temperature: f32) !void {
```

- fn `getAverageTemperature`

```zig
pub fn getAverageTemperature(self: ThermalMonitor) f32 {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*PerformanceProfiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceProfiler) void {
```

- fn `enablePowerMonitoring`

Enable power monitoring with specified sampling interval


```zig
pub fn enablePowerMonitoring(self: *PerformanceProfiler, sampling_interval_ms: u32) !void {
```

- fn `setBaseline`

Set baseline results for comparison


```zig
pub fn setBaseline(self: *PerformanceProfiler, baseline_results: []const BenchmarkResult) !void {
```

- fn `startTiming`

Start timing a specific operation with hierarchical support


```zig
pub fn startTiming(self: *PerformanceProfiler, name: []const u8) !void {
```

- fn `endTiming`

End timing for the current operation


```zig
pub fn endTiming(self: *PerformanceProfiler) !void {
```

- fn `runBenchmarkSuite`

Run comprehensive benchmark suite with enhanced metrics


```zig
pub fn runBenchmarkSuite(self: *PerformanceProfiler, config: BenchmarkConfig) !void {
```

- fn `runWorkloadBenchmark`

Run benchmark for specific workload with comprehensive metrics


```zig
pub fn runWorkloadBenchmark(
```

- fn `generateReport`

Generate comprehensive performance report with statistical analysis


```zig
pub fn generateReport(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `exportToJson`

Export results to JSON with comprehensive metadata


```zig
pub fn exportToJson(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `exportToCsv`

Export results to CSV format


```zig
pub fn exportToCsv(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]const u8 {
```

- type `MemoryBandwidthBenchmark`

Memory bandwidth benchmark with enhanced capabilities


```zig
pub const MemoryBandwidthBenchmark = struct {
```

- type `AccessPattern`

```zig
pub const AccessPattern = enum {
```

- fn `toString`

```zig
pub fn toString(self: AccessPattern) []const u8 {
```

- type `MemoryBenchmarkResult`

```zig
pub const MemoryBenchmarkResult = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*MemoryBandwidthBenchmark {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryBandwidthBenchmark) void {
```

- fn `measureBandwidth`

Measure GPU memory bandwidth


```zig
pub fn measureBandwidth(self: *MemoryBandwidthBenchmark, buffer_size: usize, iterations: u32) !f64 {
```

- fn `runMemoryBenchmark`

Run comprehensive memory benchmark


```zig
pub fn runMemoryBenchmark(self: *MemoryBandwidthBenchmark) !void {
```

- type `ComputeThroughputBenchmark`

GPU Compute Throughput Benchmark


```zig
pub const ComputeThroughputBenchmark = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*ComputeThroughputBenchmark {
```

- fn `deinit`

```zig
pub fn deinit(self: *ComputeThroughputBenchmark) void {
```

- fn `measureComputeThroughput`

Measure compute throughput (FLOPS)


```zig
pub fn measureComputeThroughput(self: *ComputeThroughputBenchmark, workgroup_size: u32, iterations: u32) !f64 {
```

- fn `runComputeBenchmark`

Run compute throughput benchmark


```zig
pub fn runComputeBenchmark(self: *ComputeThroughputBenchmark) !void {
```

## src\gpu\backends\backends.zig

- type `Backend`

Supported GPU backends


```zig
pub const Backend = enum {
```

- fn `toString`

```zig
pub fn toString(self: Backend) []const u8 {
```

- fn `getPriority`

```zig
pub fn getPriority(self: Backend) u8 {
```

- type `Capabilities`

Backend capabilities and features


```zig
pub const Capabilities = struct {
```

- fn `format`

```zig
pub fn format(
```

- const `BackendConfig`

Backend-specific configuration


```zig
pub const BackendConfig = union(Backend) {
```

- type `VulkanConfig`

Vulkan-specific configuration


```zig
pub const VulkanConfig = struct {
```

- type `MetalConfig`

Metal-specific configuration


```zig
pub const MetalConfig = struct {
```

- type `DX12Config`

DirectX 12-specific configuration


```zig
pub const DX12Config = struct {
```

- type `OpenGLConfig`

OpenGL-specific configuration


```zig
pub const OpenGLConfig = struct {
```

- type `CUDAConfig`

CUDA-specific configuration


```zig
pub const CUDAConfig = struct {
```

- type `OpenCLConfig`

OpenCL-specific configuration


```zig
pub const OpenCLConfig = struct {
```

- type `WebGPUConfig`

WebGPU-specific configuration


```zig
pub const WebGPUConfig = struct {
```

- type `CPUConfig`

CPU fallback configuration


```zig
pub const CPUConfig = struct {
```

- type `BackendManager`

Multi-Backend Manager


```zig
pub const BackendManager = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*BackendManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *BackendManager) void {
```

- fn `detectAvailableBackends`

Detect which backends are available on this system


```zig
pub fn detectAvailableBackends(self: *BackendManager) !void {
```

- fn `setDefaultConfigs`

Set default configurations for all backends


```zig
pub fn setDefaultConfigs(self: *BackendManager) !void {
```

- fn `selectBestBackend`

Select the best available backend


```zig
pub fn selectBestBackend(self: *BackendManager) ?Backend {
```

- fn `selectBackend`

Force a specific backend


```zig
pub fn selectBackend(self: *BackendManager, backend: Backend) !void {
```

- fn `getCapabilities`

Get capabilities for a backend


```zig
pub fn getCapabilities(self: *BackendManager, backend: Backend) !Capabilities {
```

- fn `createRenderer`

Create a renderer for the current backend


```zig
pub fn createRenderer(self: *BackendManager, config: gpu_renderer.GPUConfig) !*gpu_renderer.GPURenderer {
```

- type `ShaderCompiler`

Backend-specific shader compiler


```zig
pub const ShaderCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, backend: Backend) !*ShaderCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *ShaderCompiler) void {
```

- fn `compileShader`

Compile shader source to backend-specific format


```zig
pub fn compileShader(self: *ShaderCompiler, source: []const u8, shader_type: enum { vertex, fragment, compute }) ![]const u8 {
```

## src\core\config.zig

- type `FrameworkConfig`

Main framework configuration


```zig
pub const FrameworkConfig = struct {
```

- fn `validate`

Validate the configuration


```zig
pub fn validate(self: FrameworkConfig) FrameworkError!void {
```

- fn `default`

Create a default configuration


```zig
pub fn default() FrameworkConfig {
```

- fn `minimal`

Create a minimal configuration for testing


```zig
pub fn minimal() FrameworkConfig {
```

- fn `production`

Create a production configuration


```zig
pub fn production() FrameworkConfig {
```

- type `AgentConfig`

Agent configuration


```zig
pub const AgentConfig = struct {
```

- fn `validate`

Validate agent configuration


```zig
pub fn validate(self: AgentConfig) FrameworkError!void {
```

- type `PersonaType`

Persona types for agents


```zig
pub const PersonaType = enum {
```

- type `AgentCapabilities`

Agent capabilities


```zig
pub const AgentCapabilities = packed struct(u64) {
```

- fn `validate`

Validate capability dependencies


```zig
pub fn validate(self: AgentCapabilities) bool {
```

- type `WebServerConfig`

Web server configuration


```zig
pub const WebServerConfig = struct {
```

- fn `validate`

Validate web server configuration


```zig
pub fn validate(self: WebServerConfig) FrameworkError!void {
```

- type `DatabaseConfig`

Database configuration


```zig
pub const DatabaseConfig = struct {
```

- fn `validate`

Validate database configuration


```zig
pub fn validate(self: DatabaseConfig) FrameworkError!void {
```

- type `IndexAlgorithm`

Index algorithms for vector database


```zig
pub const IndexAlgorithm = enum {
```

- type `PluginConfig`

Plugin configuration


```zig
pub const PluginConfig = struct {
```

- fn `validate`

Validate plugin configuration


```zig
pub fn validate(self: PluginConfig) FrameworkError!void {
```

- type `ConfigLoader`

Configuration loader for loading configurations from files


```zig
pub const ConfigLoader = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `loadFrameworkConfig`

Load framework configuration from JSON file


```zig
pub fn loadFrameworkConfig(self: *Self, path: []const u8) !FrameworkConfig {
```

- fn `saveFrameworkConfig`

Save framework configuration to JSON file


```zig
pub fn saveFrameworkConfig(self: *Self, config: FrameworkConfig, path: []const u8) !void {
```

- fn `loadAgentConfig`

Load agent configuration from JSON file


```zig
pub fn loadAgentConfig(self: *Self, path: []const u8) !AgentConfig {
```

- fn `saveAgentConfig`

Save agent configuration to JSON file


```zig
pub fn saveAgentConfig(self: *Self, config: AgentConfig, path: []const u8) !void {
```

## src\core\errors.zig

- const `FrameworkError`

Main framework error type


```zig
pub const FrameworkError = error{
```

- type `ErrorCategory`

Error categories for better error handling


```zig
pub const ErrorCategory = enum {
```

- type `ErrorInfo`

Structured error information


```zig
pub const ErrorInfo = struct {
```

- fn `init`

```zig
pub fn init(code: u32, category: ErrorCategory, message: []const u8) ErrorInfo {
```

- fn `withDetails`

```zig
pub fn withDetails(self: ErrorInfo, details: []const u8) ErrorInfo {
```

- fn `withSource`

```zig
pub fn withSource(self: ErrorInfo, source: []const u8) ErrorInfo {
```

- fn `withStackTrace`

```zig
pub fn withStackTrace(self: ErrorInfo, stack_trace: []const u8) ErrorInfo {
```

- type `ErrorCodes`

Error code definitions


```zig
pub const ErrorCodes = struct {
```

- const `OUT_OF_MEMORY`

```zig
pub const OUT_OF_MEMORY = 1001;
```

- const `INVALID_INPUT`

```zig
pub const INVALID_INPUT = 1002;
```

- const `INVALID_CONFIGURATION`

```zig
pub const INVALID_CONFIGURATION = 1003;
```

- const `OPERATION_FAILED`

```zig
pub const OPERATION_FAILED = 1004;
```

- const `TIMEOUT`

```zig
pub const TIMEOUT = 1005;
```

- const `RESOURCE_EXHAUSTED`

```zig
pub const RESOURCE_EXHAUSTED = 1006;
```

- const `UNSUPPORTED_OPERATION`

```zig
pub const UNSUPPORTED_OPERATION = 1007;
```

- const `NETWORK_ERROR`

```zig
pub const NETWORK_ERROR = 2001;
```

- const `CONNECTION_FAILED`

```zig
pub const CONNECTION_FAILED = 2002;
```

- const `CONNECTION_TIMEOUT`

```zig
pub const CONNECTION_TIMEOUT = 2003;
```

- const `CONNECTION_RESET`

```zig
pub const CONNECTION_RESET = 2004;
```

- const `CONNECTION_REFUSED`

```zig
pub const CONNECTION_REFUSED = 2005;
```

- const `DATABASE_ERROR`

```zig
pub const DATABASE_ERROR = 3001;
```

- const `DATABASE_CONNECTION_FAILED`

```zig
pub const DATABASE_CONNECTION_FAILED = 3002;
```

- const `DATABASE_QUERY_FAILED`

```zig
pub const DATABASE_QUERY_FAILED = 3003;
```

- const `DATABASE_TRANSACTION_FAILED`

```zig
pub const DATABASE_TRANSACTION_FAILED = 3004;
```

- const `DATABASE_CORRUPTION`

```zig
pub const DATABASE_CORRUPTION = 3005;
```

- const `AGENT_ERROR`

```zig
pub const AGENT_ERROR = 4001;
```

- const `AGENT_NOT_FOUND`

```zig
pub const AGENT_NOT_FOUND = 4002;
```

- const `AGENT_INITIALIZATION_FAILED`

```zig
pub const AGENT_INITIALIZATION_FAILED = 4003;
```

- const `AGENT_PROCESSING_FAILED`

```zig
pub const AGENT_PROCESSING_FAILED = 4004;
```

- const `AGENT_TIMEOUT`

```zig
pub const AGENT_TIMEOUT = 4005;
```

- const `PLUGIN_ERROR`

```zig
pub const PLUGIN_ERROR = 5001;
```

- const `PLUGIN_NOT_FOUND`

```zig
pub const PLUGIN_NOT_FOUND = 5002;
```

- const `PLUGIN_LOAD_FAILED`

```zig
pub const PLUGIN_LOAD_FAILED = 5003;
```

- const `PLUGIN_INITIALIZATION_FAILED`

```zig
pub const PLUGIN_INITIALIZATION_FAILED = 5004;
```

- const `PLUGIN_EXECUTION_FAILED`

```zig
pub const PLUGIN_EXECUTION_FAILED = 5005;
```

- const `SECURITY_ERROR`

```zig
pub const SECURITY_ERROR = 6001;
```

- const `AUTHENTICATION_FAILED`

```zig
pub const AUTHENTICATION_FAILED = 6002;
```

- const `AUTHORIZATION_FAILED`

```zig
pub const AUTHORIZATION_FAILED = 6003;
```

- const `INVALID_TOKEN`

```zig
pub const INVALID_TOKEN = 6004;
```

- const `TOKEN_EXPIRED`

```zig
pub const TOKEN_EXPIRED = 6005;
```

- const `VALIDATION_ERROR`

```zig
pub const VALIDATION_ERROR = 7001;
```

- const `INVALID_FORMAT`

```zig
pub const INVALID_FORMAT = 7002;
```

- const `INVALID_SCHEMA`

```zig
pub const INVALID_SCHEMA = 7003;
```

- const `INVALID_DATA`

```zig
pub const INVALID_DATA = 7004;
```

- const `INVALID_PARAMETER`

```zig
pub const INVALID_PARAMETER = 7005;
```

- const `SYSTEM_ERROR`

```zig
pub const SYSTEM_ERROR = 8001;
```

- const `FILE_NOT_FOUND`

```zig
pub const FILE_NOT_FOUND = 8002;
```

- const `PERMISSION_DENIED`

```zig
pub const PERMISSION_DENIED = 8003;
```

- const `DISK_FULL`

```zig
pub const DISK_FULL = 8004;
```

- const `SYSTEM_OVERLOAD`

```zig
pub const SYSTEM_OVERLOAD = 8005;
```

- type `ErrorHandler`

Error handler for managing and processing errors


```zig
pub const ErrorHandler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `handleError`

Handle an error


```zig
pub fn handleError(self: *Self, error_info: ErrorInfo) void {
```

- fn `logError`

Log an error


```zig
pub fn logError(self: *Self, error_info: ErrorInfo) void {
```

- fn `addErrorCallback`

Add error callback


```zig
pub fn addErrorCallback(self: *Self, callback: ErrorCallback) !void {
```

- fn `removeErrorCallback`

Remove error callback


```zig
pub fn removeErrorCallback(self: *Self, index: usize) void {
```

- fn `getErrorLog`

Get error log


```zig
pub fn getErrorLog(self: *const Self) []const ErrorInfo {
```

- fn `clearErrorLog`

Clear error log


```zig
pub fn clearErrorLog(self: *Self) void {
```

- fn `getErrorStats`

Get error statistics


```zig
pub fn getErrorStats(self: *const Self) ErrorStats {
```

- type `ErrorCallback`

Error callback function type


```zig
pub const ErrorCallback = struct {
```

- type `ErrorStats`

Error statistics


```zig
pub const ErrorStats = struct {
```

- type `ErrorContext`

Error context for providing additional error information


```zig
pub const ErrorContext = struct {
```

- fn `init`

```zig
pub fn init(operation: []const u8, component: []const u8) ErrorContext {
```

- fn `withUser`

```zig
pub fn withUser(self: ErrorContext, user_id: []const u8) ErrorContext {
```

- fn `withSession`

```zig
pub fn withSession(self: ErrorContext, session_id: []const u8) ErrorContext {
```

- fn `withRequest`

```zig
pub fn withRequest(self: ErrorContext, request_id: []const u8) ErrorContext {
```

- type `ErrorRecovery`

Error recovery strategies


```zig
pub const ErrorRecovery = enum {
```

- type `ErrorRecoveryHandler`

Error recovery handler


```zig
pub const ErrorRecoveryHandler = struct {
```

- fn `init`

```zig
pub fn init(strategy: ErrorRecovery) ErrorRecoveryHandler {
```

- fn `withRetries`

```zig
pub fn withRetries(self: ErrorRecoveryHandler, max_retries: u32, delay_ms: u32) ErrorRecoveryHandler {
```

- fn `withFallback`

```zig
pub fn withFallback(self: ErrorRecoveryHandler, fallback: *const fn () anyerror!void) ErrorRecoveryHandler {
```

- fn `frameworkErrorToInfo`

Utility functions for error handling
Convert framework error to error info


```zig
pub fn frameworkErrorToInfo(err: FrameworkError, message: []const u8) ErrorInfo {
```

- fn `getErrorCategory`

Get error category from error code


```zig
pub fn getErrorCategory(code: u32) ErrorCategory {
```

- fn `getErrorMessage`

Get error message from error code


```zig
pub fn getErrorMessage(code: u32) []const u8 {
```

## src\core\framework.zig

- type `Framework`

Main framework instance


```zig
pub const Framework = struct {
```

- fn `init`

Initialize the framework with the given configuration


```zig
pub fn init(allocator: std.mem.Allocator, framework_config: FrameworkConfig) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the framework and clean up resources


```zig
pub fn deinit(self: *Self) void {
```

- fn `getState`

Get the current framework state


```zig
pub fn getState(self: *const Self) FrameworkState {
```

- fn `getConfig`

Get framework configuration


```zig
pub fn getConfig(self: *const Self) FrameworkConfig {
```

- fn `getComponents`

Get component registry


```zig
pub fn getComponents(self: *Self) *ComponentRegistry {
```

- fn `getMetrics`

Get metrics collector


```zig
pub fn getMetrics(self: *Self) *MetricsCollector {
```

- fn `getLogger`

Get logger


```zig
pub fn getLogger(self: *Self) *Logger {
```

- fn `registerComponent`

Register a component with the framework


```zig
pub fn registerComponent(self: *Self, name: []const u8, component: anytype) !void {
```

- fn `getComponent`

Get a registered component


```zig
pub fn getComponent(self: *Self, name: []const u8) ?anyopaque {
```

- fn `healthCheck`

Health check for the framework


```zig
pub fn healthCheck(self: *const Self) HealthStatus {
```

- type `ComponentRegistry`

Component registry for managing framework components


```zig
pub const ComponentRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `register`

```zig
pub fn register(self: *Self, name: []const u8, component: anytype) !void {
```

- fn `get`

```zig
pub fn get(self: *Self, name: []const u8) ?*anyopaque {
```

- fn `unregister`

```zig
pub fn unregister(self: *Self, name: []const u8) bool {
```

- fn `list`

```zig
pub fn list(self: *const Self) []const []const u8 {
```

- type `HealthStatus`

Health status for the framework


```zig
pub const HealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *HealthStatus) void {
```

- type `ComponentHealth`

Component health status


```zig
pub const ComponentHealth = struct {
```

- type `HealthLevel`

Health levels


```zig
pub const HealthLevel = enum {
```

- type `MetricsCollector`

Metrics collector for framework metrics


```zig
pub const MetricsCollector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(self: *Self, name: []const u8, value: f64, tags: ?[]const []const u8) !void {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *const Self, name: []const u8) ?Metric {
```

- fn `getAllMetrics`

```zig
pub fn getAllMetrics(self: *const Self) []const Metric {
```

- type `Metric`

Individual metric


```zig
pub const Metric = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- type `Logger`

Logger for framework logging


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: std.log.Level) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `info`

```zig
pub fn info(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `debug`

```zig
pub fn debug(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `warn`

```zig
pub fn warn(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `err`

```zig
pub fn err(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) ErrorHandler {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, _config: FrameworkConfig) ConfigManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) LifecycleManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*GPUBackendManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SIMDOperations {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*MemoryTracker {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceProfiler {
```

## src\core\lifecycle.zig

- type `FrameworkState`

Framework lifecycle states


```zig
pub const FrameworkState = enum {
```

- type `FrameworkStateMethods`

Framework lifecycle state methods


```zig
pub const FrameworkStateMethods = struct {
```

- fn `canTransitionTo`

Check if the framework can transition to a new state


```zig
pub fn canTransitionTo(self: FrameworkState, new_state: FrameworkState) bool {
```

- fn `getDescription`

Get human-readable state description


```zig
pub fn getDescription(self: FrameworkState) []const u8 {
```

- type `ComponentState`

Component lifecycle states


```zig
pub const ComponentState = enum {
```

- fn `canTransitionTo`

Check if the component can transition to a new state


```zig
pub fn canTransitionTo(self: ComponentState, new_state: ComponentState) bool {
```

- fn `getDescription`

Get human-readable state description


```zig
pub fn getDescription(self: ComponentState) []const u8 {
```

- type `ComponentLifecycle`

Component lifecycle interface


```zig
pub const ComponentLifecycle = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8, dependencies: []const []const u8) Self {
```

- fn `withInit`

```zig
pub fn withInit(self: Self, fn_init: *const fn (allocator: std.mem.Allocator) anyerror!void) Self {
```

- fn `withStart`

```zig
pub fn withStart(self: Self, fn_start: *const fn () anyerror!void) Self {
```

- fn `withStop`

```zig
pub fn withStop(self: Self, fn_stop: *const fn () anyerror!void) Self {
```

- fn `withDeinit`

```zig
pub fn withDeinit(self: Self, fn_deinit: *const fn () void) Self {
```

- fn `withHealthCheck`

```zig
pub fn withHealthCheck(self: Self, fn_health_check: *const fn () bool) Self {
```

- fn `initialize`

```zig
pub fn initialize(self: *Self, allocator: std.mem.Allocator) FrameworkError!void {
```

- fn `start`

```zig
pub fn start(self: *Self) FrameworkError!void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) FrameworkError!void {
```

- fn `deinitialize`

```zig
pub fn deinitialize(self: *Self) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const Self) bool {
```

- type `LifecycleManager`

Lifecycle manager for managing component lifecycles


```zig
pub const LifecycleManager = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `registerComponent`

Register a component with the lifecycle manager


```zig
pub fn registerComponent(self: *Self, component: ComponentLifecycle) FrameworkError!void {
```

- fn `unregisterComponent`

Unregister a component from the lifecycle manager


```zig
pub fn unregisterComponent(self: *Self, name: []const u8) bool {
```

- fn `getComponent`

Get a component by name


```zig
pub fn getComponent(self: *Self, name: []const u8) ?*ComponentLifecycle {
```

- fn `initializeAll`

Initialize all components in dependency order


```zig
pub fn initializeAll(self: *Self) FrameworkError!void {
```

- fn `startAll`

Start all components


```zig
pub fn startAll(self: *Self) FrameworkError!void {
```

- fn `stopAll`

Stop all components in reverse order


```zig
pub fn stopAll(self: *Self) FrameworkError!void {
```

- fn `deinitializeAll`

Deinitialize all components


```zig
pub fn deinitializeAll(self: *Self) void {
```

- fn `getComponentStates`

Get component states


```zig
pub fn getComponentStates(self: *const Self) std.StringHashMap(ComponentState) {
```

- fn `healthCheckAll`

Health check all components


```zig
pub fn healthCheckAll(self: *const Self) std.StringHashMap(bool) {
```

- type `LifecycleEvent`

Lifecycle event types


```zig
pub const LifecycleEvent = enum {
```

- type `LifecycleEventHandler`

Lifecycle event handler


```zig
pub const LifecycleEventHandler = struct {
```

- type `LifecycleEventManager`

Lifecycle event manager


```zig
pub const LifecycleEventManager = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `registerHandler`

Register an event handler


```zig
pub fn registerHandler(self: *Self, handler: LifecycleEventHandler) !void {
```

- fn `unregisterHandler`

Unregister an event handler


```zig
pub fn unregisterHandler(self: *Self, index: usize) void {
```

- fn `emitEvent`

Emit a lifecycle event


```zig
pub fn emitEvent(self: *const Self, event: LifecycleEvent, component_name: ?[]const u8, data: ?*anyopaque) void {
```

- type `LifecycleStats`

Lifecycle statistics


```zig
pub const LifecycleStats = struct {
```

- fn `update`

```zig
pub fn update(self: *LifecycleStats, components: std.StringHashMap(ComponentLifecycle)) void {
```

## src\core\mod.zig

- const `FrameworkError`

Framework-wide error set for consistent error handling


```zig
pub const FrameworkError = error{
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- const `ArrayList`

```zig
pub const ArrayList = std.ArrayList;
```

- type `Vector`

SIMD vector types with automatic detection


```zig
pub const Vector = struct {
```

- const `f32x4`

4-float SIMD vector


```zig
pub const f32x4 = if (@hasDecl(std.simd, "f32x4")) std.simd.f32x4 else @Vector(4, f32);
```

- const `f32x8`

8-float SIMD vector


```zig
pub const f32x8 = if (@hasDecl(std.simd, "f32x8")) std.simd.f32x8 else @Vector(8, f32);
```

- const `f32x16`

16-float SIMD vector


```zig
pub const f32x16 = if (@hasDecl(std.simd, "f32x16")) std.simd.f32x16 else @Vector(16, f32);
```

- fn `load`

Load vector from slice (compatible with both std.simd and @Vector)


```zig
pub fn load(comptime T: type, data: []const f32) T {
```

- fn `store`

Store vector to slice (compatible with both std.simd and @Vector)


```zig
pub fn store(data: []f32, vec: anytype) void {
```

- fn `splat`

Create splat vector (compatible with both std.simd and @Vector)


```zig
pub fn splat(comptime T: type, value: f32) T {
```

- fn `isSimdAvailable`

Check if SIMD is available for a given vector size


```zig
pub fn isSimdAvailable(comptime size: usize) bool {
```

- fn `getOptimalSize`

Get optimal SIMD vector size for given dimension


```zig
pub fn getOptimalSize(dimension: usize) usize {
```

- type `VectorOps`

SIMD-optimized vector operations


```zig
pub const VectorOps = struct {
```

- fn `distance`

Calculate Euclidean distance between two vectors using SIMD


```zig
pub fn distance(a: []const f32, b: []const f32) f32 {
```

- fn `cosineSimilarity`

Calculate cosine similarity between two vectors


```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
```

- fn `add`

Add two vectors using SIMD


```zig
pub fn add(result: []f32, a: []const f32, b: []const f32) void {
```

- fn `dotProduct`

Calculate dot product of two vectors


```zig
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- fn `scale`

Multiply vector by scalar using SIMD


```zig
pub fn scale(result: []f32, vector: []const f32, scalar: f32) void {
```

- fn `normalize`

Normalize vector to unit length


```zig
pub fn normalize(result: []f32, vector: []const f32) void {
```

- type `MatrixOps`

Matrix operations with SIMD acceleration


```zig
pub const MatrixOps = struct {
```

- fn `matrixVectorMultiply`

Matrix-vector multiplication: result = matrix * vector


```zig
pub fn matrixVectorMultiply(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) void {
```

- type `PerformanceMonitor`

Performance monitoring for SIMD operations


```zig
pub const PerformanceMonitor = struct {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *PerformanceMonitor, duration_ns: u64, used_simd: bool) void {
```

- fn `getAverageTime`

```zig
pub fn getAverageTime(self: *const PerformanceMonitor) f64 {
```

- fn `getSimdUsageRate`

```zig
pub fn getSimdUsageRate(self: *const PerformanceMonitor) f64 {
```

- fn `printStats`

```zig
pub fn printStats(self: *const PerformanceMonitor) void {
```

- fn `getPerformanceMonitor`

Get global performance monitor


```zig
pub fn getPerformanceMonitor() *PerformanceMonitor {
```

- type `Features`

Compile-time feature detection


```zig
pub const Features = struct {
```

- const `has_simd`

```zig
pub const has_simd = @hasDecl(std.simd, "f32x4");
```

- const `has_avx`

```zig
pub const has_avx = @import("builtin").target.cpu.arch == .x86_64 and
```

- const `has_neon`

```zig
pub const has_neon = @import("builtin").target.cpu.arch == .aarch64 and
```

- type `Validation`

Common validation utilities


```zig
pub const Validation = struct {
```

- fn `validateDimensions`

Validate that dimensions match


```zig
pub fn validateDimensions(expected: usize, actual: usize) FrameworkError!void {
```

- fn `validateNonEmpty`

Validate that slice is not empty


```zig
pub fn validateNonEmpty(slice: anytype) FrameworkError!void {
```

- fn `validateAlignment`

Validate alignment requirements


```zig
pub fn validateAlignment(ptr: anytype, alignment: usize) FrameworkError!void {
```

- const `f32x4`

```zig
pub const f32x4 = Vector.f32x4;
```

- const `f32x8`

```zig
pub const f32x8 = Vector.f32x8;
```

- const `f32x16`

```zig
pub const f32x16 = Vector.f32x16;
```

- const `distance`

```zig
pub const distance = VectorOps.distance;
```

- const `cosineSimilarity`

```zig
pub const cosineSimilarity = VectorOps.cosineSimilarity;
```

- const `add`

```zig
pub const add = VectorOps.add;
```

- const `dotProduct`

```zig
pub const dotProduct = VectorOps.dotProduct;
```

- const `scale`

```zig
pub const scale = VectorOps.scale;
```

- const `matrixVectorMultiply`

```zig
pub const matrixVectorMultiply = MatrixOps.matrixVectorMultiply;
```

## src\connectors\mod.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- type `ProviderType`

```zig
pub const ProviderType = enum { ollama, openai };
```

- type `OllamaConfig`

```zig
pub const OllamaConfig = struct {
```

- type `OpenAIConfig`

```zig
pub const OpenAIConfig = struct {
```

- const `ProviderConfig`

```zig
pub const ProviderConfig = union(ProviderType) {
```

- const `plugin`

```zig
pub const plugin = @import("plugin.zig");
```

- const `ConnectorsError`

```zig
pub const ConnectorsError = error{
```

- fn `embedText`

```zig
pub fn embedText(allocator: Allocator, config: ProviderConfig, text: []const u8) ConnectorsError![]f32 {
```

## src\connectors\ollama.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- fn `embedText`

```zig
pub fn embedText(allocator: Allocator, host: []const u8, model: []const u8, text: []const u8) ![]f32 {
```

## src\connectors\openai.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- const `Error`

Errors that can occur during OpenAI API operations


```zig
pub const Error = error{
```

- fn `embedText`

Embeds the given text using the OpenAI embeddings API

Args:
- allocator: Memory allocator for dynamic allocations
- base_url: Base URL for the OpenAI API (e.g., "https://api.openai.com/v1")
- api_key: OpenAI API key for authentication
- model: The model to use for embeddings (e.g., "text-embedding-ada-002")
- text: The text to embed

Returns:
- A slice of f32 values representing the embedding vector
- The caller owns the returned memory and must free it

Errors:
- MissingApiKey: If the api_key is empty
- NetworkError: If there's a network communication error or non-200 response
- InvalidResponse: If the API response format is unexpected
- OutOfMemory: If memory allocation fails


```zig
pub fn embedText(allocator: Allocator, base_url: []const u8, api_key: []const u8, model: []const u8, text: []const u8) Error![]f32 {
```

## src\connectors\plugin.zig

- type `EmbeddingApi`

```zig
pub const EmbeddingApi = extern struct {
```

- var `PLUGIN_INTERFACE`

```zig
pub var PLUGIN_INTERFACE: iface.PluginInterface = .{
```

- fn `abi_plugin_create`

```zig
pub fn abi_plugin_create() callconv(.c) ?*const iface.PluginInterface {
```

- fn `getInterface`

```zig
pub fn getInterface() *const iface.PluginInterface {
```

## src\cli\main.zig

- fn `main`

```zig
pub fn main() !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *TrainingData) void {
```

## src\ai\activation.zig

- type `ActivationType`

Available activation function types with detailed mathematical definitions


```zig
pub const ActivationType = enum {
```

- type `ActivationConfig`

Activation function configuration


```zig
pub const ActivationConfig = struct {
```

- type `ActivationProcessor`

High-performance activation function processor


```zig
pub const ActivationProcessor = struct {
```

- fn `init`

```zig
pub fn init(config: ActivationConfig) ActivationProcessor {
```

- fn `activate`

Apply activation function to a single value


```zig
pub fn activate(self: *const ActivationProcessor, x: f32) f32 {
```

- fn `activateBatch`

Apply activation function to an array (with SIMD optimization)


```zig
pub fn activateBatch(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
```

- fn `derivative`

Compute derivative of activation function


```zig
pub fn derivative(self: *const ActivationProcessor, x: f32, y: f32) f32 {
```

- fn `derivativeBatch`

Batch derivative computation


```zig
pub fn derivativeBatch(self: *const ActivationProcessor, output: []f32, input: []const f32, forward_output: []const f32) void {
```

- type `ActivationRegistry`

Activation function registry for dynamic dispatch


```zig
pub const ActivationRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) ActivationRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *ActivationRegistry) void {
```

- fn `register`

```zig
pub fn register(self: *ActivationRegistry, name: []const u8, func: ActivationFn) !void {
```

- fn `get`

```zig
pub fn get(self: *ActivationRegistry, name: []const u8) ?ActivationFn {
```

## src\ai\agent.zig

- const `AgentError`

Agent-specific error types


```zig
pub const AgentError = error{
```

- type `BackendType`

Backend provider types


```zig
pub const BackendType = enum {
```

- fn `getDefaultCapabilities`

```zig
pub fn getDefaultCapabilities(self: BackendType) AgentCapabilities {
```

- type `BackendConfig`

Backend configuration


```zig
pub const BackendConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: BackendConfig) AgentError!void {
```

- type `PersonaType`

Agent personas with enhanced characteristics


```zig
pub const PersonaType = enum {
```

- fn `getDescription`

Get persona description


```zig
pub fn getDescription(self: PersonaType) []const u8 {
```

- fn `getScoring`

Get persona scoring weights for different query types


```zig
pub fn getScoring(self: PersonaType) PersonaScoring {
```

- type `PersonaScoring`

Persona scoring characteristics


```zig
pub const PersonaScoring = struct {
```

- type `AgentState`

Agent state with enhanced state management


```zig
pub const AgentState = enum(u8) {
```

- fn `canTransitionTo`

Validate state transitions


```zig
pub fn canTransitionTo(from: AgentState, to: AgentState) bool {
```

- type `AgentCapabilities`

Agent capabilities with packed representation


```zig
pub const AgentCapabilities = packed struct(u32) {
```

- fn `validate`

Validate capability dependencies


```zig
pub fn validate(self: AgentCapabilities) bool {
```

- type `MessageRole`

Message role in conversation


```zig
pub const MessageRole = enum {
```

- type `Message`

Conversation message with metadata


```zig
pub const Message = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, role: MessageRole, content: []const u8) !Message {
```

- fn `deinit`

```zig
pub fn deinit(self: Message, allocator: Allocator) void {
```

- type `MemoryEntry`

Advanced memory entry with vectorized operations


```zig
pub const MemoryEntry = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, content: []const u8, importance: f32) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self, allocator: Allocator) void {
```

- fn `updateAccess`

```zig
pub fn updateAccess(self: *Self, enable_simd: bool) void {
```

- fn `computeSimilarity`

Compute similarity using SIMD if available


```zig
pub fn computeSimilarity(self: *const Self, other_embedding: []const f32, use_simd: bool) f32 {
```

- type `ThreadPool`

Thread pool for concurrent operations


```zig
pub const ThreadPool = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, thread_count: u32) !*ThreadPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *ThreadPool) void {
```

- fn `submit`

```zig
pub fn submit(self: *ThreadPool, func: *const fn (*anyopaque) void, data: *anyopaque) !void {
```

- type `AgentAllocator`

Custom allocator optimized for agent operations


```zig
pub const AgentAllocator = struct {
```

- fn `init`

```zig
pub fn init(backing_allocator: Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `allocator`

```zig
pub fn allocator(self: *Self) Allocator {
```

- fn `createMemoryEntry`

```zig
pub fn createMemoryEntry(self: *Self) !*MemoryEntry {
```

- fn `destroyMemoryEntry`

```zig
pub fn destroyMemoryEntry(self: *Self, entry: *MemoryEntry) void {
```

- fn `createMessage`

```zig
pub fn createMessage(self: *Self) !*Message {
```

- fn `destroyMessage`

```zig
pub fn destroyMessage(self: *Self, message: *Message) void {
```

- type `AgentConfig`

Enhanced agent configuration


```zig
pub const AgentConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: AgentConfig) AgentError!void {
```

- type `PerformanceStats`

Performance statistics with comprehensive metrics


```zig
pub const PerformanceStats = struct {
```

- fn `updateResponseTime`

```zig
pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
```

- fn `recordSuccess`

```zig
pub fn recordSuccess(self: *PerformanceStats, persona: PersonaType) void {
```

- fn `recordFailure`

```zig
pub fn recordFailure(self: *PerformanceStats) void {
```

- fn `getSuccessRate`

```zig
pub fn getSuccessRate(self: *const PerformanceStats) f32 {
```

- fn `recordSimdOperation`

```zig
pub fn recordSimdOperation(self: *PerformanceStats) void {
```

- fn `recordVectorOperation`

```zig
pub fn recordVectorOperation(self: *PerformanceStats) void {
```

- type `Profiler`

Profiler for performance monitoring


```zig
pub const Profiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, enabled: bool) Profiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *Profiler) void {
```

- fn `startOperation`

```zig
pub fn startOperation(self: *Profiler, operation: []const u8) i64 {
```

- fn `endOperation`

```zig
pub fn endOperation(self: *Profiler, start_time: i64, operation: []const u8, memory_used: usize) !void {
```

- fn `getReport`

```zig
pub fn getReport(self: *const Profiler) []const ProfileSample {
```

- type `AgentCache`

Cache for frequently accessed data


```zig
pub const AgentCache = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, max_size: usize) AgentCache {
```

- fn `deinit`

```zig
pub fn deinit(self: *AgentCache) void {
```

- fn `get`

```zig
pub fn get(self: *AgentCache, key: u64) ?[]const u8 {
```

- fn `put`

```zig
pub fn put(self: *AgentCache, key: u64, data: []const u8, importance: f32) !void {
```

- type `Agent`

Unified AI Agent with enhanced capabilities


```zig
pub const Agent = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `processInput`

Process user input with intelligent persona routing


```zig
pub fn processInput(self: *Self, input: []const u8) AgentError![]const u8 {
```

- fn `storeMemory`

Store information in agent memory with enhanced features


```zig
pub fn storeMemory(self: *Self, content: []const u8, importance: f32) AgentError!void {
```

- fn `getState`

Get current agent state safely


```zig
pub fn getState(self: *const Self) AgentState {
```

- fn `getStats`

Get comprehensive performance statistics


```zig
pub fn getStats(self: *const Self) PerformanceStats {
```

- fn `setPersona`

Set persona explicitly


```zig
pub fn setPersona(self: *Self, persona: PersonaType) void {
```

- fn `getPersona`

Get current persona


```zig
pub fn getPersona(self: *const Self) PersonaType {
```

- fn `clearHistory`

Clear conversation history


```zig
pub fn clearHistory(self: *Self) void {
```

- fn `clearMemory`

Clear memory


```zig
pub fn clearMemory(self: *Self) void {
```

- fn `clearCache`

Clear cache


```zig
pub fn clearCache(self: *Self) void {
```

- fn `getProfilingReport`

Get profiling report


```zig
pub fn getProfilingReport(self: *const Self) []const Profiler.ProfileSample {
```

- fn `warmUp`

Warm up the agent (pre-allocate resources, load models, etc.)


```zig
pub fn warmUp(self: *Self) AgentError!void {
```

- fn `benchmark`

Run benchmarks to assess performance


```zig
pub fn benchmark(self: *Self) AgentError!void {
```

## src\ai\dynamic.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `RouterError`

Router-specific error types


```zig
pub const RouterError = error{
```

- type `Persona`

Represents a single conversational persona with basic metrics.


```zig
pub const Persona = struct {
```

- fn `validate`

```zig
pub fn validate(self: Persona) bool {
```

- type `Query`

Represents a user query with context information.


```zig
pub const Query = struct {
```

- fn `validate`

```zig
pub fn validate(self: Query) RouterError!void {
```

- type `TransformerModel`

Placeholder transformer model used to evaluate personas.


```zig
pub const TransformerModel = struct {
```

- fn `scorePersona`

Score a persona for the given query.


```zig
pub fn scorePersona(self: TransformerModel, persona: Persona, query: Query) RouterError!f32 {
```

- type `DynamicPersonaRouter`

Router selects the best persona for a given query.


```zig
pub const DynamicPersonaRouter = struct {
```

- fn `init`

```zig
pub fn init(personas: []const Persona) RouterError!DynamicPersonaRouter {
```

- fn `select`

Select a persona based on query context and user needs.


```zig
pub fn select(self: DynamicPersonaRouter, query: Query) RouterError!Persona {
```

- fn `example`

Example usage of the router.


```zig
pub fn example() !void {
```

## src\ai\enhanced_agent.zig

- type `AgentState`

Agent state management with compile-time validation


```zig
pub const AgentState = enum(u8) {
```

- fn `canTransitionTo`

Compile-time state transition validation


```zig
pub fn canTransitionTo(comptime from: AgentState, comptime to: AgentState) bool {
```

- type `AgentCapabilities`

Agent capabilities with packed struct for memory efficiency


```zig
pub const AgentCapabilities = packed struct(u32) {
```

- fn `validateCapabilities`

Compile-time capability validation


```zig
pub fn validateCapabilities(comptime caps: AgentCapabilities) bool {
```

- type `AgentConfig`

Enhanced agent configuration with compile-time optimizations


```zig
pub const AgentConfig = struct {
```

- fn `validate`

Compile-time validation of configuration


```zig
pub fn validate(comptime config: AgentConfig) !void {
```

- type `MemoryEntry`

Advanced memory entry with vectorized operations support


```zig
pub const MemoryEntry = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, content: []const u8, importance: f32) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self, allocator: Allocator) void {
```

- fn `addTag`

```zig
pub fn addTag(self: *Self, allocator: Allocator, key: []const u8, value: []const u8) !void {
```

- fn `updateAccess`

Update access statistics with SIMD-optimized importance calculation


```zig
pub fn updateAccess(self: *Self, enable_simd: bool) void {
```

- fn `init`

```zig
pub fn init(base_allocator: Allocator, pool_size: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `allocator`

```zig
pub fn allocator(self: *Self) Allocator {
```

- fn `getStats`

Get current memory usage statistics


```zig
pub fn getStats(self: *Self) struct { total_allocated: usize, peak_allocated: usize, free_blocks: usize } {
```

- fn `reset`

Reset the allocator, freeing all allocations


```zig
pub fn reset(self: *Self) void {
```

- fn `defragment`

Defragment the free list by merging adjacent blocks


```zig
pub fn defragment(self: *Self) void {
```

- type `EnhancedAgent`

Enhanced AI Agent with advanced performance optimizations


```zig
pub const EnhancedAgent = struct {
```

- type `PerformanceStats`

Enhanced performance tracking with SIMD support


```zig
pub const PerformanceStats = struct {
```

- fn `updateResponseTime`

```zig
pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
```

- fn `recordSuccess`

```zig
pub fn recordSuccess(self: *PerformanceStats) void {
```

- fn `recordFailure`

```zig
pub fn recordFailure(self: *PerformanceStats) void {
```

- fn `init`

Initialize enhanced agent with compile-time validation


```zig
pub fn init(allocator: Allocator, comptime config: AgentConfig) !*Self {
```

- fn `deinit`

Deinitialize agent with proper cleanup


```zig
pub fn deinit(self: *Self) void {
```

- fn `processInput`

Process user input with enhanced error handling and concurrency


```zig
pub fn processInput(self: *Self, input: []const u8) ![]const u8 {
```

- fn `storeMemory`

Enhanced memory storage with SIMD optimization


```zig
pub fn storeMemory(self: *Self, content: []const u8, importance: f32) !void {
```

- fn `getStats`

Enhanced statistics with detailed metrics


```zig
pub fn getStats(self: *const Self) PerformanceStats {
```

- fn `searchMemory`

Enhanced semantic memory search with vector similarity


```zig
pub fn searchMemory(self: *const Self, query: []const u8) ![]MemoryEntry {
```

- fn `learn`

Enhanced learning with reinforcement-based importance adjustment


```zig
pub fn learn(self: *Self, input: []const u8, feedback: f32) !void {
```

- fn `getState`

Get current agent state safely


```zig
pub fn getState(self: *const Self) AgentState {
```

- fn `healthCheck`

Health check for agent status


```zig
pub fn healthCheck(self: *const Self) struct { healthy: bool, issues: []const []const u8 } {
```

## src\ai\layer.zig

- type `LayerType`

Neural network layer types with enhanced coverage


```zig
pub const LayerType = enum {
```

- type `WeightInit`

Weight initialization strategies with enhanced coverage


```zig
pub const WeightInit = enum {
```

- type `PaddingMode`

Padding modes for convolution layers


```zig
pub const PaddingMode = enum {
```

- type `PoolingMode`

Pooling modes for pooling layers


```zig
pub const PoolingMode = enum {
```

- type `AttentionType`

Attention mechanisms


```zig
pub const AttentionType = enum {
```

- type `RNNCellType`

RNN cell types


```zig
pub const RNNCellType = enum {
```

- type `Regularization`

Advanced regularization configuration with comprehensive techniques


```zig
pub const Regularization = struct {
```

- type `LayerConfig`

Enhanced layer configuration structure with comprehensive parameters


```zig
pub const LayerConfig = struct {
```

- type `Layer`

Enhanced neural network layer with comprehensive functionality


```zig
pub const Layer = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: LayerConfig) core.FrameworkError!*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initializeWeights`

Initialize weights and biases for the layer


```zig
pub fn initializeWeights(self: *Self, rng: *std.rand.Random) core.FrameworkError!void {
```

- fn `forward`

Forward pass through the layer


```zig
pub fn forward(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
```

- fn `backward`

Backward pass through the layer


```zig
pub fn backward(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) core.FrameworkError!void {
```

- fn `getInputSize`

Get input size for the layer


```zig
pub fn getInputSize(self: *const Self) usize {
```

- fn `getOutputSize`

Get output size for the layer


```zig
pub fn getOutputSize(self: *const Self) usize {
```

- fn `setTraining`

Set training mode


```zig
pub fn setTraining(self: *Self, is_training: bool) void {
```

- fn `setInference`

Set inference mode


```zig
pub fn setInference(self: *Self) void {
```

- fn `freeze`

Freeze layer parameters


```zig
pub fn freeze(self: *Self) void {
```

- fn `unfreeze`

Unfreeze layer parameters


```zig
pub fn unfreeze(self: *Self) void {
```

- fn `resetState`

Reset layer state (useful for RNNs)


```zig
pub fn resetState(self: *Self) void {
```

- fn `getMemoryUsage`

Get memory usage of the layer


```zig
pub fn getMemoryUsage(self: *const Self) usize {
```

- fn `getParameterCount`

Get parameter count


```zig
pub fn getParameterCount(self: *const Self) usize {
```

## src\ai\localml.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `MLError`

LocalML-specific error types


```zig
pub const MLError = error{
```

- type `DataRow`

Represents a single data point with two features and a label


```zig
pub const DataRow = struct {
```

- fn `validate`

Validates that all values in the data row are finite numbers


```zig
pub fn validate(self: DataRow) MLError!void {
```

- fn `fromArray`

Creates a DataRow from an array of values
Expects exactly 3 values: [x1, x2, label]


```zig
pub fn fromArray(values: []const f64) MLError!DataRow {
```

- fn `toArray`

Converts the DataRow to an array representation


```zig
pub fn toArray(self: DataRow) [3]f64 {
```

- fn `normalize`

Creates a copy of the DataRow with normalized features


```zig
pub fn normalize(self: DataRow, x1_min: f64, x1_max: f64, x2_min: f64, x2_max: f64) DataRow {
```

- fn `distance`

Calculates the Euclidean distance between two data points


```zig
pub fn distance(self: DataRow, other: DataRow) f64 {
```

- type `Model`

A simple linear/logistic regression model


```zig
pub const Model = struct {
```

- fn `init`

Creates a new untrained model with zero-initialized parameters


```zig
pub fn init() Model {
```

- fn `initWithParams`

Creates a model with pre-initialized parameters


```zig
pub fn initWithParams(w1: f64, w2: f64, bias: f64) Model {
```

- fn `predict`

Makes a prediction for a given input
Returns the raw linear combination for regression


```zig
pub fn predict(self: Model, row: DataRow) MLError!f64 {
```

- fn `predictProba`

Makes a classification prediction using logistic function
Returns a probability between 0 and 1


```zig
pub fn predictProba(self: Model, row: DataRow) MLError!f64 {
```

- fn `train`

Trains the model using gradient descent


```zig
pub fn train(self: *Model, data: []const DataRow, learning_rate: f64, epochs: usize) MLError!void {
```

- fn `evaluate`

Evaluates the model on test data and returns mean squared error


```zig
pub fn evaluate(self: Model, test_data: []const DataRow) MLError!f64 {
```

- fn `accuracy`

Calculates classification accuracy on binary classification data


```zig
pub fn accuracy(self: Model, test_data: []const DataRow, threshold: f64) MLError!f64 {
```

- fn `reset`

Resets the model to untrained state


```zig
pub fn reset(self: *Model) void {
```

- fn `toJson`

Serializes the model to JSON format for persistence


```zig
pub fn toJson(self: Model, allocator: Allocator) ![]u8 {
```

- fn `fromJson`

Deserializes a model from JSON format


```zig
pub fn fromJson(allocator: Allocator, json_data: []const u8) !Model {
```

- type `DataProcessor`

Data preprocessing utilities


```zig
pub const DataProcessor = struct {
```

- fn `normalizeDataset`

Normalizes a dataset by scaling features to [0, 1] range


```zig
pub fn normalizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
```

- fn `trainTestSplit`

Splits dataset into training and testing sets


```zig
pub fn trainTestSplit(allocator: Allocator, data: []const DataRow, train_ratio: f64, random_seed: u64) !struct { train: []DataRow, @"test": []DataRow } {
```

- fn `standardizeDataset`

Standardizes a dataset using z-score normalization (mean=0, std=1)


```zig
pub fn standardizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
```

- type `CrossValidator`

Cross-validation utilities


```zig
pub const CrossValidator = struct {
```

- fn `kFoldValidation`

Performs k-fold cross-validation on a model


```zig
pub fn kFoldValidation(allocator: Allocator, data: []const DataRow, k: usize, learning_rate: f64, epochs: usize) !struct { mean_accuracy: f64, std_accuracy: f64 } {
```

- type `KNNClassifier`

K-Nearest Neighbors classifier for non-parametric classification


```zig
pub const KNNClassifier = struct {
```

- fn `init`

```zig
pub fn init(training_data: []const DataRow, k: usize) MLError!KNNClassifier {
```

- fn `predict`

```zig
pub fn predict(self: KNNClassifier, allocator: Allocator, query_point: DataRow) !f64 {
```

- fn `readDataset`

Reads a dataset from a CSV file
Expected format: x1,x2,label (one row per line)


```zig
pub fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]DataRow {
```

- fn `saveDataset`

Saves a dataset to a CSV file


```zig
pub fn saveDataset(path: []const u8, data: []const DataRow) !void {
```

- fn `saveModel`

Saves a trained model to file in a simple text format


```zig
pub fn saveModel(path: []const u8, model: Model) !void {
```

- fn `loadModel`

Loads a trained model from file


```zig
pub fn loadModel(path: []const u8) !Model {
```

## src\ai\mod.zig

- type `ActivationUtils`

High-performance activation function utilities


```zig
pub const ActivationUtils = struct {
```

- pub `inline`

Inline fast approximation functions for better performance


```zig
pub inline fn fastSigmoid(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastTanh(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastExp(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastGelu(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastSqrt(x: f32) f32 {
```

- pub `inline`

Vectorized activation functions with manual loop unrolling


```zig
pub inline fn vectorizedRelu(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedSigmoid(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedTanh(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedLeakyRelu(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedGelu(data: []f32) void {
```

- pub `inline`

Optimized softmax with numerical stability


```zig
pub inline fn stableSoftmax(data: []f32) void {
```

- pub `inline`

Optimized log softmax with numerical stability


```zig
pub inline fn stableLogSoftmax(data: []f32) void {
```

- type `LayerType`

Neural network layer types with enhanced coverage


```zig
pub const LayerType = enum {
```

- type `Activation`

Enhanced activation functions with optimized implementations


```zig
pub const Activation = enum {
```

- pub `inline`

Inline function for quick activation checks


```zig
pub inline fn isNonlinear(self: Activation) bool {
```

- pub `inline`

Inline function for gradient requirements


```zig
pub inline fn requiresGradient(self: Activation) bool {
```

- type `WeightInit`

Comprehensive weight initialization strategies


```zig
pub const WeightInit = enum {
```

- type `Regularization`

Advanced regularization configuration


```zig
pub const Regularization = struct {
```

- type `MemoryStrategy`

Memory allocation strategy


```zig
pub const MemoryStrategy = enum {
```

- type `ComputeBackend`

Computation backend


```zig
pub const ComputeBackend = enum {
```

- type `Layer`

Enhanced neural network layer with comprehensive functionality


```zig
pub const Layer = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, layer_type: LayerType, input_shape: []const usize, output_shape: []const usize) !*Layer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Layer, allocator: Allocator) void {
```

- fn `initializeWeights`

```zig
pub fn initializeWeights(self: *Layer, allocator: Allocator, rng: *Random) !void {
```

- fn `forward`

```zig
pub fn forward(self: *Layer, input: []const f32, output: []f32) !void {
```

- type `LossFunction`

Loss functions with comprehensive coverage


```zig
pub const LossFunction = enum {
```

- type `Optimizer`

Optimizers with state-of-the-art algorithms


```zig
pub const Optimizer = enum {
```

- type `LRScheduler`

Learning rate scheduling strategies


```zig
pub const LRScheduler = enum {
```

- type `DataAugmentation`

Data augmentation techniques


```zig
pub const DataAugmentation = struct {
```

- type `TrainingConfig`

Model training configuration with advanced options


```zig
pub const TrainingConfig = struct {
```

- type `TrainingMetrics`

Comprehensive training metrics


```zig
pub const TrainingMetrics = struct {
```

- type `NeuralNetwork`

Neural network model with enhanced capabilities


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_shape: []const usize, output_shape: []const usize) !*NeuralNetwork {
```

- fn `deinit`

```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `setTraining`

```zig
pub fn setTraining(self: *NeuralNetwork, is_training: bool) void {
```

- fn `addLayer`

```zig
pub fn addLayer(self: *NeuralNetwork, layer: *Layer) !void {
```

- fn `addDenseLayer`

```zig
pub fn addDenseLayer(self: *NeuralNetwork, units: usize, activation: ?Activation) !void {
```

- fn `addConv2DLayer`

```zig
pub fn addConv2DLayer(self: *NeuralNetwork, filters: usize, kernel_size: [2]usize, activation: ?Activation) !void {
```

- fn `addDropoutLayer`

```zig
pub fn addDropoutLayer(self: *NeuralNetwork, rate: f32) !void {
```

- fn `addBatchNormLayer`

```zig
pub fn addBatchNormLayer(self: *NeuralNetwork) !void {
```

- fn `addLSTMLayer`

```zig
pub fn addLSTMLayer(self: *NeuralNetwork, units: usize, return_sequences: bool) !void {
```

- fn `addAttentionLayer`

```zig
pub fn addAttentionLayer(self: *NeuralNetwork, num_heads: usize, head_dim: usize) !void {
```

- fn `compile`

```zig
pub fn compile(self: *NeuralNetwork) !void {
```

- fn `forward`

```zig
pub fn forward(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `predict`

```zig
pub fn predict(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `predictBatch`

```zig
pub fn predictBatch(self: *NeuralNetwork, inputs: []const []const f32, outputs: [][]f32) !void {
```

- fn `getParameterCount`

```zig
pub fn getParameterCount(self: *const NeuralNetwork) usize {
```

- fn `getMemoryUsage`

```zig
pub fn getMemoryUsage(self: *const NeuralNetwork) usize {
```

- type `EmbeddingGenerator`

Advanced embedding generator with multiple architectures


```zig
pub const EmbeddingGenerator = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize) !*EmbeddingGenerator {
```

- fn `initTransformer`

```zig
pub fn initTransformer(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize, num_heads: usize) !*EmbeddingGenerator {
```

- fn `deinit`

```zig
pub fn deinit(self: *EmbeddingGenerator) void {
```

- fn `generateEmbedding`

```zig
pub fn generateEmbedding(self: *EmbeddingGenerator, input: []const f32, embedding: []f32) !void {
```

- fn `generateEmbeddingsBatch`

```zig
pub fn generateEmbeddingsBatch(self: *EmbeddingGenerator, inputs: []const []const f32, embeddings: [][]f32) !void {
```

- fn `computeSimilarity`

```zig
pub fn computeSimilarity(self: *EmbeddingGenerator, embedding1: []const f32, embedding2: []const f32) f32 {
```

- fn `findNearestNeighbors`

```zig
pub fn findNearestNeighbors(
```

- type `ModelTrainer`

Enhanced model trainer with comprehensive optimization support


```zig
pub const ModelTrainer = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelTrainer) void {
```

- fn `train`

```zig
pub fn train(
```

- const `Network`

```zig
pub const Network = NeuralNetwork;
```

- const `Embedding`

```zig
pub const Embedding = EmbeddingGenerator;
```

- const `Neural`

```zig
pub const Neural = @import("neural.zig");
```

- const `LocalML`

```zig
pub const LocalML = @import("localml.zig");
```

- const `DynamicRouter`

```zig
pub const DynamicRouter = @import("dynamic.zig");
```

- const `DataStructures`

```zig
pub const DataStructures = @import("data_structures/mod.zig");
```

- const `Trainer`

```zig
pub const Trainer = ModelTrainer;
```

- const `Config`

```zig
pub const Config = TrainingConfig;
```

- const `Metrics`

```zig
pub const Metrics = TrainingMetrics;
```

- const `Loss`

```zig
pub const Loss = LossFunction;
```

- const `Opt`

```zig
pub const Opt = Optimizer;
```

- const `agent`

```zig
pub const agent = @import("agent.zig");
```

- const `enhanced_agent`

```zig
pub const enhanced_agent = @import("enhanced_agent.zig");
```

- fn `createMLP`

```zig
pub fn createMLP(allocator: std.mem.Allocator, layer_sizes: []const usize, activations: []const Activation) !*NeuralNetwork {
```

- fn `createCNN`

```zig
pub fn createCNN(allocator: std.mem.Allocator, input_shape: []const usize, num_classes: usize) !*NeuralNetwork {
```

## src\ai\neural.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- type `LayerType`

Neural network layer types


```zig
pub const LayerType = enum {
```

- type `Activation`

Activation functions with mixed precision support


```zig
pub const Activation = enum {
```

- fn `apply`

Apply activation function to a f32 value


```zig
pub fn apply(self: Activation, x: f32) f32 {
```

- fn `applyF16`

Apply activation function to a f16 value


```zig
pub fn applyF16(self: Activation, x: f16) f16 {
```

- fn `derivative`

Derivative of activation function (f32)


```zig
pub fn derivative(self: Activation, x: f32) f32 {
```

- fn `derivativeF16`

Derivative of activation function (f16)


```zig
pub fn derivativeF16(self: Activation, x: f16) f16 {
```

- type `Precision`

Precision mode for computations


```zig
pub const Precision = enum {
```

- type `TrainingConfig`

Neural network training configuration with enhanced memory options


```zig
pub const TrainingConfig = struct {
```

- type `LayerConfig`

Layer configuration


```zig
pub const LayerConfig = struct {
```

- type `NetworkConfig`

Complete neural network configuration


```zig
pub const NetworkConfig = struct {
```

- type `MemoryPool`

Memory pool for efficient buffer reuse


```zig
pub const MemoryPool = struct {
```

- type `PoolConfig`

Memory pool configuration


```zig
pub const PoolConfig = struct {
```

- type `PooledBuffer`

Pooled buffer


```zig
pub const PooledBuffer = struct {
```

- fn `release`

Return buffer to pool


```zig
pub fn release(self: *PooledBuffer) void {
```

- fn `slice`

Get buffer as slice of requested size


```zig
pub fn slice(self: *PooledBuffer, size: usize) []f32 {
```

- type `TrackedBuffer`

Enhanced buffer with liveness tracking


```zig
pub const TrackedBuffer = struct {
```

- fn `isStale`

Check if buffer is stale (not accessed recently)


```zig
pub fn isStale(self: TrackedBuffer, current_time: u64, stale_threshold_ns: u64) bool {
```

- fn `markAccessed`

Update access time


```zig
pub fn markAccessed(self: *TrackedBuffer, current_time: u64) void {
```

- type `LivenessConfig`

Liveness analysis configuration


```zig
pub const LivenessConfig = struct {
```

- fn `init`

Initialize memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !*MemoryPool {
```

- fn `deinit`

Deinitialize memory pool


```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate buffer from pool or create new one


```zig
pub fn allocBuffer(self: *MemoryPool, size: usize) !*PooledBuffer {
```

- fn `returnBuffer`

Return buffer to pool for reuse


```zig
pub fn returnBuffer(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `getStats`

Get pool statistics


```zig
pub fn getStats(self: *MemoryPool) struct {
```

- fn `initLivenessAnalysis`

Initialize liveness analysis


```zig
pub fn initLivenessAnalysis(self: *MemoryPool, config: LivenessConfig) void {
```

- fn `recordBufferAccess`

Record buffer access for liveness analysis


```zig
pub fn recordBufferAccess(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `performLivenessCleanup`

Perform liveness-based cleanup


```zig
pub fn performLivenessCleanup(self: *MemoryPool, current_time: u64) void {
```

- fn `getLivenessStats`

Get liveness statistics


```zig
pub fn getLivenessStats(self: *MemoryPool) struct {
```

- type `Layer`

Neural network layer with enhanced memory safety and mixed precision support


```zig
pub const Layer = struct {
```

- fn `init`

Initialize a new layer with memory pool support


```zig
pub fn init(allocator: std.mem.Allocator, config: LayerConfig, memory_pool: ?*MemoryPool) !*Layer {
```

- fn `initF16`

Initialize f16 versions for mixed precision training


```zig
pub fn initF16(self: *Layer) !void {
```

- fn `syncToF32`

Synchronize f16 weights/biases back to f32 after training


```zig
pub fn syncToF32(self: *Layer) void {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *Layer, input: []const f32, use_f16: bool) ![]f32 {
```

- fn `backwardMixed`

Backward pass with mixed precision support


```zig
pub fn backwardMixed(
```

- fn `allocBuffer`

Allocate buffer using memory pool if available, fallback to allocator


```zig
pub fn allocBuffer(self: *Layer, size: usize) ![]f32 {
```

- fn `freeBuffer`

Free buffer using memory pool if available, fallback to allocator


```zig
pub fn freeBuffer(self: *Layer, buffer: []f32) void {
```

- fn `deinit`

Free layer resources with proper cleanup


```zig
pub fn deinit(self: *Layer) void {
```

- fn `forward`

Forward pass through the layer with memory pool support


```zig
pub fn forward(self: *Layer, input: []const f32) ![]f32 {
```

- fn `backward`

Backward pass through the layer with memory pool support


```zig
pub fn backward(
```

- type `CheckpointState`

Gradient checkpointing state


```zig
pub const CheckpointState = struct {
```

- type `NeuralNetwork`

Neural network for learning embeddings with enhanced memory safety


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

Initialize a new neural network with optional memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: TrainingConfig) !*NeuralNetwork {
```

- fn `initDefault`

Initialize a new neural network with default configuration (backward compatibility)


```zig
pub fn initDefault(allocator: std.mem.Allocator) !*NeuralNetwork {
```

- fn `deinit`

Free network resources with proper cleanup


```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `deinitEnhanced`

Deinitialize with enhanced cleanup (for MemoryPool with liveness analysis)


```zig
pub fn deinitEnhanced(self: *MemoryPool) void {
```

- fn `addLayer`

Add a layer to the network with memory pool support


```zig
pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void {
```

- fn `saveToFile`

Save network to file (basic implementation)


```zig
pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void {
```

- fn `loadFromFile`

Load network from file (basic implementation)


```zig
pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork {
```

- fn `forward`

Forward pass through the network with memory optimization


```zig
pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `trainStep`

Train the network on a single sample with memory optimization


```zig
pub fn trainStep(
```

- fn `trainStepMixed`

Train the network on a single sample with mixed precision support


```zig
pub fn trainStepMixed(
```

## src\ai\data_structures\batch_queue.zig

- type `BatchQueue`

High-performance batch queue for processing data in batches


```zig
pub const BatchQueue = struct {
```

- fn `init`

Initialize a new batch queue


```zig
pub fn init(allocator: std.mem.Allocator, batch_size: usize) !*Self {
```

- fn `deinit`

Deinitialize the queue


```zig
pub fn deinit(self: *Self) void {
```

- fn `enqueue`

Add data to the queue


```zig
pub fn enqueue(self: *Self, data: []const u8) !void {
```

- fn `dequeueBatch`

Get the next batch if available


```zig
pub fn dequeueBatch(self: *Self) ?[]u8 {
```

## src\ai\data_structures\bloom_filter.zig

- type `BloomFilter`

Bloom filter for efficient set membership testing


```zig
pub const BloomFilter = struct {
```

- fn `init`

Initialize a new bloom filter


```zig
pub fn init(allocator: std.mem.Allocator, size: usize, hash_count: u32) !*Self {
```

- fn `deinit`

Deinitialize the filter


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the filter


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `contains`

Check if an item might be in the filter


```zig
pub fn contains(self: *Self, data: []const u8) bool {
```

## src\ai\data_structures\cache.zig

- fn `ThreadSafeCache`

Thread-safe LRU cache implementation


```zig
pub fn ThreadSafeCache(comptime K: type, comptime V: type) type {
```

- fn `init`

Initialize a new thread-safe cache


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the cache


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get a value from the cache


```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `put`

Put a value in the cache


```zig
pub fn put(self: *Self, key: K, value: V) !void {
```

- fn `LRUCache`

LRU Cache implementation


```zig
pub fn LRUCache(comptime K: type, comptime V: type) type {
```

- fn `init`

Initialize a new LRU cache


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the cache


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get a value from the cache


```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `put`

Put a value in the cache


```zig
pub fn put(self: *Self, key: K, value: V) !void {
```

## src\ai\data_structures\circular_buffer.zig

- fn `CircularBuffer`

High-performance circular buffer for time series data


```zig
pub fn CircularBuffer(comptime T: type) type {
```

- fn `init`

Initialize a new circular buffer


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the buffer


```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

Add an element to the buffer


```zig
pub fn push(self: *Self, value: T) void {
```

- fn `pop`

Remove and return the oldest element


```zig
pub fn pop(self: *Self) ?T {
```

- fn `RingBuffer`

Alias for CircularBuffer - RingBuffer is the same implementation


```zig
pub fn RingBuffer(comptime T: type) type {
```

## src\ai\data_structures\compressed_vector.zig

- type `CompressedVector`

Compressed vector implementation using sparse storage


```zig
pub const CompressedVector = struct {
```

- fn `init`

Initialize a new compressed vector


```zig
pub fn init(allocator: std.mem.Allocator, dimension: usize) !*Self {
```

- fn `deinit`

Deinitialize the vector


```zig
pub fn deinit(self: *Self) void {
```

- fn `set`

Set a value at the specified index


```zig
pub fn set(self: *Self, index: usize, value: f32) !void {
```

- fn `get`

Get a value at the specified index


```zig
pub fn get(self: *Self, index: usize) f32 {
```

- fn `nnz`

Get the number of non-zero elements


```zig
pub fn nnz(self: *Self) usize {
```

- fn `dot`

Calculate dot product with another compressed vector


```zig
pub fn dot(self: *Self, other: *const Self) f32 {
```

- fn `add`

Add another compressed vector to this one


```zig
pub fn add(self: *Self, other: *const Self) !void {
```

## src\ai\data_structures\dense_matrix.zig

- type `DenseMatrix`

Dense matrix implementation with contiguous memory layout


```zig
pub const DenseMatrix = struct {
```

- fn `init`

Initialize a new dense matrix


```zig
pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
```

- fn `deinit`

Deinitialize the matrix


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get element at position (i, j)


```zig
pub fn get(self: *Self, i: usize, j: usize) f32 {
```

- fn `set`

Set element at position (i, j)


```zig
pub fn set(self: *Self, i: usize, j: usize, value: f32) void {
```

- fn `getRow`

Get a row as a slice


```zig
pub fn getRow(self: *Self, i: usize) ?[]f32 {
```

- fn `getCol`

Get a column as a slice (creates a copy)


```zig
pub fn getCol(self: *Self, j: usize) ![]f32 {
```

- fn `mul`

Matrix multiplication (self * other)


```zig
pub fn mul(self: *Self, other: *const Self) !*Self {
```

- fn `add`

Element-wise addition


```zig
pub fn add(self: *Self, other: *const Self) !void {
```

- fn `mulScalar`

Scalar multiplication


```zig
pub fn mulScalar(self: *Self, scalar: f32) void {
```

## src\ai\data_structures\graph.zig

- type `Graph`

Generic graph implementation


```zig
pub const Graph = struct {
```

- fn `init`

Initialize a new graph


```zig
pub fn init(allocator: std.mem.Allocator, vertices: usize, directed: bool) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add an edge between two vertices


```zig
pub fn addEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `removeEdge`

Remove an edge between two vertices


```zig
pub fn removeEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `getNeighbors`

Get neighbors of a vertex


```zig
pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
```

- fn `bfs`

Perform breadth-first search


```zig
pub fn bfs(self: *Self, start: usize, visitor: anytype) !void {
```

- fn `dfs`

Perform depth-first search


```zig
pub fn dfs(self: *Self, start: usize, visitor: anytype) !void {
```

- type `DirectedGraph`

Directed graph (alias for Graph with directed=true)


```zig
pub const DirectedGraph = struct {
```

- fn `init`

Initialize a new directed graph


```zig
pub fn init(allocator: std.mem.Allocator, vertices: usize) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add a directed edge


```zig
pub fn addEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `getNeighbors`

Get neighbors (outgoing edges)


```zig
pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
```

- fn `getReverseNeighbors`

Get reverse neighbors (incoming edges)


```zig
pub fn getReverseNeighbors(self: *Self, vertex: usize) !std.ArrayList(usize) {
```

- type `BipartiteGraph`

Bipartite graph implementation


```zig
pub const BipartiteGraph = struct {
```

- fn `init`

Initialize a new bipartite graph


```zig
pub fn init(allocator: std.mem.Allocator, size_a: usize, size_b: usize) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add an edge between sets (only allowed between different sets)


```zig
pub fn addEdge(self: *Self, a_vertex: usize, b_vertex: usize) !void {
```

- fn `validateBipartite`

Check if the graph is bipartite (validate no edges within same set)


```zig
pub fn validateBipartite(self: *Self) bool {
```

- fn `getSetA`

Get vertices in set A


```zig
pub fn getSetA(self: *Self) std.ArrayList(usize) {
```

- fn `getSetB`

Get vertices in set B


```zig
pub fn getSetB(self: *Self) std.ArrayList(usize) {
```

## src\ai\data_structures\lockfree.zig

- const `LockFreeError`

Lock-free data structure errors


```zig
pub const LockFreeError = error{
```

- type `LockFreeStats`

Performance statistics for lock-free operations


```zig
pub const LockFreeStats = struct {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *LockFreeStats, success: bool, latency_ns: u64) void {
```

- fn `successRate`

```zig
pub fn successRate(self: *const LockFreeStats) f32 {
```

- fn `lockFreeQueue`

Lock-free queue using Michael & Scott algorithm


```zig
pub fn lockFreeQueue(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `enqueue`

```zig
pub fn enqueue(self: *Self, data: T) !void {
```

- fn `dequeue`

```zig
pub fn dequeue(self: *Self) ?T {
```

- fn `lockFreeStack`

Lock-free stack using Treiber algorithm


```zig
pub fn lockFreeStack(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

```zig
pub fn push(self: *Self, data: T) !void {
```

- fn `pop`

```zig
pub fn pop(self: *Self) ?T {
```

- fn `lockFreeHashMap`

Lock-free hash map using hopscotch hashing


```zig
pub fn lockFreeHashMap(comptime K: type, comptime V: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `put`

```zig
pub fn put(self: *Self, key: K, value: V) !bool {
```

- fn `get`

```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `workStealingDeque`

Lock-free work-stealing deque


```zig
pub fn workStealingDeque(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

```zig
pub fn push(self: *Self, item: T) bool {
```

- fn `pop`

```zig
pub fn pop(self: *Self) ?T {
```

- fn `steal`

```zig
pub fn steal(self: *Self) ?T {
```

- fn `mpmcQueue`

Multi-producer, multi-consumer queue with batching


```zig
pub fn mpmcQueue(comptime T: type, comptime capacity: usize) type {
```

- fn `init`

```zig
pub fn init() Self {
```

- fn `enqueue`

```zig
pub fn enqueue(self: *Self, item: T) bool {
```

- fn `dequeue`

```zig
pub fn dequeue(self: *Self) ?T {
```

## src\ai\data_structures\memory_pool.zig

- fn `MemoryPool`

Generic memory pool for object reuse


```zig
pub fn MemoryPool(comptime T: type) type {
```

- fn `init`

Initialize a new memory pool


```zig
pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the pool


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get an object from the pool


```zig
pub fn get(self: *Self) ?*T {
```

- fn `put`

Return an object to the pool


```zig
pub fn put(self: *Self, object: *T) void {
```

## src\ai\data_structures\mod.zig

- const `LockFreeQueue`

```zig
pub const LockFreeQueue = @import("lockfree.zig").LockFreeQueue;
```

- const `LockFreeStack`

```zig
pub const LockFreeStack = @import("lockfree.zig").LockFreeStack;
```

- const `AtomicList`

```zig
pub const AtomicList = @import("lockfree.zig").AtomicList;
```

- const `ConcurrentHashMap`

```zig
pub const ConcurrentHashMap = @import("lockfree.zig").ConcurrentHashMap;
```

- const `CircularBuffer`

```zig
pub const CircularBuffer = @import("circular_buffer.zig").CircularBuffer;
```

- const `RingBuffer`

```zig
pub const RingBuffer = @import("circular_buffer.zig").RingBuffer;
```

- const `BatchQueue`

```zig
pub const BatchQueue = @import("batch_queue.zig").BatchQueue;
```

- const `MemoryPool`

```zig
pub const MemoryPool = @import("memory_pool.zig").MemoryPool;
```

- const `ObjectPool`

```zig
pub const ObjectPool = @import("object_pool.zig").ObjectPool;
```

- const `ThreadSafeCache`

```zig
pub const ThreadSafeCache = @import("cache.zig").ThreadSafeCache;
```

- const `LRUCache`

```zig
pub const LRUCache = @import("cache.zig").LRUCache;
```

- const `BloomFilter`

```zig
pub const BloomFilter = @import("bloom_filter.zig").BloomFilter;
```

- const `CountMinSketch`

```zig
pub const CountMinSketch = @import("probabilistic.zig").CountMinSketch;
```

- const `HyperLogLog`

```zig
pub const HyperLogLog = @import("probabilistic.zig").HyperLogLog;
```

- const `VectorStore`

```zig
pub const VectorStore = @import("vector_store.zig").VectorStore;
```

- const `SparseMatrix`

```zig
pub const SparseMatrix = @import("sparse_matrix.zig").SparseMatrix;
```

- const `DenseMatrix`

```zig
pub const DenseMatrix = @import("dense_matrix.zig").DenseMatrix;
```

- const `CompressedVector`

```zig
pub const CompressedVector = @import("compressed_vector.zig").CompressedVector;
```

- const `KDTree`

```zig
pub const KDTree = @import("spatial.zig").KDTree;
```

- const `QuadTree`

```zig
pub const QuadTree = @import("spatial.zig").QuadTree;
```

- const `BallTree`

```zig
pub const BallTree = @import("spatial.zig").BallTree;
```

- const `LSHForest`

```zig
pub const LSHForest = @import("spatial.zig").LSHForest;
```

- const `Graph`

```zig
pub const Graph = @import("graph.zig").Graph;
```

- const `DirectedGraph`

```zig
pub const DirectedGraph = @import("graph.zig").DirectedGraph;
```

- const `BipartiteGraph`

```zig
pub const BipartiteGraph = @import("graph.zig").BipartiteGraph;
```

- const `TimeSeries`

```zig
pub const TimeSeries = @import("time_series.zig").TimeSeries;
```

- const `TimeSeriesBuffer`

```zig
pub const TimeSeriesBuffer = @import("time_series.zig").TimeSeriesBuffer;
```

- const `SlidingWindow`

```zig
pub const SlidingWindow = @import("sliding_window.zig").SlidingWindow;
```

- const `ExponentialMovingAverage`

```zig
pub const ExponentialMovingAverage = @import("statistics.zig").ExponentialMovingAverage;
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- type `DataStructureConfig`

Configuration for data structure initialization


```zig
pub const DataStructureConfig = struct {
```

- type `DataStructureStats`

Performance statistics for data structures


```zig
pub const DataStructureStats = struct {
```

- fn `reset`

Reset all statistics


```zig
pub fn reset(self: *DataStructureStats) void {
```

- fn `recordOperation`

Update operation statistics


```zig
pub fn recordOperation(self: *DataStructureStats, success: bool, latency_ns: u64) void {
```

- fn `createLockFreeQueue`

Initialize a lock-free queue with the specified capacity


```zig
pub fn createLockFreeQueue(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeQueue(T) {
```

- fn `createLockFreeStack`

Initialize a lock-free stack with the specified capacity


```zig
pub fn createLockFreeStack(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeStack(T) {
```

- fn `createConcurrentHashMap`

Initialize a concurrent hash map with the specified capacity


```zig
pub fn createConcurrentHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*ConcurrentHashMap(K, V) {
```

- fn `createCircularBuffer`

Initialize a circular buffer for time series data


```zig
pub fn createCircularBuffer(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*CircularBuffer(T) {
```

- fn `createMemoryPool`

Initialize a memory pool for object reuse


```zig
pub fn createMemoryPool(comptime T: type, allocator: std.mem.Allocator, pool_size: usize) !*MemoryPool(T) {
```

- fn `createLRUCache`

Initialize a thread-safe LRU cache


```zig
pub fn createLRUCache(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*LRUCache(K, V) {
```

- fn `createVectorStore`

Initialize a vector store for embedding storage and similarity search


```zig
pub fn createVectorStore(comptime T: type, allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*VectorStore(T) {
```

- fn `createKDTree`

Initialize a KD-tree for spatial indexing


```zig
pub fn createKDTree(comptime T: type, allocator: std.mem.Allocator, dimensions: usize) !*KDTree(T) {
```

- fn `createSparseMatrix`

Initialize a sparse matrix for efficient storage of sparse data


```zig
pub fn createSparseMatrix(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) !*SparseMatrix(T) {
```

- fn `createTimeSeriesBuffer`

Initialize a time series buffer with automatic windowing


```zig
pub fn createTimeSeriesBuffer(comptime T: type, allocator: std.mem.Allocator, window_size: usize) !*TimeSeriesBuffer(T) {
```

- type `DataStructureFactory`

Data structure factory for creating optimized instances based on use case


```zig
pub const DataStructureFactory = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: DataStructureConfig) DataStructureFactory {
```

- fn `createOptimizedQueue`

Create an optimized queue for the specified use case


```zig
pub fn createOptimizedQueue(self: *DataStructureFactory, comptime T: type, use_case: enum { high_throughput, low_latency, memory_efficient }) !*LockFreeQueue(T) {
```

- fn `createOptimizedCache`

Create an optimized cache for the specified access pattern


```zig
pub fn createOptimizedCache(self: *DataStructureFactory, comptime K: type, comptime V: type, access_pattern: enum { temporal, random, sequential }) !*LRUCache(K, V) {
```

- fn `getStats`

Get current statistics


```zig
pub fn getStats(self: DataStructureFactory) DataStructureStats {
```

## src\ai\data_structures\object_pool.zig

- fn `ObjectPool`

Generic object pool for type-safe object reuse


```zig
pub fn ObjectPool(comptime T: type) type {
```

- fn `init`

Initialize a new object pool


```zig
pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the pool


```zig
pub fn deinit(self: *Self) void {
```

- fn `acquire`

Get an object from the pool


```zig
pub fn acquire(self: *Self) ?*T {
```

- fn `release`

Return an object to the pool


```zig
pub fn release(self: *Self, object: *T) void {
```

## src\ai\data_structures\probabilistic.zig

- type `CountMinSketch`

Count-Min Sketch for frequency estimation


```zig
pub const CountMinSketch = struct {
```

- fn `init`

Initialize a new Count-Min Sketch


```zig
pub fn init(allocator: std.mem.Allocator, depth: usize, width: usize) !*Self {
```

- fn `deinit`

Deinitialize the sketch


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the sketch


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `estimate`

Estimate the frequency of an item


```zig
pub fn estimate(self: *Self, data: []const u8) u32 {
```

- type `HyperLogLog`

HyperLogLog for cardinality estimation


```zig
pub const HyperLogLog = struct {
```

- fn `init`

Initialize a new HyperLogLog


```zig
pub fn init(allocator: std.mem.Allocator, b: u32) !*Self {
```

- fn `deinit`

Deinitialize the HyperLogLog


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the HyperLogLog


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `estimate`

Estimate the cardinality


```zig
pub fn estimate(self: *Self) usize {
```

## src\ai\data_structures\sliding_window.zig

- type `SlidingWindow`

Sliding window data structure


```zig
pub const SlidingWindow = struct {
```

- fn `init`

Initialize a new sliding window


```zig
pub fn init(allocator: std.mem.Allocator, max_size: usize) !*Self {
```

- fn `deinit`

Deinitialize the window


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add a value to the window


```zig
pub fn add(self: *Self, value: f32) void {
```

- fn `size`

Get current window size


```zig
pub fn size(self: *Self) usize {
```

- fn `isFull`

Check if window is full


```zig
pub fn isFull(self: *Self) bool {
```

- fn `average`

Get average of current window


```zig
pub fn average(self: *Self) f32 {
```

- fn `min`

Get minimum value in window


```zig
pub fn min(self: *Self) f32 {
```

- fn `max`

Get maximum value in window


```zig
pub fn max(self: *Self) f32 {
```

- fn `stdDev`

Get standard deviation of current window


```zig
pub fn stdDev(self: *Self) f32 {
```

- fn `get`

Get value at specific index (0 = newest)


```zig
pub fn get(self: *Self, index: usize) ?f32 {
```

- fn `clear`

Clear all data from the window


```zig
pub fn clear(self: *Self) void {
```

- fn `getAll`

Get all values as a slice


```zig
pub fn getAll(self: *Self) []f32 {
```

## src\ai\data_structures\sparse_matrix.zig

- fn `SparseMatrix`

Sparse matrix implementation using COO (Coordinate) format


```zig
pub fn SparseMatrix(comptime T: type) type {
```

- fn `init`

Initialize a new sparse matrix


```zig
pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
```

- fn `deinit`

Deinitialize the matrix


```zig
pub fn deinit(self: *Self) void {
```

- fn `set`

Set a value at the specified position


```zig
pub fn set(self: *Self, row: usize, col: usize, value: T) !void {
```

- fn `get`

Get a value at the specified position


```zig
pub fn get(self: *Self, row: usize, col: usize) T {
```

- fn `nnz`

Get the number of non-zero elements


```zig
pub fn nnz(self: *Self) usize {
```

## src\ai\data_structures\spatial.zig

- fn `KDTree`

KD-tree for efficient nearest neighbor search in k-dimensional space


```zig
pub fn KDTree(comptime T: type) type {
```

- fn `init`

Initialize a new KD-tree


```zig
pub fn init(allocator: std.mem.Allocator, dimensions: usize) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- fn `insert`

Insert a point into the tree


```zig
pub fn insert(self: *Self, point: []const T) !void {
```

- fn `nearestNeighbor`

Find nearest neighbor to a query point


```zig
pub fn nearestNeighbor(self: *Self, query: []const T) !?[]T {
```

- type `QuadTree`

Quad-tree for 2D spatial indexing


```zig
pub const QuadTree = struct {
```

- fn `init`

Initialize a new quad tree


```zig
pub fn init(allocator: std.mem.Allocator, x: f32, y: f32, width: f32, height: f32, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- fn `insert`

Insert a point into the tree


```zig
pub fn insert(self: *Self, x: f32, y: f32) !void {
```

- type `BallTree`

Ball-tree for hierarchical clustering


```zig
pub const BallTree = struct {
```

- fn `init`

Initialize a new ball tree


```zig
pub fn init(allocator: std.mem.Allocator, points: []const []const f32) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- type `LSHForest`

LSH Forest for approximate nearest neighbor search


```zig
pub const LSHForest = struct {
```

- fn `init`

Initialize a new LSH forest


```zig
pub fn init(allocator: std.mem.Allocator, num_tables: usize, num_hashes: usize) !*Self {
```

- fn `deinit`

Deinitialize the forest


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add a point to the forest


```zig
pub fn add(self: *Self, point: []const f32) !void {
```

- fn `query`

Query approximate nearest neighbors


```zig
pub fn query(self: *Self, query_point: []const f32, k: usize) !std.ArrayList([]f32) {
```

## src\ai\data_structures\statistics.zig

- type `ExponentialMovingAverage`

Exponential moving average calculator


```zig
pub const ExponentialMovingAverage = struct {
```

- fn `init`

Initialize a new EMA calculator


```zig
pub fn init(allocator: std.mem.Allocator, alpha: f32) !*Self {
```

- fn `deinit`

Deinitialize the EMA calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value and update the EMA


```zig
pub fn update(self: *Self, new_value: f32) void {
```

- fn `get`

Get current EMA value


```zig
pub fn get(self: *Self) f32 {
```

- fn `reset`

Reset the EMA calculator


```zig
pub fn reset(self: *Self) void {
```

- fn `getAlpha`

Get the smoothing factor


```zig
pub fn getAlpha(self: *Self) f32 {
```

- fn `setAlpha`

Set a new smoothing factor


```zig
pub fn setAlpha(self: *Self, alpha: f32) !void {
```

- type `RunningStats`

Running statistics calculator


```zig
pub const RunningStats = struct {
```

- fn `init`

Initialize a new running statistics calculator


```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

Deinitialize the calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value


```zig
pub fn update(self: *Self, value: f32) void {
```

- fn `mean`

Get the mean


```zig
pub fn mean(self: *Self) f32 {
```

- fn `variance`

Get the variance


```zig
pub fn variance(self: *Self) f32 {
```

- fn `stdDev`

Get the standard deviation


```zig
pub fn stdDev(self: *Self) f32 {
```

- fn `min`

Get the minimum value


```zig
pub fn min(self: *Self) f32 {
```

- fn `max`

Get the maximum value


```zig
pub fn max(self: *Self) f32 {
```

- fn `range`

Get the range (max - min)


```zig
pub fn range(self: *Self) f32 {
```

- fn `reset`

Reset all statistics


```zig
pub fn reset(self: *Self) void {
```

- fn `summary`

Get a summary of all statistics


```zig
pub fn summary(self: *Self) struct {
```

- type `OnlineVariance`

Online variance calculator using Welford's method


```zig
pub const OnlineVariance = struct {
```

- fn `init`

Initialize a new online variance calculator


```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

Deinitialize the calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value


```zig
pub fn update(self: *Self, value: f32) void {
```

- fn `getMean`

Get the current mean


```zig
pub fn getMean(self: *Self) f32 {
```

- fn `getVariance`

Get the current variance


```zig
pub fn getVariance(self: *Self) f32 {
```

- fn `getStdDev`

Get the current standard deviation


```zig
pub fn getStdDev(self: *Self) f32 {
```

- fn `reset`

Reset the calculator


```zig
pub fn reset(self: *Self) void {
```

## src\ai\data_structures\time_series.zig

- type `TimeSeriesPoint`

Time series data point


```zig
pub const TimeSeriesPoint = struct {
```

- type `TimeSeriesBuffer`

Time series buffer for storing time-stamped data


```zig
pub const TimeSeriesBuffer = struct {
```

- fn `init`

Initialize a new time series buffer


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the buffer


```zig
pub fn deinit(self: *Self) void {
```

- fn `addPoint`

Add a data point


```zig
pub fn addPoint(self: *Self, timestamp: i64, value: f32) !void {
```

- fn `getValueAt`

Get value at specific timestamp (exact match)


```zig
pub fn getValueAt(self: *Self, timestamp: i64) ?f32 {
```

- fn `getInterpolatedValueAt`

Get interpolated value at timestamp


```zig
pub fn getInterpolatedValueAt(self: *Self, timestamp: i64) ?f32 {
```

- fn `getValuesInRange`

Get values in time range


```zig
pub fn getValuesInRange(self: *Self, start_time: i64, end_time: i64) !std.ArrayList(TimeSeriesPoint) {
```

- fn `simpleMovingAverage`

Calculate simple moving average


```zig
pub fn simpleMovingAverage(self: *Self, window_size: usize) !std.ArrayList(f32) {
```

- fn `getStats`

Get statistics for the time series


```zig
pub fn getStats(self: *Self) struct {
```

- const `TimeSeries`

Time series data structure (alias for TimeSeriesBuffer for backward compatibility)


```zig
pub const TimeSeries = TimeSeriesBuffer;
```

## src\ai\data_structures\vector_store.zig

- fn `VectorStore`

Vector store for embedding storage and similarity search


```zig
pub fn VectorStore(comptime T: type) type {
```

- fn `init`

Initialize a new vector store


```zig
pub fn init(allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the vector store


```zig
pub fn deinit(self: *Self) void {
```

- fn `addVector`

Add a vector to the store


```zig
pub fn addVector(self: *Self, vector: []const T) !void {
```

- fn `getVector`

Get a vector by index


```zig
pub fn getVector(self: *Self, index: usize) ?[]T {
```

- fn `cosineSimilarity`

Calculate cosine similarity between two vectors


```zig
pub fn cosineSimilarity(self: *Self, a: []const T, b: []const T) f32 {
```

